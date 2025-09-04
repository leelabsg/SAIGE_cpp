// null_model_engine.cpp
// ------------------------------------------------------------
// NullModelEngine: fits the null model (binary/quant/survival),
// computes offsets, and optionally runs LOCO in a single place.
// Glue points are provided to hook your existing C++ kernels
// (PCG/GLMM/LOCO) without R.
//
// Depends: Eigen3, (optional) TBB if you parallelize LOCO.
// ------------------------------------------------------------

#include "null_model_engine.hpp"
#include "saige_null.hpp"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace fs = std::filesystem;

namespace saige {

// ======= Utility: map raw vectors to Eigen without copying =======
static inline Eigen::Map<const Eigen::MatrixXd>
map_mat(const std::vector<double>& buf, int n, int p) {
  if (buf.size() != static_cast<size_t>(n) * static_cast<size_t>(p)) {
    throw std::invalid_argument("Design.X buffer size does not match n*p");
  }
  return Eigen::Map<const Eigen::MatrixXd>(buf.data(), n, p);
}

static inline Eigen::Map<const Eigen::VectorXd>
map_vec(const std::vector<double>& buf, int n) {
  if (buf.size() != static_cast<size_t>(n)) {
    throw std::invalid_argument("Vector buffer size does not match n");
  }
  return Eigen::Map<const Eigen::VectorXd>(buf.data(), n);
}

static inline Eigen::VectorXd zeros(int n) { return Eigen::VectorXd::Zero(n); }

// ======= Minimal logging helper =======
static inline void log(const std::string& s) {
  // Replace with spdlog or your logger of choice
  std::fprintf(stderr, "[NullModelEngine] %s\n", s.c_str());
}

// ======= Baseline GLM (IRLS for logistic; OLS for gaussian) =======
struct BaselineGLMOut {
  Eigen::VectorXd beta;   // p
  Eigen::VectorXd eta;    // n
  Eigen::VectorXd mu;     // n
};

static BaselineGLMOut glm_gaussian(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
  // OLS via normal equations with LDLT (robust to collinearity)
  Eigen::VectorXd beta = X.colPivHouseholderQr().solve(y);
  Eigen::VectorXd eta  = X * beta;
  return {beta, eta, eta};
}

static BaselineGLMOut glm_logistic(const Eigen::MatrixXd& X,
                                   const Eigen::VectorXd& y,
                                   int maxit, double tol) {
  const int n = static_cast<int>(X.rows());
  const int p = static_cast<int>(X.cols());
  Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd eta  = X * beta;
  Eigen::VectorXd mu   = (1.0 / (1.0 + (-eta.array()).exp())).matrix();

  double dev_prev = std::numeric_limits<double>::infinity();
  for (int it = 0; it < maxit; ++it) {
    // Weights and working response
    Eigen::VectorXd w  = (mu.array() * (1.0 - mu.array())).matrix(); // n
    // Guard against zeros in w
    for (int i = 0; i < n; ++i) { if (w[i] < 1e-12) w[i] = 1e-12; }
    Eigen::VectorXd z  = eta + (y - mu).cwiseQuotient(w);

    // Weighted least squares step
    // Solve (X' W X) beta = X' W z
    Eigen::MatrixXd Xw = X.array().colwise() * w.array(); // n x p
    Eigen::MatrixXd XtWX = X.transpose() * Xw;            // p x p
    Eigen::VectorXd XtWz = X.transpose() * (w.asDiagonal() * z);

    Eigen::VectorXd delta = XtWX.ldlt().solve(XtWz) - beta;
    beta += delta;
    eta   = X * beta;
    mu    = (1.0 / (1.0 + (-eta.array()).exp())).matrix();

    // Deviance for convergence check
    double dev = 0.0;
    for (int i = 0; i < n; ++i) {
      double yi = y[i], mui = std::min(std::max(mu[i], 1e-12), 1.0 - 1e-12);
      dev += -2.0 * (yi * std::log(mui) + (1.0 - yi) * std::log(1.0 - mui));
    }
    if (std::abs(dev - dev_prev) < tol) break;
    dev_prev = dev;
  }
  return {beta, eta, mu};
}

// ======= QR transform with back-transform support =======
struct QRMap {
  Eigen::MatrixXd R;  // rank x rank (upper tri)
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P;
  int rank{0};
  bool scaled_sqrt_n{false};
  bool valid{false};
};

// Build X_qr with optional √n scaling; extract Q from the SAME QR object.
static Eigen::MatrixXd qr_transform(const Eigen::MatrixXd& X,
                                               QRMap& map,
                                               bool scale_sqrt_n)
{
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
  const int rank = qr.rank();
  map.rank = rank;
  map.R = qr.matrixR().topLeftCorner(rank, rank).template triangularView<Eigen::Upper>();
  map.P = qr.colsPermutation();
  map.scaled_sqrt_n = scale_sqrt_n;
  map.valid = true;

  // Thin Q from the same factorization
  Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(X.rows(), rank);

  if (scale_sqrt_n) {
    Q *= std::sqrt(static_cast<double>(X.rows()));
  }
  return Q;  // n x rank
}

// Backtransform coefficients (α_qr are coefficients in the Q-basis you returned above)
static Eigen::VectorXd qr_backtransform_beta(const Eigen::VectorXd& alpha_in_basis,
                                             const QRMap& map,
                                             int p_orig, int n_rows)
{
  if (!map.valid) return alpha_in_basis;

  // If Q was scaled by √n, match that here: effective RHS to R^{-1} is √n·γ (γ = coeffs for Q√n)
  Eigen::VectorXd alpha = alpha_in_basis;
  if (map.scaled_sqrt_n) {
    alpha *= std::sqrt(static_cast<double>(n_rows));
  }

  Eigen::VectorXd x = map.R.topLeftCorner(map.rank, map.rank)
                        .triangularView<Eigen::Upper>()
                        .solve(alpha);                  // x = R^{-1} alpha

  Eigen::VectorXd alpha_full = Eigen::VectorXd::Zero(p_orig);
  alpha_full.head(map.rank) = x;

  // Undo permutation back to original column order
  return map.P * alpha_full;
}

// Optional: recover keep_cols (original indices) like your Rcpp function exposed
static std::vector<int> qr_keep_cols_original(const QRMap& map) {
  std::vector<int> keep;
  keep.reserve(map.rank);
  const auto& idx = map.P.indices();   // length = p
  for (int k = 0; k < map.rank; ++k) keep.push_back(static_cast<int>(idx[k]));
  return keep; // these are original column indices of the pivoted-first rank columns
}
// ======= External solver hooks (plug in your native kernels) =======
// You can implement these elsewhere and link, or wrap your existing functions.
// The NullModelEngine will throw if a required hook is missing.

using BinarySolverFn = FitNullResult (*)(const Paths&,
                                         const FitNullConfig&,
                                         const Design& /*design used for GLMM*/,
                                         const std::vector<double>& /*offset*/,
                                         const std::vector<double>& /*beta_init (optional)*/);

using QuantSolverFn = FitNullResult (*)(const Paths&,
                                        const FitNullConfig&,
                                        const Design&,
                                        const std::vector<double>& /*offset*/,
                                        const std::vector<double>& /*beta_init (optional)*/);

static BinarySolverFn g_binary_solver = nullptr;
static QuantSolverFn  g_quant_solver  = nullptr;

void register_binary_solver(BinarySolverFn fn) { g_binary_solver = fn; }
void register_quant_solver (QuantSolverFn  fn) { g_quant_solver  = fn; }

// ======= LOCO runner hook (optional batch implementation) =======
// Provide a single entry point that executes LOCO across chromosomes;
// If you already have a native LOCO function, wrap it and assign here.

using LocoBatchFn = void (*)(const Paths&,
                             const FitNullConfig&,
                             const LocoRanges&,
                             const Design&,
                             const std::vector<double>& /*theta*/,
                             const std::vector<double>& /*alpha*/,
                             const std::vector<double>& /*offset*/);

static LocoBatchFn g_loco_batch = nullptr;
void register_loco_batch(LocoBatchFn fn) { g_loco_batch = fn; }

// ======= NullModelEngine impl =======

NullModelEngine::NullModelEngine(const Paths& paths,
                                 const FitNullConfig& cfg,
                                 const LocoRanges& chr)
  : paths_(paths), cfg_(cfg), chr_(chr) {}

static void ensure_parent_dir(const std::string& out_path) {
  fs::path p(out_path);
  if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
}

static std::string default_model_path(const std::string& prefix) {
  return prefix + ".nullmodel.json"; // JSON stub; replace with your serializer
}

FitNullResult NullModelEngine::run(const Design& design_in) {
  if (design_in.n <= 0) throw std::invalid_argument("Design.n must be > 0");
  if (design_in.p < 0)  throw std::invalid_argument("Design.p must be >= 0");
  if (design_in.y.size() != static_cast<size_t>(design_in.n)) {
    throw std::invalid_argument("Design.y length must equal n");
  }

  // --- Map inputs ---
  Eigen::Map<const Eigen::MatrixXd> X0 = (design_in.p > 0)
    ? map_mat(design_in.X, design_in.n, design_in.p)
    : Eigen::Map<const Eigen::MatrixXd>(nullptr, 0, 0); // unused if p==0
  Eigen::Map<const Eigen::VectorXd> y  = map_vec(design_in.y, design_in.n);

  Eigen::VectorXd offset_in = design_in.offset.empty()
    ? zeros(design_in.n)
    : map_vec(design_in.offset, design_in.n);

  // --- Optional QR transform on covariates ---
  Eigen::MatrixXd X = X0;
  QRMap qrmap;
  if (cfg_.covariate_qr && design_in.p > 0) {

    X = qr_transform(X0, qrmap, design_in.n); // X becomes orthonormal columns (rank cols)


    // --- Debug print: dimensions + first few entries ---
    std::cout << "[design] after covariate QR transform: "
              << X.rows() << " x " << X.cols() << "\n";

    int max_rows = std::min<int>(5, X.rows());
    int max_cols = std::min<int>(8, X.cols());

    // column headers
    std::cout << "      ";
    for (int j = 0; j < max_cols; ++j)
        std::cout << "col" << j << "\t";
    if (X.cols() > max_cols) std::cout << "...";
    std::cout << "\n";

    // preview first few rows
    for (int i = 0; i < max_rows; ++i) {
        std::cout << "row" << i << " ";
        for (int j = 0; j < max_cols; ++j)
            std::cout << std::fixed << std::setprecision(4) << X(i, j) << "\t";
        if (X.cols() > max_cols) std::cout << "...";
        std::cout << "\n";
    }
    if (X.rows() > max_rows)
        std::cout << "... (" << X.rows() << " rows total)\n";
      //
  }

  // --- Baseline GLM (on transformed or original X) ---
  BaselineGLMOut glm;
  const bool is_binary    = (cfg_.trait == "binary" || cfg_.trait == "survival");
  const bool is_quant     = (cfg_.trait == "quantitative");
  const int  maxit_glm    = std::max(5, cfg_.maxiter);
  const double tol_glm    = std::max(1e-10, cfg_.tol);

  if (design_in.p == 0) {
    // Intercept-only baseline (eta = offset, mu = link^{-1}(eta))
    Eigen::VectorXd eta  = offset_in;
    Eigen::VectorXd mu   = is_binary ? (1.0 / (1.0 + (-eta.array()).exp())).matrix() : eta;
    glm = {Eigen::VectorXd::Zero(0), eta, mu};
  } else if (is_binary) {
    glm = glm_logistic(X, y - offset_in, maxit_glm, tol_glm); // treat offset as prior eta, subtract here
    glm.eta.array() += offset_in.array();                     // restore total eta
    glm.mu  = (1.0 / (1.0 + (-glm.eta.array()).exp())).matrix();
  } else if (is_quant) {
    glm = glm_gaussian(X, y - offset_in);                    // linear link; offset handled by subtracting
    glm.eta.array() += offset_in.array();
    glm.mu = glm.eta;
  } else {
    throw std::invalid_argument("Unsupported trait: " + cfg_.trait);
  }


  // Back-transform beta if QR was used
  Eigen::VectorXd beta_cov;
  if (design_in.p == 0) {
    beta_cov = Eigen::VectorXd::Zero(0);
  } else if (cfg_.covariate_qr) {
    beta_cov = qr_backtransform_beta(glm.beta, qrmap, design_in.p, design_in.n);
  } else {
    beta_cov = glm.beta;
  }


  // --- Compute final offset for GLMM ---
  // If covariate_offset=true, we fold fixed effects into offset and pass intercept-only to GLMM.
  // Otherwise, offset stays as provided (if any), and GLMM can refit fixed effects.
  // ensure offset_glmm has size n
  std::vector<double> offset_glmm;
  offset_glmm.reserve(design_in.n);

  if (!design_in.offset.empty()) {
      offset_glmm = design_in.offset;  // already aligned by preprocessing
  } else {
      offset_glmm.assign(design_in.n, 0.0);  // no offset given -> zeros
  }

  if (cfg_.covariate_offset) {
    // Recompute X * beta_cov on ORIGINAL design scale (X0) and add input offset
    if (design_in.p > 0) {
      Eigen::VectorXd xb = X0 * beta_cov;
      for (int i = 0; i < design_in.n; ++i) offset_glmm[i] = offset_in[i] + xb[i];
    } else {
      for (int i = 0; i < design_in.n; ++i) offset_glmm[i] = offset_in[i];
    }
  } else {
    // Keep offset as input; GLMM will estimate fixed effects again.
    for (int i = 0; i < design_in.n; ++i) offset_glmm[i] = offset_in[i];
  }

  // (optional) sanity check
  if (offset_glmm.size() != static_cast<size_t>(design_in.n)) {
      throw std::runtime_error("offset_glmm must have length n; got " +
                              std::to_string(offset_glmm.size()) + " vs n=" +
                              std::to_string(design_in.n));
  }

  // Warm start is fine to keep as-is
  std::vector<double> beta_init;
  if (design_in.p > 0) {
      beta_init.assign(beta_cov.data(), beta_cov.data() + beta_cov.size());
  }

  // --- Call GLMM solver via hooks (you plug in your existing C++ kernels) ---
  FitNullResult out;
  if (is_binary) {
    if (!g_binary_solver) {
      throw std::runtime_error("Binary/survival GLMM solver not registered. Call register_binary_solver().");
    }
    // Optional warm start: pass beta_cov to your solver if useful
    std::vector<double> beta_init;
    if (design_in.p > 0) {
      beta_init.assign(beta_cov.data(), beta_cov.data() + beta_cov.size());
    }

    out = g_binary_solver(paths_, cfg_, design_in, offset_glmm, beta_init);
    log("glmm solver Called-b") ; 
  } else {
    if (!g_quant_solver) {
      throw std::runtime_error("Quantitative GLMM solver not registered. Call register_quant_solver().");
    }
    std::vector<double> beta_init;
    if (design_in.p > 0) {
      beta_init.assign(beta_cov.data(), beta_cov.data() + beta_cov.size());
    }
    out = g_quant_solver(paths_, cfg_, design_in, offset_glmm, beta_init);
  }

  // debug
  log("glmm solver Called") ; 
  //

  // --- Attach offsets (ensure returned result has them) ---
  if (out.offset.empty()) {
    out.offset = std::move(offset_glmm);
  }
  out.loco        = cfg_.loco && chr_.enabled;
  out.lowmem_loco = cfg_.lowmem_loco;

  // --- LOCO batch (optional) ---
  if (out.loco && g_loco_batch) {
    log("Running LOCO batch");
    g_loco_batch(paths_, cfg_, chr_, design_in, out.theta, out.alpha, out.offset);
  }

  // --- Persist a compact JSON stub (optional; replace with your serializer) ---
  out.model_rda_path = default_model_path(paths_.out_prefix);
  try {
    ensure_parent_dir(out.model_rda_path);
    std::ofstream js(out.model_rda_path);
    js << "{\n";
    js << "  \"trait\": \"" << cfg_.trait << "\",\n";
    js << "  \"n\": " << design_in.n << ", \"p\": " << design_in.p << ",\n";
    js << "  \"theta\": [";
    for (size_t i = 0; i < out.theta.size(); ++i) js << (i ? "," : "") << out.theta[i];
    js << "],\n  \"alpha\": [";
    for (size_t i = 0; i < out.alpha.size(); ++i) js << (i ? "," : "") << out.alpha[i];
    js << "],\n  \"loco\": " << (out.loco ? "true" : "false")
       << ", \"lowmem_loco\": " << (out.lowmem_loco ? "true" : "false") << "\n";
    js << "}\n";
  } catch (...) {
    // Non-fatal: leave path set; upstream can decide how to handle
    log("Warning: failed to write model JSON; continuing.");
  }

  return out;
}

} // namespace saige
