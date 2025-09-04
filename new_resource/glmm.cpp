#include "glmm.hpp"
#include "saige_ai.hpp"           // Armadillo/PCG wrappers (pure C++ structs)
#include "score.hpp"   // build_score_null_binary/quant
#include "SAIGE_step1_fast.hpp"
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>
#include <string>

namespace saige {

// ---------- small helpers ----------

static inline arma::fvec to_fvec(const std::vector<double>& v) {
  arma::fvec out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[static_cast<arma::uword>(i)] = static_cast<float>(v[i]);
  return out;
}

static inline arma::fvec map_y(const Design& d) { return to_fvec(d.y); }

static inline arma::fvec map_offset(const Design& d, const std::vector<double>& offset_in) {
  if (!offset_in.empty()) return to_fvec(offset_in);
  arma::fvec out(d.n, arma::fill::zeros);
  return out;
}

static inline arma::fmat map_X_row_major_to_fmat(const Design& d) {
  const int n = d.n, p = d.p;
  arma::fmat X(n, p);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < p; ++j)
      X(i, j) = static_cast<float>(d.X[static_cast<size_t>(i)*p + j]);
  return X;
}

static inline arma::fvec map_beta_init(const Design& d, const std::vector<double>& init) {
  if (init.empty() || d.p == 0) return arma::fvec(d.p, arma::fill::zeros);
  arma::fvec out(d.p);
  for (int j = 0; j < d.p; ++j) out[j] = static_cast<float>(init[static_cast<size_t>(j)]);
  return out;
}

// relative change helper
template <typename Vec>
static inline double rel_change_inf(const Vec& a, const Vec& b, double eps) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.n_elem; ++i) {
    num = std::max(num, std::fabs(static_cast<double>(a[i] - b[i])));
    den = std::max(den, std::fabs(static_cast<double>(a[i])) + std::fabs(static_cast<double>(b[i])) + eps);
  }
  return num / den;
}

static inline double clamp_nonneg(double x) { return x < 0.0 ? 0.0 : x; }

// ---------- score-null packing into FitNullResult ----------

static inline std::vector<double> flatten_rowmajor(const arma::fmat& M) {
  std::vector<double> v(M.n_rows * M.n_cols);
  for (arma::uword i = 0; i < M.n_rows; ++i)
    for (arma::uword j = 0; j < M.n_cols; ++j)
      v[i * M.n_cols + j] = static_cast<double>(M(i, j));
  return v;
}

static void stash_score_null_into(FitNullResult& out,
                                  const saige::ScoreNull& s,
                                  int n, int p)
{
  out.obj_noK.n   = n;
  out.obj_noK.p   = p;
  out.obj_noK.V   = std::vector<double>(s.V.begin(),   s.V.end());
  out.obj_noK.S_a = std::vector<double>(s.S_a.begin(), s.S_a.end());
  out.obj_noK.XVX     = flatten_rowmajor(s.XVX);
  out.obj_noK.XVX_inv = flatten_rowmajor(s.XVX_inv);
}

// ---------- OPTIONAL: JSON exporter that supports LOCO as well ----------
// Call this from main() AFTER fit_null(...) if you want files on disk.
// It writes baseline obj_noK and, if populated, a "loco" array with per-chr packs.
static void export_score_null_json(const Paths& paths, const FitNullResult& res) {
  const std::string out_path = paths.out_prefix + ".obj_noK.json";
  std::ofstream os(out_path, std::ios::out | std::ios::trunc);
  if (!os) return;

  auto dump_pack = [&](const ScoreNullPack& p, std::ostream& s) {
    s << "{";
    s << "\"n\":" << p.n << ",\"p\":" << p.p << ",";
    s << "\"V\":[";
    for (size_t i=0;i<p.V.size();++i){ if(i) s<<","; s<<p.V[i]; }
    s << "],\"S_a\":[";
    for (size_t i=0;i<p.S_a.size();++i){ if(i) s<<","; s<<p.S_a[i]; }
    s << "],\"XVX\":[";
    for (size_t i=0;i<p.XVX.size();++i){ if(i) s<<","; s<<p.XVX[i]; }
    s << "],\"XVX_inv\":[";
    for (size_t i=0;i<p.XVX_inv.size();++i){ if(i) s<<","; s<<p.XVX_inv[i]; }
    s << "]}";
  };

  os << std::setprecision(10);
  os << "{\n  \"baseline\": ";
  dump_pack(res.obj_noK, os);

  // LOCO section (present if loco_obj_noK has any non-empty entries)
  bool any_loco = false;
  for (const auto& pk : res.loco_obj_noK) if (pk.n > 0) { any_loco = true; break; }
  if (any_loco) {
    os << ",\n  \"loco\": [\n";
    for (size_t c = 0; c < res.loco_obj_noK.size(); ++c) {
      if (res.loco_obj_noK[c].n == 0) continue;
      os << "    {\"chrom\":" << (c+1) << ",\"pack\":";
      dump_pack(res.loco_obj_noK[c], os);
      os << "}";
      // comma handling: emit comma if next non-empty exists
      size_t k = c + 1;
      while (k < res.loco_obj_noK.size() && res.loco_obj_noK[k].n == 0) ++k;
      if (k < res.loco_obj_noK.size()) os << ",";
      os << "\n";
    }
    os << "  ]\n";
  } else {
    os << "\n";
  }
  os << "}\n";
}

// ---------- family pieces (IRLS scaffolding) ----------

static inline void irls_binary_build(const arma::fvec& eta,
                                     const arma::fvec& y,
                                     const arma::fvec& offset,
                                     arma::fvec& mu,
                                     arma::fvec& mu_eta,
                                     arma::fvec& W,
                                     arma::fvec& Y) {
  mu = 1.0f / (1.0f + arma::exp(-eta));
  mu_eta = mu % (1.0f - mu);
  arma::fvec varmu = mu % (1.0f - mu);
  arma::fvec sqrtW = mu_eta / arma::sqrt(varmu + 1e-20f);
  W = sqrtW % sqrtW;                // == varmu (logistic)
  Y = eta - offset + (y - mu) / (mu_eta + 1e-20f);
}

static inline void irls_gaussian_build(const arma::fvec& eta,
                                       const arma::fvec& y,
                                       const arma::fvec& offset,
                                       arma::fvec& mu,
                                       arma::fvec& mu_eta,
                                       arma::fvec& W,
                                       arma::fvec& Y) {
  mu = eta;
  mu_eta.set_size(mu.n_elem); mu_eta.fill(1.0f);
  W.set_size(mu.n_elem);      W.fill(1.0f);
  Y = eta - offset + (y - mu) / mu_eta;   // = y - offset
}

// ---------- Binary solver (AI-REML on tau[1]; tau[0] fixed=1) ----------

inline arma::fvec sigmoid_stable_f(const arma::fvec& eta_f) {
  arma::vec eta = arma::conv_to<arma::vec>::from(eta_f);
  // clamp to avoid exp overflow/underflow in double
  eta = arma::clamp(eta, -40.0, 40.0);
  arma::vec mu = 1.0 / (1.0 + arma::exp(-eta));
  return arma::conv_to<arma::fvec>::from(mu);
}

// robust relative change: max_i |a-b| / (|a|+|b|+tol)
inline double rel_change_tau_Rstyle(const arma::fvec& a,
                                    const arma::fvec& b,
                                    float tol) {
  arma::fvec num = arma::abs(a - b);
  arma::fvec den = arma::abs(a) + arma::abs(b) + tol;
  return static_cast<double>(num.max() / den.max());
}


// FitNullResult binary_glmm_solver(const Paths& paths,
//                                  const FitNullConfig& cfg,
//                                  const Design& d,
//                                  const std::vector<double>& offset_in,
//                                  const std::vector<double>& beta_init_in)
// {
//   const int n = d.n, p = d.p;

//   // Map inputs
//   arma::fmat X = (p > 0) ? map_X_row_major_to_fmat(d) : arma::fmat(n, 0);
//   arma::fvec y = map_y(d);
//   arma::fvec offset = map_offset(d, offset_in);
//   arma::fvec beta_init = map_beta_init(d, beta_init_in);

//   // Basic shape assertions (fail fast if Design/inputs are inconsistent)
//   if ((int)X.n_rows != n) throw std::runtime_error("binary_glmm_solver: X.n_rows != n");
//   if ((int)X.n_cols != p) throw std::runtime_error("binary_glmm_solver: X.n_cols != p");
//   if ((int)y.n_elem != n) throw std::runtime_error("binary_glmm_solver: y.n_elem != n");
//   if ((int)offset.n_elem != n) throw std::runtime_error("binary_glmm_solver: offset.n_elem != n");
//   if (p > 0 && (int)beta_init.n_elem != p)
//     throw std::runtime_error("binary_glmm_solver: beta_init.n_elem != p");

//   // Initial linear predictor
//   arma::fvec eta = (p > 0) ? (X * beta_init + offset) : offset;

//   // Tuning/limits
//   const int   maxiter    = std::max(5, cfg.maxiter);
//   const float tol_coef   = static_cast<float>(std::max(1e-4, cfg.tol));
//   const int   maxiterPCG = cfg.maxiterPCG > 0 ? cfg.maxiterPCG : 500;
//   const float tolPCG     = cfg.tolPCG > 0.0 ? static_cast<float>(cfg.tolPCG) : 1e-5f;
//   const int   nrun       = cfg.nrun > 0 ? cfg.nrun : 30;
//   const float trace_cut  = cfg.traceCVcutoff > 0.0 ? static_cast<float>(cfg.traceCVcutoff) : 0.1f;

//   // Important numerical floors/caps
//   const float W_FLOOR    = 1e-6f;    // floor for IRLS weights mu*(1-mu)
//   const double AI_FLOOR  = 1e-10;    // floor for AI denominator
//   const double TAU1_CAP  = 1e6;      // conservative cap to avoid runaways

//   // Variance components (tau[0] is residual/dispersion; tau[1] random-effect VC)
//   arma::fvec tau(2); tau[0] = 1.0f; tau[1] = 0.5f;

//   // Work vectors
//   arma::fvec mu(n, arma::fill::zeros);
//   arma::fvec W(n,  arma::fill::zeros);
//   arma::fvec Y(n,  arma::fill::zeros);

//   arma::fvec alpha_prev(p, arma::fill::zeros);
//   arma::fvec tau_prev   = tau;

//   for (int it = 0; it < maxiter; ++it) {
//     // -------- (A) Update fixed effects and IRLS quantities for CURRENT tau --------
//     // We call the coefficient/PCG routine first so Sigma_iY/X/cov reflect current tau.
//     // getCoefficients_cpp should solve (X' Σ^-1 X) alpha = X' Σ^-1 Y (or equivalent)
//     // and return alpha, Sigma_iY (n), Sigma_iX (n×p), cov (p×p).
//     auto coef = getCoefficients_cpp(Y, X, W, tau, maxiterPCG, tolPCG);
//     // ^ Ensure that getCoefficients_cpp fills:
//     //   - coef.alpha (length p, possibly p==0)
//     //   - coef.Sigma_iY (length n)
//     //   - coef.Sigma_iX (n×p)
//     //   - coef.cov     (p×p)

//     if (p > 0 && (int)coef.alpha.n_elem != p)
//       throw std::runtime_error("getCoefficients_cpp: alpha.n_elem != p");
//     if ((int)coef.Sigma_iY.n_elem != n)
//       throw std::runtime_error("getCoefficients_cpp: Sigma_iY.n_elem != n");
//     if ((int)coef.Sigma_iX.n_rows != n || (int)coef.Sigma_iX.n_cols != p)
//       throw std::runtime_error("getCoefficients_cpp: Sigma_iX not n×p");
//     if ((int)coef.cov.n_rows != p || (int)coef.cov.n_cols != p)
//       throw std::runtime_error("getCoefficients_cpp: cov not p×p");

//     arma::fvec alpha = (p > 0) ? coef.alpha : arma::fvec(); // allow p==0
//     eta = (p > 0) ? (X * alpha + offset) : offset;

//     // Stable logistic and floored weights
//     mu = sigmoid_stable_f(eta);
//     W  = mu % (1.0f - mu);
//     W  = arma::clamp(W, W_FLOOR, std::numeric_limits<float>::infinity());

//     // Working response Y = eta - offset + (y - mu) / mu.eta; for logit, mu.eta = mu*(1-mu)
//     // Use the same floor in the denominator to avoid blow-ups.
//     arma::fvec mu_eta = W; // already floored mu*(1-mu)
//     Y = eta - offset + (y - mu) / mu_eta;

//     // -------- (B) Variance-component (AI/PCG) update using FRESH Y/W/etc. --------
//     auto ai = getAIScore_cpp(Y, X, W, tau,
//                              coef.Sigma_iY, coef.Sigma_iX, coef.cov,
//                              nrun, maxiterPCG, tolPCG, trace_cut);

//     // Accumulate in double and apply damping to avoid giant Newton steps
//     const double AI    = std::max(AI_FLOOR, static_cast<double>(ai.AI));
//     const double score = static_cast<double>(ai.YPAPY - ai.Trace);
//     double delta = score / AI;

//     // Simple, effective damping: shrink big steps to (0,1] scale
//     double step  = 1.0 / (1.0 + std::fabs(delta));
//     double tau1_candidate = static_cast<double>(tau[1]) + step * delta;

//     // Enforce nonnegativity + cap
//     double tau1_new = std::min(std::max(0.0, tau1_candidate), TAU1_CAP);

//     tau_prev = tau;
//     tau[1]   = static_cast<float>(tau1_new);

//     // Convergence check (use R-style metric to avoid division by ~0)
//     double rc_tau   = rel_change_tau_Rstyle(tau, tau_prev, tol_coef);
//     double rc_alpha = (p > 0) ? rel_change_tau_Rstyle(alpha, alpha_prev, tol_coef) : 0.0;

//     if (std::max(rc_tau, rc_alpha) < tol_coef) {
//       // finalize and stash score-null
//       arma::fvec mu_final = sigmoid_stable_f(eta); // use stable sigmoid for final mu
//       auto sn = saige::build_score_null_binary(X, y, mu_final);

//       FitNullResult out;
//       out.alpha  = (p > 0)
//                    ? std::vector<double>(alpha.begin(), alpha.end())
//                    : std::vector<double>();
//       out.theta  = {static_cast<double>(tau[0]), static_cast<double>(tau[1])};
//       out.offset = offset_in;

//       stash_score_null_into(out, sn, n, p);
//       // export_score_null_json(paths, out);  // optional
//       return out;
//     }

//     alpha_prev = (p > 0) ? alpha : arma::fvec(); // keep previous for next iteration
//   }

//   // -------- fallthrough: not converged within maxiter; finalize safely --------
//   {
//     // Recompute with last eta
//     mu = sigmoid_stable_f(eta);
//     W  = arma::clamp(mu % (1.0f - mu), W_FLOOR, std::numeric_limits<float>::infinity());
//     arma::fvec mu_eta = W;
//     Y = eta - offset + (y - mu) / mu_eta;

//     auto coef = getCoefficients_cpp(Y, X, W, tau, maxiterPCG, tolPCG);

//     arma::fvec mu_final = sigmoid_stable_f(eta);
//     auto sn = saige::build_score_null_binary(X, y, mu_final);

//     FitNullResult out;
//     out.alpha  = (p > 0)
//                  ? std::vector<double>(coef.alpha.begin(), coef.alpha.end())
//                  : std::vector<double>();
//     out.theta  = {static_cast<double>(tau[0]), static_cast<double>(tau[1])};
//     out.offset = offset_in;

//     stash_score_null_into(out, sn, n, p);
//     export_score_null_json(paths, out);
//     return out;
//   }
// }
FitNullResult binary_glmm_solver(const Paths& paths,
                                 const FitNullConfig& cfg,
                                 const Design& d,
                                 const std::vector<double>& offset_in,
                                 const std::vector<double>& beta_init_in)
{
  const int n = d.n, p = d.p;

  arma::fmat X = (p > 0) ? map_X_row_major_to_fmat(d) : arma::fmat(n, 0);
  arma::fvec y = map_y(d);
  arma::fvec offset = map_offset(d, offset_in);
  arma::fvec beta_init = map_beta_init(d, beta_init_in);
  arma::fvec eta = (p > 0) ? (X * beta_init + offset) : offset;

// std::cout << "shape = " << X.n_rows << " x " << X.n_cols << "\n";
// std::cout << "shape = " << y.n_rows << " y " << y.n_cols << "\n";
// std::cout << "shape = " << offset.n_rows << " offset " << offset.n_cols << "\n";
// std::cout << "shape = " << eta.n_rows << " eta " << eta.n_cols << "\n";
// std::cout << "shape = " << beta_init.n_rows << " beta_init " << beta_init.n_cols << std::endl;;

  const int   maxiter    = std::max(5, cfg.maxiter);
  const float tol_coef   = static_cast<float>(std::max(1e-6, cfg.tol));
  const int   maxiterPCG = cfg.maxiterPCG > 0 ? cfg.maxiterPCG : 500;
  const float tolPCG     = cfg.tolPCG > 0.0 ? static_cast<float>(cfg.tolPCG) : 1e-5f;
  const int   nrun       = cfg.nrun > 0 ? cfg.nrun : 30;
  const float trace_cut  = cfg.traceCVcutoff > 0.0 ? static_cast<float>(cfg.traceCVcutoff) : 0.1f;

  arma::fvec tau(2); tau[0] = 1.0f; tau[1] = 0.5f;

  arma::fvec mu, mu_eta, W, Y;
  arma::fvec alpha_prev(p, arma::fill::zeros);
  arma::fvec tau_prev   = tau;

auto check_dims = [&](const char* where){
  if ((int)X.n_rows != n) throw std::runtime_error(std::string(where)+": X.n_rows!=n");
  if ((int)X.n_cols != p) throw std::runtime_error(std::string(where)+": X.n_cols!=p");
  if ((int)y.n_elem != n) throw std::runtime_error(std::string(where)+": y.n_elem!=n");
  if ((int)offset.n_elem != n) throw std::runtime_error(std::string(where)+": offset.n_elem!=n");
  if (p>0 && (int)beta_init.n_elem != p) throw std::runtime_error(std::string(where)+": beta_init.n_elem!=p");
};
check_dims("inputs");

  for (int it = 0; it < maxiter; ++it) {
    irls_binary_build(eta, y, offset, mu, mu_eta, W, Y);

    auto coef = getCoefficients_cpp(Y, X, W, tau, maxiterPCG, tolPCG);

  auto check_coef = [&](const char* where){
  if (p>0 && (int)coef.alpha.n_elem != p) throw std::runtime_error(std::string(where)+": alpha.n_elem!=p");
  if ((int)coef.Sigma_iY.n_elem != n) throw std::runtime_error(std::string(where)+": Sigma_iY.n_elem!=n");
  if ((int)coef.Sigma_iX.n_rows != n) throw std::runtime_error(std::string(where)+": Sigma_iX.n_rows!=n");
  if ((int)coef.Sigma_iX.n_cols != p) throw std::runtime_error(std::string(where)+": Sigma_iX.n_cols!=p");
  if ((int)coef.cov.n_rows != p || (int)coef.cov.n_cols != p) throw std::runtime_error(std::string(where)+": cov not p×p");
};
check_coef("getCoefficients_cpp");

    std::cout << "Finished getCoefficients_cpp"  << std::endl;
    auto ai = getAIScore_cpp(Y, X, W, tau, coef.Sigma_iY, coef.Sigma_iX, coef.cov,
                             nrun, maxiterPCG, tolPCG, trace_cut);

if ((int)Y.n_elem != n) throw std::runtime_error("Y.n_elem!=n before AI");
if ((int)W.n_elem != n) throw std::runtime_error("W.n_elem!=n before AI");

    std::cout << "Finished getAI"  << std::endl;
    double score = static_cast<double>(ai.YPAPY - ai.Trace);
    double AI    = std::max(1e-12, static_cast<double>(ai.AI));
    double tau1_new = clamp_nonneg(static_cast<double>(tau[1]) + tau[1]* tau[1]* score / n);

    arma::fvec alpha = coef.alpha;
    eta = (p > 0) ? (X * alpha + offset) : offset;

    tau_prev = tau;  tau[1] = static_cast<float>(tau1_new);
    double rc_tau   = rel_change_inf(tau, tau_prev, tol_coef);
    double rc_alpha = (p > 0) ? rel_change_inf(alpha, alpha_prev, tol_coef) : 0.0;

    std::cout << "tau0: "  << tau_prev[1] << std::endl;
    std::cout << "tau: "  << tau[1] << std::endl;

    if (std::max(rc_tau, rc_alpha) < tol_coef) {
      // finalize + stash score-null
      arma::fvec mu_final = 1.0f / (1.0f + arma::exp(-eta));
      auto sn = saige::build_score_null_binary(X, y, mu_final);

      FitNullResult out;
      out.alpha  = std::vector<double>(alpha.begin(), alpha.end());
      out.theta  = {static_cast<double>(tau[0]), static_cast<double>(tau[1])};
      out.offset = offset_in;

      stash_score_null_into(out, sn, n, p);
      // If you still want a file on disk (incl. LOCO support when present), call:
      export_score_null_json(paths, out);
      alpha.print("alpha:");
      return out;
      std::cout << "tau0: "  << tau_prev[1] << std::endl;
    }

    alpha_prev = alpha;
  }

  // fallthrough
  irls_binary_build(eta, y, offset, mu, mu_eta, W, Y);
  auto coef = getCoefficients_cpp(Y, X, W, tau, maxiterPCG, tolPCG);

  arma::fvec mu_final = 1.0f / (1.0f + arma::exp(-eta));
  auto sn = saige::build_score_null_binary(X, y, mu_final);

  FitNullResult out;
  out.alpha  = std::vector<double>(coef.alpha.begin(), coef.alpha.end());
  out.theta  = {static_cast<double>(tau[0]), static_cast<double>(tau[1])};
  out.offset = offset_in;

  stash_score_null_into(out, sn, n, p);
  // export_score_null_json(paths, out);
  return out;
}

// ---------- Quantitative solver (2×2 AI-REML on [A0, A]) ----------

FitNullResult quant_glmm_solver(const Paths& paths,
                                const FitNullConfig& cfg,
                                const Design& d,
                                const std::vector<double>& offset_in,
                                const std::vector<double>& beta_init_in)
{
  const int n = d.n, p = d.p;

  arma::fmat X = (p > 0) ? map_X_row_major_to_fmat(d) : arma::fmat(n, 0);
  arma::fvec y = map_y(d);
  arma::fvec offset = map_offset(d, offset_in);
  arma::fvec beta_init = map_beta_init(d, beta_init_in);

  arma::fvec eta = (p > 0) ? (X * beta_init + offset) : offset;

  const int   maxiter    = std::max(5, cfg.maxiter);
  const float tol_coef   = static_cast<float>(std::max(1e-4, cfg.tol));
  const int   maxiterPCG = cfg.maxiterPCG > 0 ? cfg.maxiterPCG : 500;
  const float tolPCG     = cfg.tolPCG > 0.0 ? static_cast<float>(cfg.tolPCG) : 1e-5f;
  const int   nrun       = cfg.nrun > 0 ? cfg.nrun : 30;
  const float trace_cut  = cfg.traceCVcutoff > 0.0 ? static_cast<float>(cfg.traceCVcutoff) : 0.1f;

  arma::fvec tau(2); tau.fill(0.5f);

  arma::fvec mu, mu_eta, W, Y;
  arma::fvec alpha_prev(p, arma::fill::zeros);
  arma::fvec tau_prev   = tau;

  for (int it = 0; it < maxiter; ++it) {
    irls_gaussian_build(eta, y, offset, mu, mu_eta, W, Y); // W=1, Y=y-offset


    auto coef = getCoefficients_cpp(Y, X, W, tau, maxiterPCG, tolPCG);

    auto aiq  = getAIScore_q_cpp(Y, X, W, tau, coef.Sigma_iY, coef.Sigma_iX,
                                 const_cast<arma::fmat&>(coef.cov),
                                 nrun, maxiterPCG, tolPCG, trace_cut);

    arma::fvec s(2);
    s[0] = aiq.YPA0PY - aiq.Trace[0];
    s[1] = aiq.YPAPY  - aiq.Trace[1];

    arma::fmat AI = aiq.AI;
    if (!AI.is_sympd()) { AI = 0.5f * (AI + AI.t()); }
    arma::fvec delta = arma::solve(AI, s, arma::solve_opts::likely_sympd + arma::solve_opts::fast);

    arma::fvec tau_new = tau + delta;
    tau_new[0] = static_cast<float>(clamp_nonneg(tau_new[0]));
    tau_new[1] = static_cast<float>(clamp_nonneg(tau_new[1]));

    arma::fvec alpha = coef.alpha;
    eta = (p > 0) ? (X * alpha + offset) : offset;

    tau_prev = tau; tau = tau_new;

    double rc_tau   = rel_change_inf(tau, tau_prev, tol_coef);
    double rc_alpha = (p > 0) ? rel_change_inf(alpha, alpha_prev, tol_coef) : 0.0;
    if (std::max(rc_tau, rc_alpha) < tol_coef) {
      arma::fvec mu_final = eta;                      // identity link
      float tau0_inv = (tau[0] > 0.0f) ? 1.0f / tau[0] : 0.0f;
      auto sn = saige::build_score_null_quant(X, y, mu_final, tau0_inv);

      FitNullResult out;
      out.alpha  = std::vector<double>(alpha.begin(), alpha.end());
      out.theta  = {static_cast<double>(tau[0]), static_cast<double>(tau[1])};
      out.offset = offset_in;

      stash_score_null_into(out, sn, n, p);
      // export_score_null_json(paths, out);
      return out;
    }
    alpha_prev = alpha;
  }

  // finalize with last coefficient refresh
  irls_gaussian_build(eta, y, offset, mu, mu_eta, W, Y);
  auto coef = getCoefficients_cpp(Y, X, W, tau, maxiterPCG, tolPCG);

  arma::fvec mu_final = eta;
  float tau0_inv = (tau[0] > 0.0f) ? 1.0f / tau[0] : 0.0f;
  auto sn = saige::build_score_null_quant(X, y, mu_final, tau0_inv);

  FitNullResult out;
  out.alpha  = std::vector<double>(coef.alpha.begin(), coef.alpha.end());
  out.theta  = {static_cast<double>(tau[0]), static_cast<double>(tau[1])};
  out.offset = offset_in;

  stash_score_null_into(out, sn, n, p);
  // export_score_null_json(paths, out);
  return out;
}

// ---------- registration ----------

void register_default_solvers() {
  register_binary_solver(&binary_glmm_solver);
  register_quant_solver (&quant_glmm_solver);
}

} // namespace saige
