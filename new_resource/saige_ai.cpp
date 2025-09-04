#include "saige_ai.hpp"
#include <stdexcept>
#include <iostream>
#include "SAIGE_step1_fast.hpp"

namespace saige {

// ----- helpers: safe inverse (mirror your original try/catch) -----
static inline arma::fmat inv_psd_or_pinv(const arma::fmat& A) {
  try {
    return arma::inv_sympd(arma::symmatu(A));
  } catch (const std::exception&) {
    std::cout << "inv_sympd failed, inverted with pinv\n";
    return arma::pinv(arma::symmatu(A));
  }
}

// ================= Non-LOCO =================

CoefficientsOut getCoefficients_cpp(const arma::fvec& Y,
                                    const arma::fmat& X,
                                    const arma::fvec& w,
                                    const arma::fvec& tau,
                                    int maxiterPCG, float tolPCG)
{
  std::cout << "X dimensions: " << X.n_rows << " x " << X.n_cols << std::endl; 
  // Sigma^{-1}Y and Sigma^{-1}X via PCG
  arma::fvec Sigma_iY = getPCG1ofSigmaAndVector(w, tau, Y, maxiterPCG, tolPCG);                // :contentReference[oaicite:0]{index=0}
  arma::fmat Sigma_iX(Y.n_rows, X.n_cols);
  for (int j = 0; j < static_cast<int>(X.n_cols); ++j) {
    Sigma_iX.col(j) = getPCG1ofSigmaAndVector(w, tau, X.col(j), maxiterPCG, tolPCG);           // :contentReference[oaicite:1]{index=1}
  }

  // cov = (X' Σ^{-1} X)^{-1} with PSD fallback
  arma::fmat cov = inv_psd_or_pinv(X.t() * Sigma_iX);                                          // :contentReference[oaicite:2]{index=2}

  // alpha = cov * X' Σ^{-1} Y
  arma::fvec alpha = cov * (Sigma_iX.t() * Y);                                                 // :contentReference[oaicite:3]{index=3}

  // eta = Y - τ0 * (Σ^{-1}Y - Σ^{-1}X α) ./ w
  arma::fvec eta = Y - tau(0) * (Sigma_iY - Sigma_iX * alpha) / w;                             // :contentReference[oaicite:4]{index=4}

  return {Sigma_iY, Sigma_iX, cov, alpha, eta};
}

AIScoreOut getAIScore_cpp(const arma::fvec& Y,
                          const arma::fmat& X,
                          const arma::fvec& w,
                          const arma::fvec& tau,
                          const arma::fvec& Sigma_iY,
                          const arma::fmat& Sigma_iX,
                          const arma::fmat& cov,
                          int nrun, int maxiterPCG,
                          float tolPCG, float traceCVcutoff)
{
  arma::fmat Sigma_iXt = Sigma_iX.t();
  arma::fvec PY = Sigma_iY - Sigma_iX * (cov * (Sigma_iXt * Y));                                // :contentReference[oaicite:5]{index=5}
  arma::fvec APY = getCrossprodMatAndKin(PY);                                                    // :contentReference[oaicite:6]{index=6}
  float YPAPY = arma::dot(PY, APY);
  float Trace  = GetTrace(Sigma_iX, X, w, tau, cov, nrun, maxiterPCG, tolPCG, traceCVcutoff);    // :contentReference[oaicite:8]{index=8}
  arma::fvec PAPY_1 = getPCG1ofSigmaAndVector(w, tau, APY, maxiterPCG, tolPCG);                  // :contentReference[oaicite:9]{index=9}
  arma::fvec PAPY   = PAPY_1 - Sigma_iX * (cov * (Sigma_iXt * PAPY_1));
  float AI = arma::dot(APY, PAPY);                                                               // :contentReference[oaicite:10]{index=10}
  return {YPAPY, Trace, PY, AI};
}

FitAIOut fitglmmaiRPCG_cpp(const arma::fvec& Y,
                           const arma::fmat& X,
                           const arma::fvec& w,
                           arma::fvec tau,
                           const arma::fvec& Sigma_iY,
                           const arma::fmat& Sigma_iX,
                           const arma::fmat& cov,
                           int nrun, int maxiterPCG,
                           float tolPCG, float tol,
                           float traceCVcutoff)
{
  // Single AI step (caller typically loops + checks convergence in higher level)
  AIScoreOut re = getAIScore_cpp(Y, X, w, tau, Sigma_iY, Sigma_iX, cov,
                                 nrun, maxiterPCG, tolPCG, traceCVcutoff);                        // :contentReference[oaicite:11]{index=11}
  // In your R path you solve AI * Δτ = score; here AI is scalar (binary/surv, 1 VC shown)
  // Caller usually computes updated tau outside; we keep tau unchanged here and only
  // return fixed-effect updates like your R fitglmmaiRPCG did after AI step.

  arma::fmat cov_upd = inv_psd_or_pinv(X.t() * Sigma_iX);
  arma::fvec alpha   = cov_upd * (Sigma_iX.t() * Y);
  arma::fvec eta     = Y - tau(0) * (Sigma_iY - Sigma_iX * alpha) / w;

  FitAIOut out{tau, cov_upd, alpha, eta};
  return out;
}

// ================= Quantitative =================

AIScoreQOut getAIScore_q_cpp(const arma::fvec& Y,
                             arma::fmat& X,
                             arma::fvec& w,
                             arma::fvec& tau,
                             const arma::fvec& Sigma_iY,
                             const arma::fmat& Sigma_iX,
                             arma::fmat& cov,
                             int nrun, int maxiterPCG,
                             float tolPCG, float traceCVcutoff)
{
  arma::fmat Sigma_iXt = Sigma_iX.t();
  arma::fmat Xmatt     = X.t();
  arma::fmat cov1      = inv_psd_or_pinv(Xmatt * Sigma_iX);                                      // :contentReference[oaicite:12]{index=12}

  arma::fvec PY    = Sigma_iY - Sigma_iX * (cov1 * (Sigma_iXt * Y));                              // :contentReference[oaicite:13]{index=13}
  arma::fvec APY   = getCrossprodMatAndKin(PY);
  float YPAPY      = arma::dot(PY, APY);                                                          // :contentReference[oaicite:14]{index=14}
  arma::fvec A0PY  = PY;                                                                          // quantitative “A0” path
  float YPA0PY     = arma::dot(PY, A0PY);                                                         // :contentReference[oaicite:15]{index=15}

  arma::fvec Trace = GetTrace_q(Sigma_iX, X, w, tau, cov1, nrun, maxiterPCG, tolPCG, traceCVcutoff); // :contentReference[oaicite:16]{index=16}

  arma::fvec PA0PY_1 = getPCG1ofSigmaAndVector(w, tau, A0PY, maxiterPCG, tolPCG);
  arma::fvec PA0PY   = PA0PY_1 - Sigma_iX * (cov1 * (Sigma_iXt * PA0PY_1));
  arma::fvec PAPY_1  = getPCG1ofSigmaAndVector(w, tau, APY,   maxiterPCG, tolPCG);
  arma::fvec PAPY    = PAPY_1  - Sigma_iX * (cov1 * (Sigma_iXt * PAPY_1));
  arma::fmat AI(2,2, arma::fill::zeros);
  AI(0,0) = arma::dot(A0PY, PA0PY);
  AI(1,1) = arma::dot(APY,  PAPY);
  AI(0,1) = arma::dot(A0PY, PAPY);
  AI(1,0) = AI(0,1);                                                                              // :contentReference[oaicite:17]{index=17}

  return {YPA0PY, YPAPY, Trace, AI};
}

FitAIOut fitglmmaiRPCG_q_cpp(const arma::fvec& Y,
                              arma::fmat& X,
                              arma::fvec& w,
                              arma::fvec  tau,
                              const arma::fvec& Sigma_iY,
                              const arma::fmat& Sigma_iX,
                              arma::fmat& cov,
                              int nrun, int maxiterPCG,
                              float tolPCG, float /*tol*/,
                              float traceCVcutoff)
{
  AIScoreQOut re = getAIScore_q_cpp(Y, X, w, tau, Sigma_iY, Sigma_iX, cov,
                                    nrun, maxiterPCG, tolPCG, traceCVcutoff);                     // :contentReference[oaicite:18]{index=18}
  // As in your R path, do a fixed-effect update here (tau update strategy is up to caller).
  arma::fmat cov_upd = inv_psd_or_pinv(X.t() * Sigma_iX);
  arma::fvec alpha   = cov_upd * (Sigma_iX.t() * Y);
  arma::fvec eta     = Y - tau(0) * (Sigma_iY - Sigma_iX * alpha) / w;

  FitAIOut out{tau, cov_upd, alpha, eta};
  return out;
}

// ================= LOCO =================

CoefficientsOut getCoefficients_LOCO_cpp(const arma::fvec& Y,
                                         const arma::fmat& X,
                                         const arma::fvec& w,
                                         const arma::fvec& tau,
                                         int maxiterPCG, float tolPCG)
{
  arma::fvec Sigma_iY = getPCG1ofSigmaAndVector_LOCO(w, tau, Y, maxiterPCG, tolPCG);             // :contentReference[oaicite:19]{index=19}
  arma::fmat Sigma_iX(Y.n_rows, X.n_cols);
  for (int j = 0; j < static_cast<int>(X.n_cols); ++j) {
    Sigma_iX.col(j) = getPCG1ofSigmaAndVector_LOCO(w, tau, X.col(j), maxiterPCG, tolPCG);        // :contentReference[oaicite:20]{index=20}
  }
  arma::fmat cov = inv_psd_or_pinv(X.t() * Sigma_iX);                                            // :contentReference[oaicite:21]{index=21}
  arma::fvec alpha = cov * (Sigma_iX.t() * Y);
  arma::fvec eta   = Y - tau(0) * (Sigma_iY - Sigma_iX * alpha) / w;                              // :contentReference[oaicite:22]{index=22}
  return {Sigma_iY, Sigma_iX, cov, alpha, eta};
}

} // namespace saige
