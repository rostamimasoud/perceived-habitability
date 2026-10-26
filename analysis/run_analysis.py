"""
Core analysis pipeline for the manuscript on perceived habitability.

Generates a synthetic, internally consistent dataset and fits the core model
(measurement, mediation, latent profiles, global projection), and reports the
validation layers: model-implied standardized coefficients, a Monte Carlo
study of bias, root mean square error, and confidence-interval coverage, a
positive control for the mixture workflow, a positive control for the
measurement model, a bifactor positive control, and sensitivity conditions
with ordinal responses and item non-response.

All quantities are written to results.json so the manuscript can stay
self-consistent. This module contains no figure-generation code; the figures
are produced by make_figures.py. The dataset is a deterministic placeholder;
replace the data-loading block with the real survey and gridded climate data
before submission.
"""


import csv
import json
import os

import numpy as np
from matplotlib.path import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, spearmanr
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATADIR = os.path.join(ROOT, "data")





def load_land(path):
    with open(path) as f:
        gj = json.load(f)
    rings = []
    for feat in gj["features"]:
        geom = feat.get("geometry")
        if geom is None:
            continue
        coords = geom["coordinates"]
        if geom["type"] == "Polygon":
            rings.append(coords[0])
        elif geom["type"] == "MultiPolygon":
            for poly in coords:
                rings.append(poly[0])
    return rings


def land_mask(glon, glat, rings):
    """Boolean land mask on the lon/lat grid from polygon outer rings."""
    LON, LAT = np.meshgrid(glon, glat)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    inside = np.zeros(pts.shape[0], dtype=bool)
    for ring in rings:
        if len(ring) < 3:
            continue
        inside |= Path(np.asarray(ring, dtype=float)).contains_points(pts)
    return inside.reshape(LON.shape)




# ----------------------------------------------------------------------
# 1. Synthetic data: region values and the generative model
# ----------------------------------------------------------------------
REGIONS = [
    ("Bangladesh", 23.7, 90.4, 0.85), ("Philippines", 12.9, 121.8, 0.80),
    ("Niger", 17.6, 8.1, 0.82), ("Somalia", 5.2, 46.2, 0.78),
    ("Honduras", 15.2, -86.2, 0.74), ("Fiji", -17.7, 178.0, 0.72),
    ("Pakistan", 30.4, 69.3, 0.76), ("Mozambique", -18.7, 35.5, 0.70),
    ("Indonesia", -2.5, 117.9, 0.68), ("Viet Nam", 16.0, 106.3, 0.66),
    ("Egypt", 26.8, 30.8, 0.58), ("Iraq", 33.2, 43.7, 0.60),
    ("Uzbekistan", 41.4, 64.6, 0.52), ("Spain", 40.4, -3.7, 0.54),
    ("Mexico", 23.6, -102.5, 0.56), ("Peru", -9.2, -75.0, 0.50),
    ("Kenya", -0.0, 37.9, 0.55), ("Thailand", 15.9, 100.9, 0.62),
    ("Brazil", -14.2, -51.9, 0.48), ("Greece", 39.1, 21.8, 0.52),
    ("United States", 39.8, -98.6, 0.44), ("China", 35.0, 104.2, 0.50),
    ("Australia", -25.3, 133.8, 0.46), ("Italy", 41.9, 12.5, 0.42),
    ("Japan", 36.2, 138.3, 0.45), ("France", 46.2, 2.2, 0.30),
    ("United Kingdom", 55.4, -3.4, 0.26), ("Norway", 60.5, 8.5, 0.20),
    ("Canada", 56.1, -106.3, 0.18), ("New Zealand", -41.3, 174.8, 0.28),
]
region_names = [r[0] for r in REGIONS]
region_lat = np.array([r[1] for r in REGIONS])
region_lon = np.array([r[2] for r in REGIONS])
region_hazard = np.array([r[3] for r in REGIONS])
# Development / adaptive capacity, a second exogenous driver on the unit
# interval. Hand-assigned, like the hazard values, to reflect the broad global
# ordering of adaptive capacity; both are synthetic and are not drawn from a
# gridded product.
region_dev = np.array([
    0.30, 0.45, 0.12, 0.08, 0.38, 0.55, 0.35, 0.18, 0.58, 0.60,
    0.55, 0.45, 0.50, 0.85, 0.68, 0.60, 0.42, 0.65, 0.66, 0.80,
    1.00, 0.70, 0.95, 0.86, 0.95, 0.90, 0.92, 0.98, 0.95, 0.90,
])

# Generating raw item loadings (the coefficients in Y_ij = lambda_j * eta + e).
LOADINGS = np.array([0.80, 0.76, 0.72, 0.78, 0.70, 0.74])

N_REGIONS = len(REGIONS)
N_PER_REGION = 400
N = N_REGIONS * N_PER_REGION


def generate_data(rng, n_per_region, regions=REGIONS, dev=region_dev,
                  a_eff=-0.45, loadings=LOADINGS):
    """Draw one synthetic dataset from the generative model.

    The call order of ``rng`` here is fixed and identical to the single-dataset
    block, so the reported numbers are reproducible from one seed. ``a_eff``
    and ``loadings`` allow the design-sensitivity sweep to vary the effect
    size and the measurement reliability without touching the primary call.
    """
    nreg = len(regions)
    haz_r = np.array([r[3] for r in regions])
    dev_r = np.asarray(dev)
    region_idx = np.repeat(np.arange(nreg), n_per_region)
    ntot = nreg * n_per_region

    hazard = haz_r[region_idx] + rng.normal(0, 0.05, ntot)
    hazard = np.clip(hazard, 0.0, 1.0)

    devv = dev_r[region_idx] + rng.normal(0, 0.05, ntot)
    devv = np.clip(devv, 0.0, 1.0)

    self_efficacy = 0.80 * devv - 0.20 * hazard + rng.normal(0, 0.50, ntot)
    social_support = 0.65 * devv + rng.normal(0, 0.50, ntot)
    threat_appraisal = 0.45 * hazard + rng.normal(0, 0.60, ntot)

    ph = (a_eff * hazard + 0.25 * self_efficacy + 0.20 * social_support
          + rng.normal(0, 0.6, ntot))
    distress = (0.35 * hazard - 0.40 * ph + 0.25 * threat_appraisal
                + rng.normal(0, 0.6, ntot))
    adapt = (0.30 * ph - 0.20 * distress + 0.15 * hazard
             + rng.normal(0, 0.7, ntot))

    ph_items = np.column_stack(
        [l * ph + rng.normal(0, np.sqrt(1 - l ** 2), ntot) for l in loadings])

    return {
        "region_idx": region_idx, "nreg": nreg, "ntot": ntot,
        "hazard": hazard, "dev": devv,
        "self_efficacy": self_efficacy, "social_support": social_support,
        "threat_appraisal": threat_appraisal, "ph": ph, "distress": distress,
        "adapt": adapt, "ph_items": ph_items,
    }


# Single dataset with the fixed seed (reproduces all reported numbers).
rng_main = np.random.default_rng(20261025)
data = generate_data(rng_main, N_PER_REGION)
region_idx = data["region_idx"]
hazard = data["hazard"]
dev = data["dev"]
self_efficacy = data["self_efficacy"]
social_support = data["social_support"]
threat_appraisal = data["threat_appraisal"]
ph = data["ph"]
distress = data["distress"]
adapt = data["adapt"]
ph_items = data["ph_items"]


# ----------------------------------------------------------------------
# Small helpers: OLS on standardized variables
# ----------------------------------------------------------------------
def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    r2 = 1 - np.var(resid, ddof=1) / np.var(y, ddof=1)
    return beta, r2


def ols_with_se(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    n, p = X.shape
    s2 = float(resid @ resid) / (n - p)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    return beta, se


def pca_first_loadings(items):
    Z = StandardScaler().fit_transform(items)
    n = len(Z)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    L = Vt[0] * np.sqrt(S[0] ** 2 / (n - 1))
    if L[0] < 0:
        L = -L
    return L


def ml_cfa_loadings(items):
    """One-factor ML CFA on standardized items; returns loadings and
    uniquenesses. The ML standardized loading is the model-implied correlation
    between an item and the factor, which accounts for unequal unique variances
    (the first principal loading does so only when the unique variances are
    equal)."""
    Z = StandardScaler().fit_transform(items)
    k = Z.shape[1]
    S = np.cov(Z, rowvar=False, bias=False)
    _, logdet_s = np.linalg.slogdet(S)
    init = pca_first_loadings(items)

    def fml(params):
        lam = params[:k]
        theta = np.exp(params[k:])
        sig = np.outer(lam, lam) + np.diag(theta)
        sign, logdet = np.linalg.slogdet(sig)
        if sign <= 0:
            return 1e6
        return logdet + np.trace(S @ np.linalg.inv(sig)) - logdet_s - k

    opt = minimize(fml, np.concatenate([init, np.log(1 - init ** 2)]),
                   method="BFGS")
    return opt.x[:k], np.exp(opt.x[k:])


def factor_score_reliability(lam, theta):
    """Reliability of the regression factor score in a one-factor model, the
    squared correlation between the estimated score and the latent factor."""
    s = float(np.sum(lam ** 2 / theta))
    return s / (1.0 + s)


def fit_mediation_std(hazard_, se_, ss_, ta_, ph_, distress_, adapt_):
    """Standardized path estimates and their OLS/delta-method standard errors."""
    cols = np.column_stack([hazard_, se_, ss_, ta_, ph_, distress_, adapt_])
    Z = StandardScaler().fit_transform(cols)
    Zh, Zse, Zss, Zta, Zph, Zdi, Zad = Z.T

    ba, sea = ols_with_se(Zph, np.column_stack([Zh, Zse, Zss]))
    bb, seb = ols_with_se(Zdi, np.column_stack([Zh, Zph, Zta]))
    bt, _ = ols_with_se(Zdi, np.column_stack([Zh, Zta]))
    bad, sead = ols_with_se(Zad, np.column_stack([Zph, Zdi, Zh]))

    a = ba[1]
    b = bb[2]
    c_direct = bb[1]
    c_total = bt[1]
    indirect = a * b
    # First-order delta-method standard error of the product (Sobel form).
    se_ind = np.sqrt(a ** 2 * seb[2] ** 2 + b ** 2 * sea[1] ** 2)

    def r2(y, X, beta):
        X1 = np.column_stack([np.ones(len(y)), X])
        return 1 - np.var(y - X1 @ beta, ddof=1) / np.var(y, ddof=1)

    return {
        "a": a, "b": b, "c_direct": c_direct, "c_total": c_total,
        "indirect": indirect,
        "se_coef": ba[2], "ss_coef": ba[3], "ta_coef": bb[3],
        "adapt_ph": bad[1], "adapt_di": bad[2], "adapt_h": bad[3],
        "se_a": sea[1], "se_b": seb[2], "se_c": seb[1], "se_ind": se_ind,
        "r2_ph": r2(Zph, np.column_stack([Zh, Zse, Zss]), ba),
        "r2_distress": r2(Zdi, np.column_stack([Zh, Zph, Zta]), bb),
        "r2_adapt": r2(Zad, np.column_stack([Zph, Zdi, Zh]), bad),
    }


# ----------------------------------------------------------------------
# 2. Measurement (single dataset)
# ----------------------------------------------------------------------
def cronbach_alpha(X):
    k = X.shape[1]
    var_sum = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - var_sum / total_var)


ph_alpha = cronbach_alpha(ph_items)
ph_item_mean = ph_items.mean(axis=0)

# Recovered standardized loadings: one-factor ML CFA on the standardized items.
Z = StandardScaler().fit_transform(ph_items)
lam_ml, theta_ml = ml_cfa_loadings(ph_items)
recovered_loadings = lam_ml
ph_omega = (lam_ml.sum() ** 2) / (lam_ml.sum() ** 2 + theta_ml.sum())
ph_fs_rel = factor_score_reliability(lam_ml, theta_ml)

# One-factor CFA fit indices
S_items = np.cov(Z, rowvar=False, bias=False)
_, logdet_s = np.linalg.slogdet(S_items)
k_items = Z.shape[1]
sig_hat = np.outer(lam_ml, lam_ml) + np.diag(theta_ml)
_, logdet_sig = np.linalg.slogdet(sig_hat)
F_ML = float(logdet_sig + np.trace(S_items @ np.linalg.inv(sig_hat))
             - logdet_s - k_items)
df_cfa = k_items * (k_items + 1) // 2 - 2 * k_items   # 21 - 12 = 9
chi2 = (N - 1) * F_ML
rmsea = float(np.sqrt(max(F_ML / df_cfa - 1 / (N - 1), 0)))
resid = (S_items - sig_hat) / np.sqrt(np.outer(np.diag(S_items),
                                               np.diag(S_items)))
srmr = float(np.sqrt(np.mean(resid[np.triu_indices(k_items, 1)] ** 2)))
S_null = np.diag(np.diag(S_items))
_, logdet0 = np.linalg.slogdet(S_null)
F_null = float(logdet0 + np.trace(S_items @ np.linalg.inv(S_null))
               - logdet_s - k_items)
chi2_null = (N - 1) * F_null
df_null = k_items * (k_items + 1) // 2 - k_items      # 21 - 6 = 15
cfi = float(1 - max(chi2 - df_cfa, 0) / max(chi2_null - df_null, 0))

# ----------------------------------------------------------------------
# 3. Regression / mediation (single dataset, all standardized)
# ----------------------------------------------------------------------
cols = np.column_stack([hazard, self_efficacy, social_support,
                        threat_appraisal, ph, distress, adapt])
scaler = StandardScaler()
Zcols = scaler.fit_transform(cols)
Zh, Zse, Zss, Zta, Zph, Zdi, Zad = Zcols.T

# Path a: hazard -> PH (controlling self-efficacy, social support)
beta_a, r2_ph = ols(Zph, np.column_stack([Zh, Zse, Zss]))
a = beta_a[1]
# Path b and direct c': hazard -> distress (controlling PH, threat appraisal)
beta_b, r2_distress = ols(Zdi, np.column_stack([Zh, Zph, Zta]))
c_direct = beta_b[1]
b = beta_b[2]
# Total effect: hazard -> distress (no PH)
beta_tot, _ = ols(Zdi, np.column_stack([Zh, Zta]))
c_total = beta_tot[1]
indirect = a * b

# Bootstrap confidence intervals (respondent-level resampling)
B = 2000
ind_bs = np.empty(B)
prop_med_bs = np.empty(B)
boot = {k: np.empty(B) for k in
        ("a", "se", "ss", "c_direct", "b", "ta", "c_total",
         "adapt_ph", "adapt_di", "adapt_h")}
boot_loadings = np.empty((B, 6))
r2_bs = np.empty((B, 3))
for i in range(B):
    idx = rng_main.integers(0, N, N)
    ba, r2a = ols(Zph[idx], np.column_stack([Zh[idx], Zse[idx], Zss[idx]]))
    bb, r2b = ols(Zdi[idx], np.column_stack([Zh[idx], Zph[idx], Zta[idx]]))
    bt, _ = ols(Zdi[idx], np.column_stack([Zh[idx], Zta[idx]]))
    bad, r2ad = ols(Zad[idx], np.column_stack([Zph[idx], Zdi[idx], Zh[idx]]))
    r2_bs[i] = [r2a, r2b, r2ad]
    ind_bs[i] = ba[1] * bb[2]
    prop_med_bs[i] = (ba[1] * bb[2]) / bt[1]
    boot["a"][i] = ba[1]
    boot["se"][i] = ba[2]
    boot["ss"][i] = ba[3]
    boot["c_direct"][i] = bb[1]
    boot["b"][i] = bb[2]
    boot["ta"][i] = bb[3]
    boot["c_total"][i] = bt[1]
    boot["adapt_ph"][i] = bad[1]
    boot["adapt_di"][i] = bad[2]
    boot["adapt_h"][i] = bad[3]
    Zi = Z[idx]
    boot_loadings[i] = ml_cfa_loadings(ph_items[idx])[0]
ci_lo, ci_hi = np.percentile(ind_bs, [2.5, 97.5])
prop_med_lo, prop_med_hi = np.percentile(prop_med_bs, [2.5, 97.5])
loadings_ci_lo = np.percentile(boot_loadings, 2.5, axis=0)
loadings_ci_hi = np.percentile(boot_loadings, 97.5, axis=0)
r2_ci = [[round(float(np.percentile(r2_bs[:, j], 2.5)), 3),
          round(float(np.percentile(r2_bs[:, j], 97.5)), 3)]
         for j in range(3)]


def boot_ci(name):
    return [round(float(np.percentile(boot[name], 2.5)), 3),
            round(float(np.percentile(boot[name], 97.5)), 3)]


# Cluster bootstrap: resample regions, keep respondents within each region
region_members = {r: np.where(region_idx == r)[0] for r in np.arange(N_REGIONS)}
ind_bs_cluster = np.empty(B)
boot_cluster = {k: np.empty(B) for k in
                ("a", "se", "ss", "c_direct", "c_total", "b", "ta",
                 "adapt_ph", "adapt_di", "adapt_h")}
for i in range(B):
    rsel = rng_main.choice(np.arange(N_REGIONS), size=N_REGIONS, replace=True)
    idx = np.concatenate([region_members[r] for r in rsel])
    a_hat, _ = ols(Zph[idx], np.column_stack([Zh[idx], Zse[idx], Zss[idx]]))
    b_hat, _ = ols(Zdi[idx], np.column_stack([Zh[idx], Zph[idx], Zta[idx]]))
    bt_hat, _ = ols(Zdi[idx], np.column_stack([Zh[idx], Zta[idx]]))
    bad_hat, _ = ols(Zad[idx], np.column_stack([Zph[idx], Zdi[idx], Zh[idx]]))
    ind_bs_cluster[i] = a_hat[1] * b_hat[2]
    boot_cluster["a"][i] = a_hat[1]
    boot_cluster["se"][i] = a_hat[2]
    boot_cluster["ss"][i] = a_hat[3]
    boot_cluster["c_direct"][i] = b_hat[1]
    boot_cluster["c_total"][i] = bt_hat[1]
    boot_cluster["b"][i] = b_hat[2]
    boot_cluster["ta"][i] = b_hat[3]
    boot_cluster["adapt_ph"][i] = bad_hat[1]
    boot_cluster["adapt_di"][i] = bad_hat[2]
    boot_cluster["adapt_h"][i] = bad_hat[3]
ci_lo_cluster, ci_hi_cluster = np.percentile(ind_bs_cluster, [2.5, 97.5])


def cluster_ci(name):
    return [round(float(np.percentile(boot_cluster[name], 2.5)), 3),
            round(float(np.percentile(boot_cluster[name], 97.5)), 3)]

# Bootstrap confidence intervals for the reliability coefficients
alpha_bs = np.empty(B)
omega_bs = np.empty(B)
for i in range(B):
    idx = rng_main.integers(0, N, N)
    Xi = ph_items[idx]
    alpha_bs[i] = cronbach_alpha(Xi)
    L, Th = ml_cfa_loadings(Xi)
    omega_bs[i] = (L.sum() ** 2) / (L.sum() ** 2 + Th.sum())
ph_alpha_ci = [round(float(np.percentile(alpha_bs, 2.5)), 3),
               round(float(np.percentile(alpha_bs, 97.5)), 3)]
ph_omega_ci = [round(float(np.percentile(omega_bs, 2.5)), 3),
               round(float(np.percentile(omega_bs, 97.5)), 3)]

# Adaptation model: adapt ~ PH + distress + hazard
beta_adapt, r2_adapt = ols(Zad, np.column_stack([Zph, Zdi, Zh]))


# Intraclass correlations ICC(1) for the clustered outcomes
def icc1(x, cluster):
    cats = np.unique(cluster)
    ssb = sum(len(x[cluster == c]) * (x[cluster == c].mean() - x.mean()) ** 2
              for c in cats)
    ssw = sum(((x[cluster == c] - x[cluster == c].mean()) ** 2).sum()
              for c in cats)
    k = len(x) / len(cats)
    msb = ssb / (len(cats) - 1)
    msw = ssw / (len(x) - len(cats))
    return (msb - msw) / (msb + (k - 1) * msw)


icc_ph = round(float(icc1(Zph, region_idx)), 3)
icc_distress = round(float(icc1(Zdi, region_idx)), 3)
icc_adapt = round(float(icc1(Zad, region_idx)), 3)
# Proportion of the total hazard-distress association carried by habitability
prop_mediated = round(float(indirect / c_total), 3)

# Factor-score sensitivity: the regression factor score estimated from the six
# items, used to check attenuation when the structural model is run on an
# estimated rather than the latent score.
ph_hat = Z @ np.linalg.solve(S_items, lam_ml)
ph_hat = (ph_hat - ph_hat.mean()) / ph_hat.std(ddof=1)
m_fs = fit_mediation_std(hazard, self_efficacy, social_support,
                         threat_appraisal, ph_hat, distress, adapt)

# Residual ICC after removing the modelled covariates: the design effect of the
# mediation paths is set by the clustering that remains after conditioning.
resid_ph = Zph - np.column_stack([np.ones(N), Zh, Zse, Zss]) @ beta_a
resid_di = Zdi - np.column_stack([np.ones(N), Zh, Zph, Zta]) @ beta_b
icc_ph_resid = round(float(icc1(resid_ph, region_idx)), 3)
icc_di_resid = round(float(icc1(resid_di, region_idx)), 3)

# ----------------------------------------------------------------------
# 4. Latent profiles (Gaussian mixture on standardized PH and distress)
# ----------------------------------------------------------------------
prof_data = np.column_stack([Zph, Zdi])
bics = []
models = {}
for k in range(1, 7):
    g = GaussianMixture(n_components=k, covariance_type="full",
                        random_state=0, n_init=10)
    g.fit(prof_data)
    bics.append(g.bic(prof_data))
    models[k] = g
k_best = int(np.argmin(bics)) + 1
gmm = models[k_best]
labels = gmm.predict(prof_data)
resp = gmm.predict_proba(prof_data)
if k_best == 1:
    entropy = 1.0
else:
    entropy = 1 - (-np.sum(resp * np.log(resp + 1e-12)) / (N * np.log(k_best)))

# Two-component diagnostics (reported as a poorly separated solution)
gmm2 = models[2]
labels2 = gmm2.predict(prof_data)
resp2 = gmm2.predict_proba(prof_data)
entropy2 = 1 - (-np.sum(resp2 * np.log(resp2 + 1e-12)) / (N * np.log(2)))
means2 = np.array([prof_data[labels2 == k].mean(axis=0) for k in range(2)])
order = np.argsort(means2[:, 0])
remap = {old: new for new, old in enumerate(order)}
labels2 = np.array([remap[l] for l in labels2])
resp2 = np.column_stack([resp2[:, order[0]], resp2[:, order[1]]])
profile_means = means2[order]
profile_covs = gmm2.covariances_[order]
class_sizes = np.array([np.mean(labels2 == k) for k in range(2)])
avg_post = np.array([resp2[labels2 == k, k].mean() for k in range(2)])
profile_adapt = np.array([Zad[labels2 == k].mean() for k in range(2)])
profile_hazard = np.array([Zh[labels2 == k].mean() for k in range(2)])

# Null calibration of the mixture workflow across replications: the number of
# components selected when the truth is a single continuous Gaussian.
R_NULL = 200
null_k = np.empty(R_NULL, dtype=int)
for r in range(R_NULL):
    rr = np.random.default_rng(20261025 + 200000 + r)
    d_null = generate_data(rr, N_PER_REGION)
    cols_null = np.column_stack([d_null["ph"], d_null["distress"]])
    cols_null = StandardScaler().fit_transform(cols_null)
    bics_null = []
    for k in range(1, 7):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=0, n_init=5)
        g.fit(cols_null)
        bics_null.append(g.bic(cols_null))
    null_k[r] = int(np.argmin(bics_null)) + 1
null_k_prop1 = round(float(np.mean(null_k == 1)), 3)
null_k_mean = round(float(np.mean(null_k)), 3)

# ----------------------------------------------------------------------
# 5. Global gridded fields (multi-driver, population-level projection)
# ----------------------------------------------------------------------
glat = np.linspace(-60, 78, 140)
glon = np.linspace(-180, 180, 280)
LON, LAT = np.meshgrid(glon, glat)


def gauss2d(lon0, lat0, slon, slat, amp):
    """Two-dimensional Gaussian kernel centred at a spatial node."""
    d = ((LON - lon0) / slon) ** 2 + ((LAT - lat0) / slat) ** 2
    return amp * np.exp(-d)


hazard_field = (
    gauss2d(25.0, 20.0, 40.0, 15.0, 1.00)      # Sahel and Middle East
    + gauss2d(80.0, 22.0, 32.0, 14.0, 0.95)    # South Asia
    + gauss2d(40.0, 8.0, 28.0, 12.0, 0.85)     # Horn of Africa
    + gauss2d(-105.0, 30.0, 38.0, 14.0, 0.80)  # North American arid belt
    + gauss2d(135.0, -24.0, 34.0, 16.0, 0.75)  # interior Australia
    + gauss2d(-60.0, -2.0, 42.0, 14.0, 0.70)   # equatorial South America
    + 0.28 * np.exp(-((LAT - 12.0) / 30.0) ** 2)
    * (1.0 + 0.45 * np.cos(np.deg2rad(LON)))
)
hazard_field = hazard_field + rng_main.normal(0, 0.03, hazard_field.shape)
hazard_field = np.clip(hazard_field / hazard_field.max(), 0, 1)

dev_field = (
    gauss2d(-95.0, 45.0, 45.0, 18.0, 1.00)    # North America
    + gauss2d(10.0, 50.0, 32.0, 14.0, 0.95)   # Europe
    + gauss2d(120.0, 33.0, 28.0, 14.0, 0.90)  # East Asia
    + gauss2d(145.0, -32.0, 30.0, 16.0, 0.80)  # Oceania
    + gauss2d(-50.0, -15.0, 40.0, 15.0, 0.65)  # southern South America
)
dev_field = dev_field + rng_main.normal(0, 0.03, dev_field.shape)
dev_field = np.clip(dev_field / dev_field.max(), 0, 1)

se_field = 0.80 * dev_field - 0.20 * hazard_field
ss_field = 0.65 * dev_field
ta_field = 0.45 * hazard_field


def distress_implied(h):
    """Location-mean distress implied by the generative structural equations."""
    se = 0.80 * dev_field - 0.20 * h
    ss = 0.65 * dev_field
    ta = 0.45 * h
    ph = -0.45 * h + 0.25 * se + 0.20 * ss
    return 0.35 * h - 0.40 * ph + 0.25 * ta


ph_field_raw = -0.45 * hazard_field + 0.25 * se_field + 0.20 * ss_field
distress_field_raw = distress_implied(hazard_field)

LAND = load_land(os.path.join(DATADIR, "world_land.geojson"))
land = land_mask(glon, glat, LAND)
ocean = ~land
for arr in (hazard_field, dev_field, se_field, ss_field, ta_field,
            ph_field_raw, distress_field_raw):
    arr[ocean] = np.nan

sigma_within = np.sqrt(0.16 * 0.36 + 0.36)   # (-0.40)^2 Var(e1) + Var(e2), Var(e)=0.6^2
tau = 1.0                                    # elevated-distress threshold


def risk_from_distress(mu):
    return 1.0 - norm.cdf((tau - mu) / sigma_within)


current_risk = risk_from_distress(distress_field_raw)
risk_mean = float(np.nanmean(current_risk))

ph_field = (ph_field_raw - np.nanmin(ph_field_raw)) \
    / (np.nanmax(ph_field_raw) - np.nanmin(ph_field_raw))
distress_field = (distress_field_raw - np.nanmin(distress_field_raw)) \
    / (np.nanmax(distress_field_raw) - np.nanmin(distress_field_raw))
current_risk_field = current_risk

future_hazard = np.clip(hazard_field * 1.35, 0, 1)
future_risk = risk_from_distress(distress_implied(future_hazard))
risk_field = future_risk

risk_mean_increase = round(float(np.nanmean(future_risk) - risk_mean), 4)
risk_increase = {
    lam: round(float(np.nanmean(risk_from_distress(
        distress_implied(np.clip(hazard_field * lam, 0, 1)))) - risk_mean), 4)
    for lam in (1.15, 1.25, 1.35, 1.50)
}

# ----------------------------------------------------------------------
# 6. Regional summary means
# ----------------------------------------------------------------------
region_ph = np.array([Zph[region_idx == k].mean() for k in range(N_REGIONS)])
region_distress = np.array([Zdi[region_idx == k].mean() for k in range(N_REGIONS)])



















# ----------------------------------------------------------------------
# 6b. Model-implied standardized coefficients (the recovery targets)
# ----------------------------------------------------------------------
# A large sample of the same generating model gives the population standardized
# OLS coefficients. The single-dataset estimates are standardized, so these
# (not the raw generating coefficients) are the values recovery should match.
rng_implied = np.random.default_rng(20261025 + 9999)
data_imp = generate_data(rng_implied, 20000)
imp = fit_mediation_std(data_imp["hazard"], data_imp["self_efficacy"],
                        data_imp["social_support"], data_imp["threat_appraisal"],
                        data_imp["ph"], data_imp["distress"], data_imp["adapt"])
implied_loadings = ml_cfa_loadings(data_imp["ph_items"])[0]

# ----------------------------------------------------------------------
# 6b'. Analytic standardized coefficients (closed form, estimator-free)
# ----------------------------------------------------------------------
# The reduced form of the generating equations expresses each variable as a
# linear combination of the two exogenous drivers (H, V) and six independent
# disturbances. Forming the population covariance from these coefficients,
# standardizing, and solving the same partial-regression normal equations
# gives the analytic standardized targets independently of any estimator.
# These should reproduce the simulation-derived model-implied values above,
# which validates the target and removes the circularity of using the
# estimator to define its own recovery target.
def analytic_standardized():
    # Rows: H, V, SE, SS, TA, eta, D, A. Columns: H, V, u1, u2, u3, e1, e2, e3.
    A = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.20, 0.80, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.65, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.45, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [-0.50, 0.33, 0.25, 0.20, 0.0, 1.0, 0.0, 0.0],
        [0.6625, -0.132, -0.10, -0.08, 0.25, -0.40, 1.0, 0.0],
        [-0.1325, 0.1254, 0.095, 0.076, -0.05, 0.38, -0.20, 1.0],
    ])
    vh = float(np.var(region_hazard, ddof=0)) + 0.05 ** 2
    vv = float(np.var(region_dev, ddof=0)) + 0.05 ** 2
    chv = float(np.cov(region_hazard, region_dev, ddof=0)[0, 1])
    Cb = np.diag([vh, vv, 0.25, 0.25, 0.36, 0.36, 0.36, 0.49])
    Cb[0, 1] = Cb[1, 0] = chv
    Sigma = A @ Cb @ A.T
    sd = np.sqrt(np.diag(Sigma))
    R = Sigma / np.outer(sd, sd)

    def partial(y, xcols):
        return np.linalg.solve(R[np.ix_(xcols, xcols)], R[xcols, y])

    ba = partial(5, [0, 2, 3])          # eta ~ H, SE, SS
    bb = partial(6, [0, 5, 4])          # D ~ H, eta, TA
    a_t = ba[0]
    b_t = bb[1]
    c_direct_t = bb[0]
    c_total_t = partial(6, [0, 4])[0]   # D ~ H, TA
    se_t = ba[1]
    ss_t = ba[2]
    ta_t = bb[2]
    av = partial(7, [5, 6, 0])          # A ~ eta, D, H
    adapt_ph_t, adapt_di_t, adapt_h_t = av[0], av[1], av[2]
    # Standardized loadings: the model-implied correlation between item j and
    # the factor, lambda_j * sigma_eta / sigma_{y_j}, where sigma_{y_j} is the
    # item standard deviation implied by the generating model.
    lam = LOADINGS
    var_eta = Sigma[5, 5]
    sd_items = np.sqrt(lam ** 2 * var_eta + (1 - lam ** 2))
    loadings_t = lam * np.sqrt(var_eta) / sd_items
    return {
        "a": a_t, "b": b_t, "c_direct": c_direct_t, "c_total": c_total_t,
        "indirect": a_t * b_t, "se_coef": se_t, "ss_coef": ss_t,
        "ta_coef": ta_t, "adapt_ph": adapt_ph_t, "adapt_di": adapt_di_t,
        "adapt_h": adapt_h_t, "loadings": loadings_t,
    }


analytic = analytic_standardized()
analytic_vals = [analytic["a"], analytic["b"], analytic["c_direct"],
                 analytic["indirect"], *list(analytic["loadings"])]
implied_vals = [imp["a"], imp["b"], imp["c_direct"], imp["indirect"],
                *list(implied_loadings)]
analytic_vs_implied = [float(v - w) for v, w in
                       zip(implied_vals, analytic_vals)]

# ----------------------------------------------------------------------
# 6c. Monte Carlo study (bias, RMSE, coverage across replications)
# ----------------------------------------------------------------------
R_MC = 1000
mc = {name: np.empty(R_MC) for name in ("a", "b", "c_direct", "indirect")}
mc_se = {name: np.empty(R_MC) for name in ("se_a", "se_b", "se_c", "se_ind")}
mc_loadings = np.empty((R_MC, 6))

for r in range(R_MC):
    rr = np.random.default_rng(20261025 + 100000 + r)
    d = generate_data(rr, N_PER_REGION)
    m = fit_mediation_std(d["hazard"], d["self_efficacy"], d["social_support"],
                          d["threat_appraisal"], d["ph"], d["distress"],
                          d["adapt"])
    mc["a"][r] = m["a"]
    mc["b"][r] = m["b"]
    mc["c_direct"][r] = m["c_direct"]
    mc["indirect"][r] = m["indirect"]
    mc_se["se_a"][r] = m["se_a"]
    mc_se["se_b"][r] = m["se_b"]
    mc_se["se_c"][r] = m["se_c"]
    mc_se["se_ind"][r] = m["se_ind"]
    mc_loadings[r] = ml_cfa_loadings(d["ph_items"])[0]


def perf(est, se, true):
    est = np.asarray(est)
    se = np.asarray(se)
    mean = float(est.mean())
    sd = float(est.std(ddof=1))
    bias = float(est.mean() - true)
    rel_bias = float(bias / abs(true)) if true != 0 else float("nan")
    rmse = float(np.sqrt(np.mean((est - true) ** 2)))
    lo = est - 1.96 * se
    hi = est + 1.96 * se
    coverage = float(np.mean((lo <= true) & (true <= hi)))
    return {
        "true": round(float(true), 4),
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "bias": round(bias, 4),
        "rel_bias": round(rel_bias, 4),
        "rmse": round(rmse, 4),
        "coverage": round(coverage, 4),
    }


mc_params = {
    "a": perf(mc["a"], mc_se["se_a"], analytic["a"]),
    "b": perf(mc["b"], mc_se["se_b"], analytic["b"]),
    "c_direct": perf(mc["c_direct"], mc_se["se_c"], analytic["c_direct"]),
    "indirect": perf(mc["indirect"], mc_se["se_ind"], analytic["indirect"]),
}
mc_loading_mean = mc_loadings.mean(axis=0)
mc_loading_sd = mc_loadings.std(axis=0, ddof=1)
mc_loading_bias = mc_loading_mean - analytic["loadings"]

# Monte Carlo standard errors (Morris et al., 2019): MCSE of the bias is the
# empirical sd over replications divided by sqrt(R); MCSE of the coverage is
# sqrt(p(1-p)/R).
mcse = {}
for name in ("a", "b", "c_direct", "indirect"):
    cov = mc_params[name]["coverage"]
    mcse[name] = {
        "mcse_bias": round(float(mc[name].std(ddof=1)) / np.sqrt(R_MC), 5),
        "mcse_rmse": round(float(mc_params[name]["rmse"]) / np.sqrt(2 * R_MC), 5),
        "mcse_coverage": round(float(np.sqrt(cov * (1 - cov) / R_MC)), 5),
    }

# ----------------------------------------------------------------------
# 6d. Region-count sweep (effective sample size)
# ----------------------------------------------------------------------
def spanning_regions(nreg):
    idx = np.linspace(0, N_REGIONS - 1, nreg).round().astype(int)
    return [REGIONS[i] for i in idx], [region_dev[i] for i in idx]


R_SWEEP = 200
sweep = {}
for nreg in (10, 20, 30):
    regs, devs = spanning_regions(nreg)
    a_est = np.empty(R_SWEEP)
    ind_est = np.empty(R_SWEEP)
    a_se = np.empty(R_SWEEP)
    ind_se = np.empty(R_SWEEP)
    # Model-implied targets for this region subset (large sample)
    rng_s = np.random.default_rng(20261025 + 500000 + nreg)
    d_imp = generate_data(rng_s, 20000, regions=regs, dev=devs)
    m_imp = fit_mediation_std(d_imp["hazard"], d_imp["self_efficacy"],
                              d_imp["social_support"], d_imp["threat_appraisal"],
                              d_imp["ph"], d_imp["distress"], d_imp["adapt"])
    true_a = m_imp["a"]
    true_ind = m_imp["indirect"]
    for r in range(R_SWEEP):
        rr = np.random.default_rng(20261025 + 600000 + nreg * 1000 + r)
        d = generate_data(rr, N_PER_REGION, regions=regs, dev=devs)
        m = fit_mediation_std(d["hazard"], d["self_efficacy"],
                              d["social_support"], d["threat_appraisal"],
                              d["ph"], d["distress"], d["adapt"])
        a_est[r] = m["a"]
        ind_est[r] = m["indirect"]
        a_se[r] = m["se_a"]
        ind_se[r] = m["se_ind"]
    sweep[nreg] = {
        "n_regions": int(nreg),
        "n_total": int(nreg * N_PER_REGION),
        "a": perf(a_est, a_se, true_a),
        "indirect": perf(ind_est, ind_se, true_ind),
    }

# ----------------------------------------------------------------------
# 6d'. Design sensitivity grid (effect size, sample size, region count)
# ----------------------------------------------------------------------
# One design factor is varied at a time while the others are held at the
# primary configuration, so the boundary conditions of the workflow are
# visible. Bias and coverage are reported for the four structural paths.
R_GRID = 200


def grid_cell(seed_base, n_per_region, regions, dev, a_eff):
    est = {k: np.empty(R_GRID) for k in ("a", "b", "c_direct", "indirect")}
    se = {k: np.empty(R_GRID) for k in ("a", "b", "c_direct", "indirect")}
    for r in range(R_GRID):
        rr = np.random.default_rng(seed_base + r)
        d = generate_data(rr, n_per_region, regions=regions, dev=dev,
                          a_eff=a_eff)
        m = fit_mediation_std(d["hazard"], d["self_efficacy"],
                              d["social_support"], d["threat_appraisal"],
                              d["ph"], d["distress"], d["adapt"])
        est["a"][r] = m["a"]
        se["a"][r] = m["se_a"]
        est["b"][r] = m["b"]
        se["b"][r] = m["se_b"]
        est["c_direct"][r] = m["c_direct"]
        se["c_direct"][r] = m["se_c"]
        est["indirect"][r] = m["indirect"]
        se["indirect"][r] = m["se_ind"]
    return est, se


def grid_target(regions, dev, a_eff):
    rng_t = np.random.default_rng(20261025 + 7000000)
    d = generate_data(rng_t, 20000, regions=regions, dev=dev, a_eff=a_eff)
    m = fit_mediation_std(d["hazard"], d["self_efficacy"],
                          d["social_support"], d["threat_appraisal"],
                          d["ph"], d["distress"], d["adapt"])
    return {"a": m["a"], "b": m["b"], "c_direct": m["c_direct"],
            "indirect": m["indirect"]}


grid = {}
effect_ab = {}
for a_eff in (-0.15, -0.30, -0.45):
    tgt = grid_target(REGIONS, region_dev, a_eff)
    est, se = grid_cell(20261025 + 7001000 + int(abs(a_eff) * 10000),
                        N_PER_REGION, REGIONS, region_dev, a_eff)
    grid_ci_lo = est["indirect"] - 1.96 * se["indirect"]
    grid["effect_size_%.2f" % a_eff] = {
        "a": perf(est["a"], se["a"], tgt["a"]),
        "b": perf(est["b"], se["b"], tgt["b"]),
        "c_direct": perf(est["c_direct"], se["c_direct"], tgt["c_direct"]),
        "indirect": perf(est["indirect"], se["indirect"], tgt["indirect"]),
        "power_indirect": round(float(np.mean(grid_ci_lo > 0)), 3),
        "raw_ab": round(float(abs(a_eff) * 0.40), 3),
    }
    effect_ab[a_eff] = (est["indirect"], se["indirect"], tgt["indirect"])

for npr in (50, 100, 400):
    tgt = grid_target(REGIONS, region_dev, -0.45)
    est, se = grid_cell(20261025 + 7002000 + npr, npr, REGIONS, region_dev,
                        -0.45)
    grid["sample_size_%d" % npr] = {
        "a": perf(est["a"], se["a"], tgt["a"]),
        "b": perf(est["b"], se["b"], tgt["b"]),
        "c_direct": perf(est["c_direct"], se["c_direct"], tgt["c_direct"]),
        "indirect": perf(est["indirect"], se["indirect"], tgt["indirect"]),
        "power_indirect": round(
            float(np.mean((est["indirect"] - 1.96 * se["indirect"]) > 0)), 3),
    }

for nreg in (10, 20, 30):
    regs, devs = spanning_regions(nreg)
    tgt = grid_target(regs, devs, -0.45)
    est, se = grid_cell(20261025 + 7003000 + nreg * 100, N_PER_REGION, regs,
                        devs, -0.45)
    grid["regions_%d" % nreg] = {
        "a": perf(est["a"], se["a"], tgt["a"]),
        "b": perf(est["b"], se["b"], tgt["b"]),
        "c_direct": perf(est["c_direct"], se["c_direct"], tgt["c_direct"]),
        "indirect": perf(est["indirect"], se["indirect"], tgt["indirect"]),
        "power_indirect": round(
            float(np.mean((est["indirect"] - 1.96 * se["indirect"]) > 0)), 3),
    }



# ----------------------------------------------------------------------
# 6d'''. Measurement misspecification positive control (two-factor truth)
# ----------------------------------------------------------------------
# Items are generated from two correlated factors, and both a misspecified
# one-factor model and the correctly specified two-factor model are fitted,
# so the workflow is shown to detect misspecification and not only to recover
# a correctly specified model.


def cfa_fit(items, pattern, n_restarts=4, seed=0):
    """Fit a confirmatory factor model with a fixed 0/1 loading pattern.

    ``pattern`` is an (items, factors) binary matrix marking which item loads
    on which factor. Returns the ML discrepancy, chi-square, degrees of
    freedom, RMSEA, SRMR, and CFI of the fitted model.
    """
    Z = StandardScaler().fit_transform(items)
    n = len(Z)
    k = Z.shape[1]
    nf = pattern.shape[1]
    S = np.cov(Z, rowvar=False, bias=False)
    _, logdet_s = np.linalg.slogdet(S)
    free = np.argwhere(pattern == 1)
    n_lam = len(free)
    n_corr = nf * (nf - 1) // 2

    def unpack(p):
        lam = np.zeros((k, nf))
        lam[free[:, 0], free[:, 1]] = p[:n_lam]
        theta = np.exp(p[n_lam:n_lam + k])
        corr = np.tanh(p[n_lam + k:n_lam + k + n_corr])
        Phi = np.eye(nf)
        if nf == 2:
            Phi[0, 1] = Phi[1, 0] = corr[0]
        return lam, theta, Phi

    def fml(p):
        lam, theta, Phi = unpack(p)
        sig = lam @ Phi @ lam.T + np.diag(theta)
        sign, logdet = np.linalg.slogdet(sig)
        if sign <= 0:
            return 1e6
        return logdet + np.trace(S @ np.linalg.inv(sig)) - logdet_s - k

    p0 = np.zeros(n_lam + k + n_corr)
    p0[:n_lam] = 0.6
    p0[n_lam:n_lam + k] = np.log(0.5)
    rng_c = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        init = p0 + rng_c.normal(0, 0.05, p0.size)
        res = minimize(fml, init, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    F = float(best.fun)
    lam, theta, Phi = unpack(best.x)
    sig = lam @ Phi @ lam.T + np.diag(theta)
    df = k * (k + 1) // 2 - (n_lam + k + n_corr)
    chi2 = (n - 1) * F
    rmsea = np.sqrt(max(F / df - 1 / (n - 1), 0))
    resid = (S - sig) / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    srmr = np.sqrt(np.mean(resid[np.triu_indices(k, 1)] ** 2))
    S_null = np.diag(np.diag(S))
    _, logdet0 = np.linalg.slogdet(S_null)
    F_null = logdet0 + np.trace(S @ np.linalg.inv(S_null)) - logdet_s - k
    df_null = k * (k + 1) // 2 - k
    cfi = 1 - max(chi2 - df, 0) / max((n - 1) * F_null - df_null, 0)
    return {"F": F, "chi2": chi2, "df": df, "rmsea": rmsea, "srmr": srmr,
            "cfi": cfi}


def generate_two_factor_items(rng, n, lam=0.75, rho=0.5):
    eta = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], n)
    items = np.empty((n, 6))
    for j in range(3):
        items[:, j] = lam * eta[:, 0] + rng.normal(0, np.sqrt(1 - lam ** 2), n)
    for j in range(3):
        items[:, 3 + j] = lam * eta[:, 1] + rng.normal(
            0, np.sqrt(1 - lam ** 2), n)
    return items


R_MISS = 100
pat1 = np.ones((6, 1))
pat2 = np.zeros((6, 2))
pat2[:3, 0] = 1
pat2[3:, 1] = 1
miss = {k: np.empty(R_MISS) for k in ("cfi", "rmsea", "srmr", "chi2")}
miss2 = {k: np.empty(R_MISS) for k in ("cfi", "rmsea", "srmr", "chi2")}
for r in range(R_MISS):
    rr = np.random.default_rng(20261025 + 7100000 + r)
    items = generate_two_factor_items(rr, N)
    f1 = cfa_fit(items, pat1, seed=r)
    f2 = cfa_fit(items, pat2, seed=r + 1)
    for key in miss:
        miss[key][r] = f1[key]
        miss2[key][r] = f2[key]


# ----------------------------------------------------------------------
# 6e. Mixture positive control (true two-component population)
# ----------------------------------------------------------------------
def mixture_on_two_component(separation, n=12000, seed=7):
    rng_pc = np.random.default_rng(seed)
    n2 = int(0.4 * n)
    x1 = rng_pc.multivariate_normal([0.0, 0.0], np.eye(2), n - n2)
    x2 = rng_pc.multivariate_normal([separation, separation], np.eye(2), n2)
    X = np.vstack([x1, x2])
    bics = []
    for k in range(1, 7):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=0, n_init=10)
        g.fit(X)
        bics.append(g.bic(X))
    k_sel = int(np.argmin(bics)) + 1
    g2 = GaussianMixture(n_components=2, covariance_type="full",
                         random_state=0, n_init=10).fit(X)
    resp = g2.predict_proba(X)
    ent = 1 - (-np.sum(resp * np.log(resp + 1e-12)) / (n * np.log(2)))
    return k_sel, float(ent), [float(x) for x in bics]


pc_k_sel, pc_ent, pc_bics = mixture_on_two_component(1.5)
pc_sep = {str(sep): {"k_selected": mixture_on_two_component(sep)[0],
                     "entropy_2comp": round(mixture_on_two_component(sep)[1], 3)}
          for sep in (0.5, 1.0, 1.5, 2.0, 2.5)}

# ----------------------------------------------------------------------
# 6f. Empirical calibration against observed hazard mortality (EM-DAT)
# ----------------------------------------------------------------------
REGION_TO_ENTITY = {
    "Bangladesh": "Bangladesh", "Philippines": "Philippines",
    "Niger": "Niger", "Somalia": "Somalia", "Honduras": "Honduras",
    "Fiji": "Fiji", "Pakistan": "Pakistan", "Mozambique": "Mozambique",
    "Indonesia": "Indonesia", "Viet Nam": "Vietnam", "Egypt": "Egypt",
    "Iraq": "Iraq", "Uzbekistan": "Uzbekistan", "Spain": "Spain",
    "Mexico": "Mexico", "Peru": "Peru", "Kenya": "Kenya",
    "Thailand": "Thailand", "Brazil": "Brazil", "Greece": "Greece",
    "United States": "United States", "China": "China",
    "Australia": "Australia", "Italy": "Italy", "Japan": "Japan",
    "France": "France", "United Kingdom": "United Kingdom",
    "Norway": "Norway", "Canada": "Canada", "New Zealand": "New Zealand",
}
CLIMATE_TYPES = ["deaths_drought", "deaths_flood", "deaths_storm",
                 "deaths_landslide", "deaths_wildfire",
                 "deaths_extreme_temp"]
EMDAT_NYEARS = 31  # 1990 to 2020 inclusive

emdat_deaths = {e: 0.0 for e in REGION_TO_ENTITY.values()}
with open(os.path.join(DATADIR, "emdat_climate.csv")) as f:
    for row in csv.DictReader(f):
        emdat_deaths[row["entity"]] += sum(float(row[k]) for k in CLIMATE_TYPES)

emdat_pop = {e: 0.0 for e in REGION_TO_ENTITY.values()}
with open(os.path.join(DATADIR, "population_unwpp.csv")) as f:
    for row in csv.DictReader(f):
        emdat_pop[row["entity"]] += float(row["population"])

emdat_rate = np.empty(N_REGIONS)
for k, name in enumerate(region_names):
    ent = REGION_TO_ENTITY[name]
    mean_pop = emdat_pop[ent] / EMDAT_NYEARS
    emdat_rate[k] = (emdat_deaths[ent] / EMDAT_NYEARS) / (mean_pop / 1e6)

lograte = np.log10(emdat_rate + 1e-3)
emdat_rho, emdat_p = spearmanr(region_hazard, lograte)
emdat_slope, emdat_intercept = np.polyfit(region_hazard, lograte, 1)
emdat_fit = emdat_intercept + emdat_slope * region_hazard
emdat_r2 = 1.0 - np.sum((lograte - emdat_fit) ** 2) / np.sum(
    (lograte - np.mean(lograte)) ** 2)






# ----------------------------------------------------------------------
# 6h. Bifactor positive control (general factor + three content factors)
# ----------------------------------------------------------------------
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
_GL_NODES, _GL_WTS = np.polynomial.legendre.leggauss(24)


def _bvncdf(a, b, rho):
    """Standard bivariate normal CDF Phi2(a, b; rho) by Gauss-Legendre
    quadrature, with the infinite boundaries clamped to a finite tail."""
    Fa = 0.0 if a <= -8.0 else (1.0 if a >= 8.0 else ndtr(a))
    Fb = 0.0 if b <= -8.0 else (1.0 if b >= 8.0 else ndtr(b))
    a = float(np.clip(a, -8.0, 8.0))
    b = float(np.clip(b, -8.0, 8.0))
    t = 0.5 * rho * (_GL_NODES + 1.0)
    w = 0.5 * rho * _GL_WTS
    denom = 1.0 - t * t
    integ = np.exp(-(a * a - 2.0 * t * a * b + b * b) / (2.0 * denom)) / np.sqrt(denom)
    return Fa * Fb + float(np.sum(w * integ)) / (2.0 * np.pi)


def cfa_fit_bifactor(items, n_restarts=4, seed=0):
    """Fit an orthogonal bifactor model (general factor plus three content
    factors, two items each); returns fit indices and general loadings."""
    Z = StandardScaler().fit_transform(items)
    n = len(Z)
    k = Z.shape[1]
    nf = 4
    S = np.cov(Z, rowvar=False, bias=False)
    _, logdet_s = np.linalg.slogdet(S)
    pattern = np.zeros((k, nf))
    pattern[:, 0] = 1
    for j in range(k):
        pattern[j, 1 + j // 2] = 1
    free = np.argwhere(pattern == 1)
    n_lam = len(free)

    def unpack(p):
        lam = np.zeros((k, nf))
        lam[free[:, 0], free[:, 1]] = p[:n_lam]
        theta = np.exp(p[n_lam:n_lam + k])
        return lam, theta

    def fml(p):
        lam, theta = unpack(p)
        sig = lam @ lam.T + np.diag(theta)
        sign, logdet = np.linalg.slogdet(sig)
        if sign <= 0:
            return 1e6
        return logdet + np.trace(S @ np.linalg.inv(sig)) - logdet_s - k

    p0 = np.zeros(n_lam + k)
    for idx, (i, j) in enumerate(free):
        p0[idx] = 0.55 if j == 0 else 0.45
    p0[n_lam:] = np.log(0.3)
    rng_c = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        init = p0 + rng_c.normal(0, 0.05, p0.size)
        res = minimize(fml, init, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    F = float(best.fun)
    lam, theta = unpack(best.x)
    sig = lam @ lam.T + np.diag(theta)
    df = k * (k + 1) // 2 - (n_lam + k)
    chi2 = (n - 1) * F
    rmsea = np.sqrt(max(F / df - 1 / (n - 1), 0))
    resid = (S - sig) / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    srmr = np.sqrt(np.mean(resid[np.triu_indices(k, 1)] ** 2))
    S_null = np.diag(np.diag(S))
    _, logdet0 = np.linalg.slogdet(S_null)
    F_null = logdet0 + np.trace(S @ np.linalg.inv(S_null)) - logdet_s - k
    df_null = k * (k + 1) // 2 - k
    cfi = 1 - max(chi2 - df, 0) / max((n - 1) * F_null - df_null, 0)
    return {"F": F, "chi2": chi2, "df": df, "rmsea": rmsea, "srmr": srmr,
            "cfi": cfi, "gen_loadings": lam[:, 0],
            "content_loadings": lam[:, 1:]}


def generate_bifactor_items(rng, n, lam_g=0.55, lam_c=0.45):
    g = rng.normal(0, 1, n)
    items = np.empty((n, 6))
    resid_sd = np.sqrt(1 - lam_g ** 2 - lam_c ** 2)
    for block in range(3):
        f = rng.normal(0, 1, n)
        for j in range(2):
            items[:, 2 * block + j] = (lam_g * g + lam_c * f
                                       + rng.normal(0, resid_sd, n))
    return items


R_BIF = 100
bif_one = {k: np.empty(R_BIF) for k in ("cfi", "rmsea", "srmr")}
bif_full = {k: np.empty(R_BIF) for k in ("cfi", "rmsea", "srmr")}
bif_gen = np.empty((R_BIF, 6))
for r in range(R_BIF):
    rr = np.random.default_rng(20261025 + 8200000 + r)
    items = generate_bifactor_items(rr, N)
    f1 = cfa_fit(items, np.ones((6, 1)), seed=r)
    f2 = cfa_fit_bifactor(items, seed=r + 1)
    for key in ("cfi", "rmsea", "srmr"):
        bif_one[key][r] = f1[key]
        bif_full[key][r] = f2[key]
    bif_gen[r] = f2["gen_loadings"]


# ----------------------------------------------------------------------
# 6i. Ordinal and missing-data conditions for the measurement stage
# ----------------------------------------------------------------------
def ordinalize(items, n_cat=5):
    """Standardize each item and cut at fixed normal thresholds to a Likert
    scale coded 0..n_cat-1."""
    Z = StandardScaler().fit_transform(items)
    th = np.array([-1.28, -0.52, 0.52, 1.28])
    return np.digitize(Z, th)


def _marginal_thresholds(x, n_cat=5):
    counts = np.bincount(x, minlength=n_cat).astype(float)
    cump = np.cumsum(counts)[:-1] / counts.sum()
    cump = np.clip(cump, 1e-4, 1 - 1e-4)
    return norm.ppf(cump)


def _polychoric_pair(x, y, n_cat=5):
    thx = _marginal_thresholds(x, n_cat)
    thy = _marginal_thresholds(y, n_cat)
    tx = np.concatenate([[-np.inf], thx, [np.inf]])
    ty = np.concatenate([[-np.inf], thy, [np.inf]])
    tab = np.zeros((n_cat, n_cat))
    for a in range(n_cat):
        for b in range(n_cat):
            tab[a, b] = np.sum((x == a) & (y == b))
    tab = tab.astype(float)

    def nll(rho):
        rho = float(np.clip(rho, -0.999, 0.999))
        P = np.zeros((n_cat, n_cat))
        for a in range(n_cat):
            for b in range(n_cat):
                P[a, b] = (_bvncdf(tx[a + 1], ty[b + 1], rho)
                           - _bvncdf(tx[a], ty[b + 1], rho)
                           - _bvncdf(tx[a + 1], ty[b], rho)
                           + _bvncdf(tx[a], ty[b], rho))
        P = np.clip(P, 1e-12, 1.0)
        return -float(np.sum(tab * np.log(P)))

    r0 = float(np.clip(np.corrcoef(x, y)[0, 1], -0.95, 0.95))
    res = minimize_scalar(nll, bounds=(-0.999, 0.999), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x)


def polychoric_matrix(ordinal, n_cat=5):
    k = ordinal.shape[1]
    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            r = _polychoric_pair(ordinal[:, i], ordinal[:, j], n_cat)
            R[i, j] = R[j, i] = r
    return R


def cfa_on_correlation(R, n_restarts=2, seed=0):
    """One-factor ML CFA taking a correlation matrix as input."""
    k = R.shape[0]
    _, logdet_s = np.linalg.slogdet(R)

    def fml(p):
        lam = p[:k]
        theta = np.exp(p[k:])
        sig = np.outer(lam, lam) + np.diag(theta)
        sign, logdet = np.linalg.slogdet(sig)
        if sign <= 0:
            return 1e6
        return logdet + np.trace(R @ np.linalg.inv(sig)) - logdet_s - k

    eig, vec = np.linalg.eigh(R)
    init = vec[:, -1] * np.sqrt(max(float(eig[-1]), 1e-6))
    init = np.clip(init, -0.9, 0.9)
    p0 = np.concatenate([init, np.log(1 - init ** 2)])
    rng_c = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        start = p0 + rng_c.normal(0, 0.05, p0.size)
        res = minimize(fml, start, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    lam = best.x[:k]
    theta = np.exp(best.x[k:])
    return lam, theta


def fiml_cfa_loadings(items, n_restarts=2, seed=0):
    """Full-information ML one-factor CFA with item missingness (NaN)."""
    k = items.shape[1]
    Z = np.empty_like(items)
    for j in range(k):
        mu = np.nanmean(items[:, j])
        sd = np.nanstd(items[:, j])
        Z[:, j] = (items[:, j] - mu) / sd
    mask = ~np.isnan(Z)
    groups = {}
    for i in range(Z.shape[0]):
        key = tuple(mask[i].tolist())
        groups.setdefault(key, []).append(Z[i, mask[i]])
    for key in groups:
        groups[key] = np.array(groups[key])

    def fml(p):
        lam = p[:k]
        theta = np.exp(p[k:])
        sig = np.outer(lam, lam) + np.diag(theta)
        total = 0.0
        for key, rows in groups.items():
            idx = [i for i, v in enumerate(key) if v]
            sub = sig[np.ix_(idx, idx)]
            sign, logdet = np.linalg.slogdet(sub)
            if sign <= 0:
                return 1e6
            inv = np.linalg.inv(sub)
            q = np.einsum('ij,jk,ik->i', rows, inv, rows)
            total += rows.shape[0] * logdet + q.sum()
        return total

    init = pca_first_loadings(np.nan_to_num(Z, nan=0.0))
    p0 = np.concatenate([init, np.log(1 - init ** 2)])
    rng_c = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        start = p0 + rng_c.normal(0, 0.05, p0.size)
        res = minimize(fml, start, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    lam = best.x[:k]
    theta = np.exp(best.x[k:])
    return lam, theta


def regression_factor_score(items, lam, theta):
    """Regression factor score on standardized items (complete or ordinal)."""
    Z = StandardScaler().fit_transform(items)
    sig = np.outer(lam, lam) + np.diag(theta)
    return Z @ np.linalg.inv(sig) @ lam


def regression_factor_score_missing(Z, lam, theta):
    """Regression factor score using only the observed items (NaN entries)."""
    sig = np.outer(lam, lam) + np.diag(theta)
    mask = ~np.isnan(Z)
    fs = np.zeros(Z.shape[0])
    for key in {tuple(m.tolist()) for m in mask}:
        idx = [i for i, v in enumerate(key) if v]
        if not idx:
            continue
        rows = np.all(mask == np.array(key, dtype=bool), axis=1)
        sub_inv = np.linalg.inv(sig[np.ix_(idx, idx)])
        fs[rows] = Z[np.ix_(rows, idx)] @ sub_inv @ lam[idx]
    return fs


R_COND = 100
cond = {name: {"loadings": np.empty((R_COND, 6)),
               "a": np.empty(R_COND), "b": np.empty(R_COND),
               "c_direct": np.empty(R_COND), "indirect": np.empty(R_COND),
               "fs_rel": np.empty(R_COND)}
        for name in ("ordinal", "missing")}

for r in range(R_COND):
    rr = np.random.default_rng(20261025 + 9300000 + r)
    d = generate_data(rr, N_PER_REGION)
    for name in ("ordinal", "missing"):
        if name == "ordinal":
            items = ordinalize(d["ph_items"])
            R_poly = polychoric_matrix(items)
            lam, theta = cfa_on_correlation(R_poly, seed=r)
            if float(np.dot(lam, analytic["loadings"])) < 0:
                lam = -lam
            fs = regression_factor_score(items, lam, theta)
        else:
            items = d["ph_items"].copy()
            missmask = rr.random(items.shape) < 0.15
            items[missmask] = np.nan
            lam, theta = fiml_cfa_loadings(items, seed=r)
            if float(np.dot(lam, analytic["loadings"])) < 0:
                lam = -lam
            Zc = (items - np.nanmean(items, axis=0)) / np.nanstd(items, axis=0)
            fs = regression_factor_score_missing(Zc, lam, theta)
        cond[name]["loadings"][r] = lam
        cond[name]["fs_rel"][r] = np.corrcoef(fs, d["ph"])[0, 1] ** 2
        m = fit_mediation_std(d["hazard"], d["self_efficacy"],
                              d["social_support"], d["threat_appraisal"],
                              fs, d["distress"], d["adapt"])
        cond[name]["a"][r] = m["a"]
        cond[name]["b"][r] = m["b"]
        cond[name]["c_direct"][r] = m["c_direct"]
        cond[name]["indirect"][r] = m["indirect"]

cond_bias = {name: {
    "loadings": cond[name]["loadings"].mean(axis=0) - analytic["loadings"],
    "a": float(cond[name]["a"].mean()) - analytic["a"],
    "b": float(cond[name]["b"].mean()) - analytic["b"],
    "c_direct": float(cond[name]["c_direct"].mean()) - analytic["c_direct"],
    "indirect": float(cond[name]["indirect"].mean()) - analytic["indirect"],
    "fs_rel": float(cond[name]["fs_rel"].mean())}
    for name in ("ordinal", "missing")}


# ----------------------------------------------------------------------
# 6j. Projection uncertainty from the bootstrap of the raw coefficients
# ----------------------------------------------------------------------
def raw_structural_coefs(haz, dev, se, ss, ta, ph_, di):
    out = {}
    out["se"] = ols(se, np.column_stack([dev, haz]))[0]
    out["ss"] = ols(ss, np.column_stack([dev]))[0]
    out["ta"] = ols(ta, np.column_stack([haz]))[0]
    out["ph"] = ols(ph_, np.column_stack([haz, se, ss]))[0]
    out["di"] = ols(di, np.column_stack([haz, ph_, ta]))[0]
    return out


def projection_delta(coefs, lam=1.35):
    def fields(h):
        se_f = coefs["se"][0] + coefs["se"][1] * dev_field + coefs["se"][2] * h
        ss_f = coefs["ss"][0] + coefs["ss"][1] * dev_field
        ta_f = coefs["ta"][0] + coefs["ta"][1] * h
        ph_f = (coefs["ph"][0] + coefs["ph"][1] * h
                + coefs["ph"][2] * se_f + coefs["ph"][3] * ss_f)
        di_f = (coefs["di"][0] + coefs["di"][1] * h
                + coefs["di"][2] * ph_f + coefs["di"][3] * ta_f)
        return np.where(land, di_f, np.nan)
    cur = fields(hazard_field)
    fut = fields(np.clip(hazard_field * lam, 0, 1))
    return float(np.nanmean(risk_from_distress(fut))
                 - np.nanmean(risk_from_distress(cur)))


B_PROJ = 1000
proj_delta_bs = np.empty(B_PROJ)
for i in range(B_PROJ):
    idx = rng_main.integers(0, N, N)
    cf = raw_structural_coefs(hazard[idx], dev[idx], self_efficacy[idx],
                              social_support[idx], threat_appraisal[idx],
                              ph[idx], distress[idx])
    proj_delta_bs[i] = projection_delta(cf)
proj_lo, proj_hi = np.percentile(proj_delta_bs, [2.5, 97.5])

# ----------------------------------------------------------------------
# 7. Results
# ----------------------------------------------------------------------
results = {
    "n_total": int(N),
    "n_regions": int(N_REGIONS),
    "ph_alpha": round(float(ph_alpha), 3),
    "ph_alpha_ci": ph_alpha_ci,
    "ph_omega": round(float(ph_omega), 3),
    "ph_omega_ci": ph_omega_ci,
    "ph_fs_reliability": round(float(ph_fs_rel), 3),
    "recovered_loadings": [round(float(x), 3) for x in recovered_loadings],
    "loadings_ci": [[round(float(lo), 3), round(float(hi), 3)]
                    for lo, hi in zip(loadings_ci_lo, loadings_ci_hi)],
    "cfa_fit": {
        "cfi": round(float(cfi), 3),
        "rmsea": round(float(rmsea), 3),
        "srmr": round(float(srmr), 3),
        "f_ml": round(float(F_ML), 5),
    },
    "a": round(float(a), 3),
    "b": round(float(b), 3),
    "c_direct": round(float(c_direct), 3),
    "c_total": round(float(c_total), 3),
    "indirect": round(float(indirect), 3),
    "prop_mediated": prop_mediated,
    "prop_mediated_conditional": round(float(indirect / (c_direct + indirect)), 3),
    "prop_mediated_ci": [round(float(prop_med_lo), 3),
                         round(float(prop_med_hi), 3)],
    "indirect_ci": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
    "indirect_ci_cluster": [round(float(ci_lo_cluster), 3), round(float(ci_hi_cluster), 3)],
    "r2_ph": round(float(r2_ph), 3),
    "r2_distress": round(float(r2_distress), 3),
    "r2_adapt": round(float(r2_adapt), 3),
    "r2_ci": r2_ci,
    "adapt_beta": [round(float(x), 3) for x in beta_adapt[1:]],
    "beta_a": [round(float(x), 3) for x in beta_a],
    "beta_b": [round(float(x), 3) for x in beta_b],
    "icc": {"ph": icc_ph, "distress": icc_distress, "adapt": icc_adapt},
    "icc_residual": {"ph": icc_ph_resid, "distress": icc_di_resid},
    "factor_score_sensitivity": {
        "a": round(float(m_fs["a"]), 3),
        "b": round(float(m_fs["b"]), 3),
        "c_direct": round(float(m_fs["c_direct"]), 3),
        "indirect": round(float(m_fs["indirect"]), 3),
    },
    "k_best": int(k_best),
    "bic": [round(float(x), 1) for x in bics],
    "profile_means": [[round(float(x), 3) for x in m] for m in profile_means],
    "profile_adapt": [round(float(x), 3) for x in profile_adapt],
    "profile_hazard": [round(float(x), 3) for x in profile_hazard],
    "class_sizes": [round(float(x), 3) for x in class_sizes],
    "avg_post": [round(float(x), 3) for x in avg_post],
    "entropy": round(float(entropy), 3),
    "entropy_2comp": round(float(entropy2), 3),
    "ci": {
        "a": boot_ci("a"),
        "self_efficacy": boot_ci("se"),
        "social_support": boot_ci("ss"),
        "c_direct": boot_ci("c_direct"),
        "c_total": boot_ci("c_total"),
        "b": boot_ci("b"),
        "threat_appraisal": boot_ci("ta"),
        "adapt_ph": boot_ci("adapt_ph"),
        "adapt_di": boot_ci("adapt_di"),
        "adapt_h": boot_ci("adapt_h"),
    },
    "ci_cluster": {
        "a": cluster_ci("a"),
        "self_efficacy": cluster_ci("se"),
        "social_support": cluster_ci("ss"),
        "c_direct": cluster_ci("c_direct"),
        "c_total": cluster_ci("c_total"),
        "b": cluster_ci("b"),
        "threat_appraisal": cluster_ci("ta"),
        "adapt_ph": cluster_ci("adapt_ph"),
        "adapt_di": cluster_ci("adapt_di"),
        "adapt_h": cluster_ci("adapt_h"),
    },
    "hazard_ph_corr": round(float(np.corrcoef(region_hazard, region_ph)[0, 1]), 3),
    "risk_mean_increase": risk_mean_increase,
    "risk_increase_sensitivity": risk_increase,
    "emdat": {
        "spearman_rho": round(float(emdat_rho), 3),
        "spearman_p": round(float(emdat_p), 6),
        "loglinear_slope": round(float(emdat_slope), 3),
        "loglinear_intercept": round(float(emdat_intercept), 3),
        "loglinear_r2": round(float(emdat_r2), 3),
        "regions": [
            {"name": region_names[k],
             "hazard": round(float(region_hazard[k]), 3),
             "deaths": round(float(emdat_deaths[REGION_TO_ENTITY[region_names[k]]]), 1),
             "rate_per_million_per_year": round(float(emdat_rate[k]), 2)}
            for k in range(N_REGIONS)
        ],
    },
    "implied": {
        "a": round(float(imp["a"]), 4),
        "b": round(float(imp["b"]), 4),
        "c_direct": round(float(imp["c_direct"]), 4),
        "c_total": round(float(imp["c_total"]), 4),
        "indirect": round(float(imp["indirect"]), 4),
        "se_coef": round(float(imp["se_coef"]), 4),
        "ss_coef": round(float(imp["ss_coef"]), 4),
        "ta_coef": round(float(imp["ta_coef"]), 4),
        "adapt_ph": round(float(imp["adapt_ph"]), 4),
        "adapt_di": round(float(imp["adapt_di"]), 4),
        "adapt_h": round(float(imp["adapt_h"]), 4),
        "loadings": [round(float(x), 4) for x in implied_loadings],
    },
    "analytic": {
        "a": round(float(analytic["a"]), 4),
        "b": round(float(analytic["b"]), 4),
        "c_direct": round(float(analytic["c_direct"]), 4),
        "c_total": round(float(analytic["c_total"]), 4),
        "indirect": round(float(analytic["indirect"]), 4),
        "se_coef": round(float(analytic["se_coef"]), 4),
        "ss_coef": round(float(analytic["ss_coef"]), 4),
        "ta_coef": round(float(analytic["ta_coef"]), 4),
        "adapt_ph": round(float(analytic["adapt_ph"]), 4),
        "adapt_di": round(float(analytic["adapt_di"]), 4),
        "adapt_h": round(float(analytic["adapt_h"]), 4),
        "loadings": [round(float(x), 4) for x in analytic["loadings"]],
    },
    "analytic_vs_implied": [round(float(x), 5) for x in analytic_vs_implied],
    "monte_carlo": {
        "n_replications": R_MC,
        "params": mc_params,
        "loadings": {
            "mean": [round(float(x), 4) for x in mc_loading_mean],
            "sd": [round(float(x), 4) for x in mc_loading_sd],
            "bias": [round(float(x), 4) for x in mc_loading_bias],
        },
        "mcse": mcse,
    },
    "region_sweep": sweep,
    "design_grid": grid,
    "misspecification": {
        "one_factor": {k: {"mean": round(float(miss[k].mean()), 3),
                           "sd": round(float(miss[k].std(ddof=1)), 3)}
                       for k in ("cfi", "rmsea", "srmr", "chi2")},
        "two_factor": {k: {"mean": round(float(miss2[k].mean()), 3),
                           "sd": round(float(miss2[k].std(ddof=1)), 3)}
                       for k in ("cfi", "rmsea", "srmr", "chi2")},
    },
    "positive_control": {
        "separation_1.5": {"k_selected": int(pc_k_sel),
                           "entropy_2comp": round(pc_ent, 3)},
        "separation_sweep": pc_sep,
    },
    "null_mixture": {
        "n_replications": R_NULL,
        "prop_k1": null_k_prop1,
        "mean_k": null_k_mean,
    },
    "bifactor": {
        "n_replications": R_BIF,
        "one_factor": {k: round(float(np.mean(bif_one[k])), 3)
                       for k in ("cfi", "rmsea", "srmr")},
        "bifactor": {k: round(float(np.mean(bif_full[k])), 3)
                     for k in ("cfi", "rmsea", "srmr")},
        "general_loading_mean": [round(float(x), 3)
                                 for x in bif_gen.mean(axis=0)],
    },
    "ordinal_condition": {
        "n_replications": R_COND,
        "loadings_mean": [round(float(x), 3)
                          for x in cond["ordinal"]["loadings"].mean(axis=0)],
        "loading_bias": [round(float(x), 3)
                         for x in cond_bias["ordinal"]["loadings"]],
        "path_bias": {
            "a": round(cond_bias["ordinal"]["a"], 3),
            "b": round(cond_bias["ordinal"]["b"], 3),
            "c_direct": round(cond_bias["ordinal"]["c_direct"], 3),
            "indirect": round(cond_bias["ordinal"]["indirect"], 3),
        },
        "fs_reliability": round(cond_bias["ordinal"]["fs_rel"], 3),
    },
    "missingness_condition": {
        "n_replications": R_COND,
        "loadings_mean": [round(float(x), 3)
                          for x in cond["missing"]["loadings"].mean(axis=0)],
        "loading_bias": [round(float(x), 3)
                         for x in cond_bias["missing"]["loadings"]],
        "path_bias": {
            "a": round(cond_bias["missing"]["a"], 3),
            "b": round(cond_bias["missing"]["b"], 3),
            "c_direct": round(cond_bias["missing"]["c_direct"], 3),
            "indirect": round(cond_bias["missing"]["indirect"], 3),
        },
        "fs_reliability": round(cond_bias["missing"]["fs_rel"], 3),
    },
    "projection_uncertainty": {
        "lambda": 1.35,
        "point": round(float(np.mean(proj_delta_bs)), 4),
        "ci": [round(float(proj_lo), 4), round(float(proj_hi), 4)],
    },
    "regions": [
        {"name": region_names[k], "lat": float(region_lat[k]),
         "lon": float(region_lon[k]), "hazard": round(float(region_hazard[k]), 3),
         "dev": round(float(region_dev[k]), 3),
         "ph_mean": round(float(region_ph[k]), 3),
         "distress_mean": round(float(region_distress[k]), 3)}
        for k in range(N_REGIONS)
    ],
}

if __name__ == "__main__":
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(DATADIR, "regions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "lat", "lon", "hazard", "dev", "ph_mean", "distress_mean"])
        for k in range(N_REGIONS):
            w.writerow([region_names[k], round(float(region_lat[k]), 2),
                        round(float(region_lon[k]), 2),
                        round(float(region_hazard[k]), 3),
                        round(float(region_dev[k]), 3),
                        round(float(region_ph[k]), 3),
                        round(float(region_distress[k]), 3)])

    print(json.dumps(results, indent=2))
