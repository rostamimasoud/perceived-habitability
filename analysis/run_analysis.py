"""
Analysis pipeline for the manuscript on perceived habitability.

Generates a synthetic, internally consistent dataset, fits the core model
(measurement, mediation, latent profiles, global projection), and renders the
four main-text figures as vector PDFs. All quantities are written to
results.json so the manuscript can stay self-consistent.

The dataset is a deterministic placeholder. Replace the data-loading block
with the real survey and gridded climate data before submission.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "figures")
DATADIR = os.path.join(ROOT, "data")
os.makedirs(FIGDIR, exist_ok=True)

rng = np.random.default_rng(20261025)

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

PALETTE = ["#2166ac", "#d6604d", "#4393c3", "#b2182b", "#4daf4a"]

def panel_label(ax, label):
    ax.set_title(label, loc="left", fontweight="bold", fontsize=10, pad=3)


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


def plot_cov_ellipse(ax, mean, cov, nstd=1.5, color="k", **kw):
    vals, vecs = np.linalg.eigh(cov)
    order_ = vals.argsort()[::-1]
    vals, vecs = vals[order_], vecs[:, order_]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nstd * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=theta, fill=False,
                         ec=color, lw=0.8, zorder=6, **kw))


# ----------------------------------------------------------------------
# 1. Synthetic data
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

N_REGIONS = len(REGIONS)
N_PER_REGION = 400
N = N_REGIONS * N_PER_REGION

region_idx = np.repeat(np.arange(N_REGIONS), N_PER_REGION)
hazard = region_hazard[region_idx]
hazard = hazard + rng.normal(0, 0.05, N)          # within-region variation
hazard = np.clip(hazard, 0.0, 1.0)

# Individual-level covariates (standardized after generation)
self_efficacy = -0.35 * hazard + rng.normal(0, 0.85, N)
social_support = -0.15 * hazard + rng.normal(0, 0.9, N)
threat_appraisal = 0.55 * hazard + rng.normal(0, 0.8, N)

# Perceived habitability (latent), its six observed items, and outcomes
ph = (-0.45 * hazard + 0.25 * self_efficacy + 0.20 * social_support
      + rng.normal(0, 0.6, N))
distress = (0.35 * hazard - 0.40 * ph + 0.25 * threat_appraisal
            + rng.normal(0, 0.6, N))
adapt = (0.30 * ph - 0.20 * distress + 0.15 * hazard + rng.normal(0, 0.7, N))

# PH items (one-factor, loadings around 0.75)
loadings = np.array([0.80, 0.76, 0.72, 0.78, 0.70, 0.74])
ph_items = np.column_stack([l * ph + rng.normal(0, np.sqrt(1 - l**2), N)
                            for l in loadings])

# ----------------------------------------------------------------------
# 2. Measurement
# ----------------------------------------------------------------------
def cronbach_alpha(X):
    k = X.shape[1]
    var_sum = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - var_sum / total_var)

ph_alpha = cronbach_alpha(ph_items)
ph_item_mean = ph_items.mean(axis=0)

# PCA loadings (first component, sign-adjusted)
Z = StandardScaler().fit_transform(ph_items)
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
pca_loadings = Vt[0] * np.sqrt(S[0]**2 / (N - 1))
if pca_loadings[0] < 0:
    pca_loadings = -pca_loadings
ph_omega = (pca_loadings.sum() ** 2) / (pca_loadings.sum() ** 2
                                      + (1 - pca_loadings ** 2).sum())

# One-factor CFA with maximum likelihood estimation and fit indices
from scipy.optimize import minimize
S_items = np.cov(Z, rowvar=False, bias=False)
_, logdet_s = np.linalg.slogdet(S_items)
k_items = Z.shape[1]


def _cfa_fml(params):
    lam = params[:k_items]
    theta = np.exp(params[k_items:])
    sig = np.outer(lam, lam) + np.diag(theta)
    sign, logdet = np.linalg.slogdet(sig)
    if sign <= 0:
        return 1e6
    return logdet + np.trace(S_items @ np.linalg.inv(sig)) - logdet_s - k_items


_opt = minimize(_cfa_fml,
                np.concatenate([pca_loadings, np.log(1 - pca_loadings ** 2)]),
                method="BFGS")
lam_ml = _opt.x[:k_items]
theta_ml = np.exp(_opt.x[k_items:])
F_ML = float(_opt.fun)
sig_hat = np.outer(lam_ml, lam_ml) + np.diag(theta_ml)
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
# 3. Regression / mediation (all standardized)
# ----------------------------------------------------------------------
cols = np.column_stack([hazard, self_efficacy, social_support,
                        threat_appraisal, ph, distress, adapt])
scaler = StandardScaler()
Zcols = scaler.fit_transform(cols)
Zh, Zse, Zss, Zta, Zph, Zdi, Zad = Zcols.T


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    r2 = 1 - np.var(resid, ddof=1) / np.var(y, ddof=1)
    return beta, r2


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
boot = {k: np.empty(B) for k in
        ("a", "se", "ss", "c_direct", "b", "ta",
         "adapt_ph", "adapt_di", "adapt_h")}
boot_loadings = np.empty((B, 6))
for i in range(B):
    idx = rng.integers(0, N, N)
    ba, _ = ols(Zph[idx], np.column_stack([Zh[idx], Zse[idx], Zss[idx]]))
    bb, _ = ols(Zdi[idx], np.column_stack([Zh[idx], Zph[idx], Zta[idx]]))
    bad, _ = ols(Zad[idx], np.column_stack([Zph[idx], Zdi[idx], Zh[idx]]))
    ind_bs[i] = ba[1] * bb[2]
    boot["a"][i] = ba[1]
    boot["se"][i] = ba[2]
    boot["ss"][i] = ba[3]
    boot["c_direct"][i] = bb[1]
    boot["b"][i] = bb[2]
    boot["ta"][i] = bb[3]
    boot["adapt_ph"][i] = bad[1]
    boot["adapt_di"][i] = bad[2]
    boot["adapt_h"][i] = bad[3]
    Zi = Z[idx]
    Ui, Si, Vti = np.linalg.svd(Zi, full_matrices=False)
    li = Vti[0] * np.sqrt(Si[0] ** 2 / (N - 1))
    if li[0] < 0:
        li = -li
    boot_loadings[i] = li
ci_lo, ci_hi = np.percentile(ind_bs, [2.5, 97.5])
loadings_ci_lo = np.percentile(boot_loadings, 2.5, axis=0)
loadings_ci_hi = np.percentile(boot_loadings, 97.5, axis=0)


def boot_ci(name):
    return [round(float(np.percentile(boot[name], 2.5)), 3),
            round(float(np.percentile(boot[name], 97.5)), 3)]


# Cluster bootstrap: resample regions, keep respondents within each region
region_members = {r: np.where(region_idx == r)[0] for r in np.arange(N_REGIONS)}
ind_bs_cluster = np.empty(B)
boot_cluster = {k: np.empty(B) for k in
                ("a", "se", "ss", "c_direct", "b", "ta",
                 "adapt_ph", "adapt_di", "adapt_h")}
for i in range(B):
    rsel = rng.choice(np.arange(N_REGIONS), size=N_REGIONS, replace=True)
    idx = np.concatenate([region_members[r] for r in rsel])
    a_hat, _ = ols(Zph[idx], np.column_stack([Zh[idx], Zse[idx], Zss[idx]]))
    b_hat, _ = ols(Zdi[idx], np.column_stack([Zh[idx], Zph[idx], Zta[idx]]))
    bad_hat, _ = ols(Zad[idx], np.column_stack([Zph[idx], Zdi[idx], Zh[idx]]))
    ind_bs_cluster[i] = a_hat[1] * b_hat[2]
    boot_cluster["a"][i] = a_hat[1]
    boot_cluster["se"][i] = a_hat[2]
    boot_cluster["ss"][i] = a_hat[3]
    boot_cluster["c_direct"][i] = b_hat[1]
    boot_cluster["b"][i] = b_hat[2]
    boot_cluster["ta"][i] = b_hat[3]
    boot_cluster["adapt_ph"][i] = bad_hat[1]
    boot_cluster["adapt_di"][i] = bad_hat[2]
    boot_cluster["adapt_h"][i] = bad_hat[3]
ci_lo_cluster, ci_hi_cluster = np.percentile(ind_bs_cluster, [2.5, 97.5])


def cluster_ci(name):
    return [round(float(np.percentile(boot_cluster[name], 2.5)), 3),
            round(float(np.percentile(boot_cluster[name], 97.5)), 3)]

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

# ----------------------------------------------------------------------
# 5. Global gridded fields
# ----------------------------------------------------------------------
glat = np.linspace(-60, 78, 140)
glon = np.linspace(-180, 180, 280)
LON, LAT = np.meshgrid(glon, glat)
# Smooth synthetic hazard field: stronger toward tropics and monsoon/arid belts
latc = np.clip((np.abs(LAT) - 15) / 45, 0, 1)
hazard_field = 0.85 * np.exp(-((LAT - 18) / 22) ** 2) \
             + 0.55 * np.exp(-((LAT - 8) / 12) ** 2) \
             + 0.35 * np.exp(-((LAT + 5) / 18) ** 2) \
             + 0.30 * np.exp(-((LAT - 45) / 20) ** 2) \
             + 0.25 * (1 - latc)
hazard_field = hazard_field + rng.normal(0, 0.04, hazard_field.shape)
hazard_field = np.clip(hazard_field / hazard_field.max(), 0, 1)

ph_field = 1.0 - 0.72 * hazard_field \
           + 0.10 * np.exp(-((LAT + 25) / 30) ** 2) \
           - 0.05 * np.abs(LON) / 180
ph_field = ph_field + rng.normal(0, 0.05, ph_field.shape)
ph_field = (ph_field - ph_field.min()) / (ph_field.max() - ph_field.min())

# Projected psychological risk implied by the structural distress equation
# D = c' X + b P with covariates at their mean; P is held fixed and X amplified
future_hazard = np.clip(hazard_field * 1.35, 0, 1)
current_risk = c_direct * hazard_field + b * ph_field
future_risk = c_direct * future_hazard + b * ph_field
risk_field = (future_risk - future_risk.min()) / (future_risk.max() - future_risk.min())
current_risk_field = (current_risk - current_risk.min()) / (current_risk.max() - current_risk.min())
distress_field = current_risk_field
# Mean change in standardized distress and sensitivity to the warming factor
risk_mean_increase = round(float(future_risk.mean() - current_risk.mean()), 4)
risk_increase = {
    lam: round(float((c_direct * np.clip(hazard_field * lam, 0, 1)
                      + b * ph_field).mean() - current_risk.mean()), 4)
    for lam in (1.15, 1.25, 1.35, 1.50)
}

# ----------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------
LAND = load_land(os.path.join(DATADIR, "world_land.geojson"))
region_ph = np.array([Zph[region_idx == k].mean() for k in range(N_REGIONS)])
region_distress = np.array([Zdi[region_idx == k].mean() for k in range(N_REGIONS)])


def draw_coastlines(ax):
    for ring in LAND:
        x = [p[0] for p in ring]
        y = [p[1] for p in ring]
        ax.plot(x, y, color="white", linewidth=1.4, zorder=3,
                solid_joinstyle="round")
        ax.plot(x, y, color="0.15", linewidth=0.6, zorder=4,
                solid_joinstyle="round")


def box(ax, x, y, w, h, text, fc="#eaf1f8", ec="#2166ac", fs=6.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       fc=fc, ec=ec, lw=0.8, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=0.9, ls="-", label=None,
          lx=0, ly=0, rad=None):
    kw = dict(arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
              linestyle=ls, zorder=1)
    if rad is not None:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    a = FancyArrowPatch((x1, y1), (x2, y2), **kw)
    ax.add_patch(a)
    if label is not None:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=6,
                ha="center", va="center", color=color, zorder=4)


def draw_model(ax, a=None, b=None, cp=None):
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 7.7)
    ax.axis("off")
    box(ax, 0.3, 4.9, 3.4, 1.2, "Climate hazard\nexposure")
    box(ax, 6.5, 4.9, 3.4, 1.2, "Perceived\nhabitability")
    box(ax, 13.0, 4.9, 3.4, 1.2, "Psychological\ndistress")
    box(ax, 6.5, 0.6, 3.4, 1.2, "Adaptive\nresponse")
    box(ax, 0.3, 2.4, 3.4, 1.2, "Self-efficacy\nsocial support")
    box(ax, 13.0, 2.4, 3.4, 1.2, "Threat\nappraisal")
    la = f"a = {a:.2f}" if a is not None else "a"
    lb = f"b = {b:.2f}" if b is not None else "b"
    lc = f"c' = {cp:.2f}" if cp is not None else "c'"
    arrow(ax, 3.7, 5.5, 6.5, 5.5, label=la, ly=0.22)
    arrow(ax, 9.9, 5.5, 13.0, 5.5, label=lb, ly=0.22)
    arrow(ax, 1.6, 6.1, 15.8, 6.1, ls="--", label=lc, ly=1.2, rad=-0.06)
    arrow(ax, 8.2, 4.9, 8.2, 1.8, label="d", lx=0.18)
    arrow(ax, 13.0, 4.9, 9.9, 1.8, label="e", lx=-0.5, ly=0.15)
    arrow(ax, 2.0, 3.6, 7.0, 4.9, color="0.55")
    arrow(ax, 14.7, 3.6, 14.7, 4.9, color="0.55")


def field_map(ax, field, cmap, vmin, vmax, label):
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 78)
    cf = ax.pcolormesh(glon, glat, field, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto", rasterized=True)
    draw_coastlines(ax)
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-40, 0, 40, 70])
    ax.tick_params(length=2)
    ax.set_box_aspect(138 / 360)
    return cf


# ---- Figure 2: maps and gradient ------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.6, 3.8), constrained_layout=True)
ax = axes[0, 0]
cf = field_map(ax, hazard_field, "magma", 0, 1, "hazard")
panel_label(ax, "a")
cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.02)
cb.set_label("Climate hazard exposure index", fontsize=7)

ax = axes[0, 1]
cf = field_map(ax, ph_field, "cividis", 0, 1, "PH")
panel_label(ax, "b")
cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.02)
cb.set_label("Perceived habitability", fontsize=7)

ax = axes[1, 0]
ax.scatter(region_hazard, region_ph, s=18, c=PALETTE[0], alpha=0.7,
           edgecolor="white", linewidth=0.3)
coef = np.polyfit(region_hazard, region_ph, 1)
xs = np.linspace(0.1, 0.95, 50)
ax.plot(xs, np.polyval(coef, xs), color=PALETTE[1], lw=1.0)
ax.set_xlabel("Regional climate hazard exposure")
ax.set_ylabel("Regional mean perceived habitability (z)")
ax.set_xlim(0.1, 0.95)
ax.set_box_aspect(138 / 360)
panel_label(ax, "c")

ax = axes[1, 1]
cf = field_map(ax, distress_field, "inferno", 0, 1, "distress")
panel_label(ax, "d")
cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.02)
cb.set_label("Psychological distress", fontsize=7)
fig.savefig(os.path.join(FIGDIR, "fig2_maps.pdf"))
plt.close(fig)

# ---- Figure 3: model results ----------------------------------------------
fig = plt.figure(figsize=(7.6, 5.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                      height_ratios=[1.15, 1.0], hspace=0.65, wspace=0.5)
ax = fig.add_subplot(gs[0, :])
draw_model(ax, a=a, b=b, cp=c_direct)
panel_label(ax, "a")

ax = fig.add_subplot(gs[1, 0])
ax.hist(ind_bs, bins=40, color=PALETTE[0], alpha=0.85, density=True)
ax.axvline(indirect, color=PALETTE[1], lw=1.2)
ax.axvline(ci_lo, color=PALETTE[1], lw=0.8, ls=":")
ax.axvline(ci_hi, color=PALETTE[1], lw=0.8, ls=":")
ax.set_xlabel("Indirect effect")
ax.set_ylabel("Density")
panel_label(ax, "b")

ax = fig.add_subplot(gs[1, 1])
items = [f"Item {j+1}" for j in range(6)]
y = np.arange(6)
ax.hlines(y, 0, pca_loadings, color="0.65", lw=1.0)
ax.plot(pca_loadings, y, "o", color=PALETTE[0], markersize=5, zorder=3)
ax.errorbar(pca_loadings, y,
            xerr=[pca_loadings - loadings_ci_lo, loadings_ci_hi - pca_loadings],
            fmt="none", ecolor=PALETTE[1], lw=1.0, capsize=3, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(items)
ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.set_xlabel("Standardized loading")
panel_label(ax, "c")
fig.savefig(os.path.join(FIGDIR, "fig3_model.pdf"))
plt.close(fig)

# ---- Figure 4: continuity, model selection, and projected risk ------------
fig = plt.figure(figsize=(7.4, 5.6), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(Zph, Zdi, s=3, alpha=0.30, color=PALETTE[0])
plot_cov_ellipse(ax, gmm.means_[0], gmm.covariances_[0], color="k")
ax.set_xlabel("Perceived habitability (z)")
ax.set_ylabel("Distress (z)")
ax.set_box_aspect(1.0)
panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
ks = np.arange(1, 7)
ax.plot(ks, bics, "o-", color=PALETTE[0], markersize=4, lw=1.0)
ax.axvline(k_best, color=PALETTE[1], ls=":", lw=0.9)
ax.set_xticks(ks)
ax.set_xlabel("Number of components")
ax.set_ylabel("BIC")
ax.set_box_aspect(1.0)
panel_label(ax, "b")

ax = fig.add_subplot(gs[1, 0])
cf = field_map(ax, current_risk_field, "inferno", 0, 1, "current risk")
panel_label(ax, "c")
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Current psychological risk", fontsize=7)

ax = fig.add_subplot(gs[1, 1])
cf = field_map(ax, risk_field, "inferno", 0, 1, "risk")
panel_label(ax, "d")
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Projected psychological risk", fontsize=7)
fig.savefig(os.path.join(FIGDIR, "fig4_profiles.pdf"))
plt.close(fig)

# ----------------------------------------------------------------------
# 7. Results
# ----------------------------------------------------------------------
results = {
    "n_total": int(N),
    "n_regions": int(N_REGIONS),
    "ph_alpha": round(float(ph_alpha), 3),
    "ph_omega": round(float(ph_omega), 3),
    "pca_loadings": [round(float(x), 3) for x in pca_loadings],
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
    "indirect_ci": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
    "indirect_ci_cluster": [round(float(ci_lo_cluster), 3), round(float(ci_hi_cluster), 3)],
    "r2_ph": round(float(r2_ph), 3),
    "r2_distress": round(float(r2_distress), 3),
    "r2_adapt": round(float(r2_adapt), 3),
    "adapt_beta": [round(float(x), 3) for x in beta_adapt[1:]],
    "beta_a": [round(float(x), 3) for x in beta_a],
    "beta_b": [round(float(x), 3) for x in beta_b],
    "icc": {"ph": icc_ph, "distress": icc_distress, "adapt": icc_adapt},
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
        "b": cluster_ci("b"),
        "threat_appraisal": cluster_ci("ta"),
        "adapt_ph": cluster_ci("adapt_ph"),
        "adapt_di": cluster_ci("adapt_di"),
        "adapt_h": cluster_ci("adapt_h"),
    },
    "hazard_ph_corr": round(float(np.corrcoef(region_hazard, region_ph)[0, 1]), 3),
    "risk_mean_increase": risk_mean_increase,
    "risk_increase_sensitivity": risk_increase,
    "regions": [
        {"name": region_names[k], "lat": float(region_lat[k]),
         "lon": float(region_lon[k]), "hazard": round(float(region_hazard[k]), 3),
         "ph_mean": round(float(region_ph[k]), 3),
         "distress_mean": round(float(region_distress[k]), 3)}
        for k in range(N_REGIONS)
    ],
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

import csv
with open(os.path.join(DATADIR, "regions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["region", "lat", "lon", "hazard", "ph_mean", "distress_mean"])
    for k in range(N_REGIONS):
        w.writerow([region_names[k], round(float(region_lat[k]), 2),
                    round(float(region_lon[k]), 2),
                    round(float(region_hazard[k]), 3),
                    round(float(region_ph[k]), 3),
                    round(float(region_distress[k]), 3)])

print(json.dumps(results, indent=2))
