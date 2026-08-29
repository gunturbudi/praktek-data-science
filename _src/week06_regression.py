# %% [markdown]
# # Praktikum 6 — Regression
#
# **Data Science · Magister Ilmu Komputer, FMIPA UGM**
# Companion to **Chapter 6**. Assignment weight: **3%**.
#
# ---
#
# ## What you will do
#
# 1. Derive the OLS estimator three ways and confirm they agree.
# 2. Diagnose all five Gauss–Markov assumptions on real residuals.
# 3. Watch multicollinearity destroy coefficients while predictions stay fine.
# 4. Compare ridge, lasso and elastic net, and see where each wins.
# 5. Fit logistic regression and read coefficients as **odds ratios**.
# 6. Choose a decision threshold from **costs**, not from the default 0.5.
#
# **The datasets.** `data/rumah_yogya.csv` (2,200 house listings) and
# `data/credit.csv`.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "checks.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import checks
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

DATA = ROOT / "data"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titleweight": "bold"})
pd.set_option("display.width", 110)

# %%
rumah = pd.read_csv(DATA / "rumah_yogya.csv")
print(rumah.shape)
print(rumah.describe().T.round(1).to_string())
rumah.head()

# %% [markdown]
# ## 1. Ordinary least squares, three ways  (DEMO)
#
# OLS minimises $\text{RSS} = \|\mathbf{y} - \mathbf{X}\boldsymbol\beta\|^2$.
# Differentiating and setting to zero gives the **normal equations**, hence
#
# $$\hat{\boldsymbol\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$$
#
# Let us confirm that the closed form, a numerical solver, and `scikit-learn`
# all agree — and then see why nobody uses the closed form in practice.

# %%
from sklearn.linear_model import LinearRegression

X_simple = rumah[["luas_bangunan"]].to_numpy()
y_simple = rumah["harga"].to_numpy()

# --- route 1: the closed form, by hand -------------------------------
Xd = np.column_stack([np.ones(len(X_simple)), X_simple])
beta_closed = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y_simple

# --- route 2: the scalar formula, slope = Cov(x,y)/Var(x) ------------
x = X_simple.ravel()
slope = np.cov(x, y_simple, ddof=1)[0, 1] / np.var(x, ddof=1)
intercept = y_simple.mean() - slope * x.mean()

# --- route 3: scikit-learn (which uses the SVD, not the inverse) -----
lr = LinearRegression().fit(X_simple, y_simple)

print(f"closed form : intercept {beta_closed[0]:9.4f}   slope {beta_closed[1]:.6f}")
print(f"Cov/Var     : intercept {intercept:9.4f}   slope {slope:.6f}")
print(f"scikit-learn: intercept {lr.intercept_:9.4f}   slope {lr.coef_[0]:.6f}")

# %% [markdown]
# The slope also equals $r \cdot s_y / s_x$ — the correlation scaled by the ratio
# of standard deviations. That identity is why **regression to the mean** happens:
# since $|r| \le 1$, a value one SD above the mean in $x$ predicts *fewer* than
# one SD above the mean in $y$.

# %%
r = np.corrcoef(x, y_simple)[0, 1]
print(f"r * s_y/s_x = {r * y_simple.std(ddof=1) / x.std(ddof=1):.6f}")
print(f"fitted slope = {slope:.6f}")
print(f"\ncorrelation r = {r:.4f}: a house 1 SD above average in size is")
print(f"predicted at only {r:.2f} SD above average in price.")

# %%
# Two properties worth verifying: residuals sum to zero, and the line
# passes through the centroid.
resid = y_simple - lr.predict(X_simple)
print(f"sum of residuals      : {resid.sum():.6e}  (should be ~0)")
print(f"line at mean(x)       : {lr.predict([[x.mean()]])[0]:.4f}")
print(f"mean(y)               : {y_simple.mean():.4f}")

# %% [markdown]
# > **Nobody computes $(X^\top X)^{-1}$ in practice.** Forming the inverse squares
# > the condition number and loses precision. Numerical libraries solve the least
# > squares problem by QR or SVD instead. The closed form is for understanding.

# %%
# Demonstrating the numerical hazard on a near-singular design.
rng = np.random.default_rng(0)
a = rng.normal(size=200)
Xbad = np.column_stack([np.ones(200), a, a + rng.normal(0, 1e-7, 200)])
ybad = 3 + 2 * a + rng.normal(0, 0.1, 200)

cond = np.linalg.cond(Xbad.T @ Xbad)
print(f"condition number of X'X : {cond:.3e}")
try:
    b_inv = np.linalg.inv(Xbad.T @ Xbad) @ Xbad.T @ ybad
    print("via explicit inverse    :", np.round(b_inv, 4))
except np.linalg.LinAlgError as e:
    print("via explicit inverse    : FAILED —", e)
b_lstsq, *_ = np.linalg.lstsq(Xbad, ybad, rcond=None)
print("via lstsq (SVD)         :", np.round(b_lstsq, 4))

# %% [markdown]
# ## 2. Multiple regression and the assumptions  (DEMO)
#
# The Gauss–Markov theorem: under assumptions 1–4, OLS is the **Best Linear
# Unbiased Estimator**. Assumption 5 (normality) is needed only for small-sample
# inference — *not* for the estimator to be BLUE.
#
# Note what is **not** assumed: normality of $y$, or of the predictors.

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

feat_num = ["luas_bangunan", "luas_tanah", "kamar_tidur", "kamar_mandi",
            "umur_bangunan", "jarak_pusat_km"]
X = pd.get_dummies(rumah[feat_num + ["kecamatan", "sertifikat"]],
                   drop_first=True).astype(float)
yv = rumah["harga"]

Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=0.2, random_state=42)
model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)

print(f"held-out RMSE : {mean_squared_error(yte, pred) ** 0.5:8.2f} juta")
print(f"held-out MAE  : {mean_absolute_error(yte, pred):8.2f} juta")
print(f"held-out R^2  : {r2_score(yte, pred):8.4f}")

# %% [markdown]
# ### Residual diagnostics
#
# The residuals-against-fitted plot is the single most informative diagnostic.
# Under a correct model it shows a formless horizontal band.

# %%
fitted = model.predict(Xtr)
res = ytr - fitted
std_res = res / res.std()

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

axes[0].scatter(fitted, res, s=8, alpha=0.35, color="#2E6F9E")
axes[0].axhline(0, color="#9C3535", lw=1)
axes[0].set(title="Residuals vs fitted", xlabel="fitted (juta)", ylabel="residual")

stats.probplot(std_res, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q")
axes[1].get_lines()[0].set(markersize=3, alpha=0.4, color="#2E6F9E")
axes[1].get_lines()[1].set(color="#9C3535")

axes[2].scatter(fitted, np.sqrt(np.abs(std_res)), s=8, alpha=0.35,
                color="#2E6F9E")
axes[2].set(title="Scale-location", xlabel="fitted",
            ylabel=r"$\sqrt{|standardised\ residual|}$")
plt.show()

# %% [markdown]
# The funnel shape in panels 1 and 3 is **heteroscedasticity**: the error
# variance grows with the fitted value. That is exactly what you expect when the
# true relationship is multiplicative and you fit it additively.
#
# Formal test, without `statsmodels`:

# %%
def breusch_pagan(residuals, X_design):
    """LM test for heteroscedasticity: regress squared residuals on X."""
    e2 = residuals ** 2
    Xd = np.column_stack([np.ones(len(X_design)), np.asarray(X_design, float)])
    beta, *_ = np.linalg.lstsq(Xd, e2, rcond=None)
    ss_res = ((e2 - Xd @ beta) ** 2).sum()
    ss_tot = ((e2 - e2.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    lm = len(e2) * r2
    return lm, 1 - stats.chi2.cdf(lm, Xd.shape[1] - 1)


lm, p_bp = breusch_pagan(res.to_numpy(), Xtr)
print(f"Breusch-Pagan LM = {lm:.2f}, p = {p_bp:.3e}")
print("-> heteroscedasticity confirmed. Coefficients remain UNBIASED;")
print("   only the standard errors are wrong.")

# %% [markdown]
# ### The fix: model on the log scale
#
# Chapter 3 showed price is log-normal and its relationship to area is a power
# law. Modelling $\log(\text{harga})$ against $\log(\text{luas})$ makes the
# relationship linear *and* the errors homoscedastic.

# %%
Xlog = X.copy()
Xlog["luas_bangunan"] = np.log(Xlog["luas_bangunan"])
Xlog["luas_tanah"] = np.log(Xlog["luas_tanah"])
ylog = np.log(yv)

Xtr_l, Xte_l, ytr_l, yte_l = train_test_split(Xlog, ylog, test_size=0.2,
                                              random_state=42)
model_log = LinearRegression().fit(Xtr_l, ytr_l)

res_log = ytr_l - model_log.predict(Xtr_l)
lm2, p_bp2 = breusch_pagan(res_log.to_numpy(), Xtr_l)

print(f"R^2 on the log scale     : {model_log.score(Xte_l, yte_l):.4f}")
print(f"Breusch-Pagan p, raw     : {p_bp:.3e}")
print(f"Breusch-Pagan p, log     : {p_bp2:.3e}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].scatter(model.predict(Xtr), res, s=7, alpha=0.3, color="#9C3535")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set(title="Raw scale: funnel", xlabel="fitted", ylabel="residual")
axes[1].scatter(model_log.predict(Xtr_l), res_log, s=7, alpha=0.3,
                color="#356B3F")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set(title="Log scale: formless band", xlabel="fitted (log)")
plt.show()

# %% [markdown]
# ### TASK 6.1
#
# Report the held-out $R^2$ of the **log-scale** model.

# %%
r2_log = None

checks.task("6.1", r2_log)

# %% [markdown]
# > **Careful with back-transformation.** A coefficient of 0.08 on a log target
# > means "about 8% higher", not "0.08 higher". And $\exp(\mathbb{E}[\log y])$
# > is the *geometric* mean, which sits below the arithmetic one.

# %%
coefs_log = pd.Series(model_log.coef_, index=Xlog.columns)
print("Log-model coefficients, read as elasticities / percentage effects:\n")
print(f"log(luas_bangunan)  {coefs_log['luas_bangunan']:+.4f}"
      f"  -> a 1% larger building implies "
      f"{coefs_log['luas_bangunan']:.2f}% higher price (elasticity)")
print(f"umur_bangunan       {coefs_log['umur_bangunan']:+.4f}"
      f"  -> each extra year is "
      f"{100*(np.exp(coefs_log['umur_bangunan'])-1):+.2f}% on price")
print(f"jarak_pusat_km      {coefs_log['jarak_pusat_km']:+.4f}"
      f"  -> each extra km is "
      f"{100*(np.exp(coefs_log['jarak_pusat_km'])-1):+.2f}% on price")

# %% [markdown]
# ## 3. Multicollinearity  (DEMO)
#
# Measured by the **variance inflation factor**
# $\text{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2$ comes from regressing $x_j$ on
# all the other predictors. The name is literal: the variance of
# $\hat\beta_j$ is inflated by exactly this factor.

# %%
def vif_table(df):
    out = {}
    cols = list(df.columns)
    for c in cols:
        others = [o for o in cols if o != c]
        r2 = LinearRegression().fit(df[others], df[c]).score(df[others], df[c])
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out).sort_values(ascending=False)


print("VIF on the house predictors:")
print(vif_table(rumah[feat_num]).round(2).to_string())

# %%
# Now inject a near-duplicate and watch what happens.
rumah_bad = rumah.copy()
rumah_bad["luas_bangunan_m2"] = (rumah_bad["luas_bangunan"]
                                 + rng.normal(0, 0.5, len(rumah_bad)))

cols_bad = feat_num + ["luas_bangunan_m2"]
print("VIF with a near-duplicate column:")
print(vif_table(rumah_bad[cols_bad]).round(1).to_string())

m_clean = LinearRegression().fit(rumah[feat_num], rumah["harga"])
m_bad = LinearRegression().fit(rumah_bad[cols_bad], rumah_bad["harga"])

print(f"\ncoefficient on luas_bangunan, clean : {m_clean.coef_[0]:>12.3f}")
print(f"coefficient on luas_bangunan, with duplicate: "
      f"{m_bad.coef_[0]:>12.3f}")
print(f"coefficient on the duplicate               : "
      f"{m_bad.coef_[-1]:>12.3f}")
print(f"their SUM                                  : "
      f"{m_bad.coef_[0] + m_bad.coef_[-1]:>12.3f}")
print(f"\nR^2 clean {m_clean.score(rumah[feat_num], rumah['harga']):.4f}   "
      f"R^2 with duplicate {m_bad.score(rumah_bad[cols_bad], rumah_bad['harga']):.4f}")
print("\nThe individual coefficients are meaningless; their SUM is stable;")
print("and the PREDICTIONS are unaffected. That is multicollinearity exactly.")

# %% [markdown]
# ## 4. Regularisation  (DEMO)
#
# Expected error decomposes as
# $\text{bias}^2 + \text{variance} + \sigma^2_{\text{irreducible}}$.
# Regularisation buys a reduction in variance at the price of a little bias.
#
# - **Ridge** ($\ell_2$): shrinks, never to exactly zero; tames collinearity.
# - **Lasso** ($\ell_1$): sets coefficients **exactly** to zero — it selects.
# - **Elastic net**: both; keeps correlated groups together.

# %%
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# A hard case: p large relative to n, and the truth is sparse.
n_s, p_s = 200, 60
Xs = rng.normal(size=(n_s, p_s))
Xs[:, 1] = Xs[:, 0] + rng.normal(0, 0.1, n_s)     # near-duplicates of feature 0
Xs[:, 2] = Xs[:, 0] + rng.normal(0, 0.1, n_s)
beta_true = np.zeros(p_s)
beta_true[:5] = [3.0, 0.0, 0.0, -2.0, 1.5]
ys = Xs @ beta_true + rng.normal(0, 2.0, n_s)

Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.3,
                                              random_state=0)
alphas = np.logspace(-3, 3, 60)

models = {
    "OLS": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
    "Lasso": make_pipeline(StandardScaler(),
                           LassoCV(alphas=alphas, max_iter=50_000,
                                   random_state=0)),
    "ElasticNet": make_pipeline(StandardScaler(),
                                ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                             alphas=alphas, max_iter=50_000,
                                             random_state=0)),
}
rows = []
for name, m in models.items():
    m.fit(Xs_tr, ys_tr)
    coef = m[-1].coef_
    rows.append({"model": name,
                 "test_RMSE": mean_squared_error(ys_te, m.predict(Xs_te)) ** 0.5,
                 "non_zero": int((np.abs(coef) > 1e-8).sum())})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nirreducible noise floor: sigma = 2.0")
print("Only 3 of the 60 coefficients are genuinely non-zero.")

# %% [markdown]
# Ridge and lasso are **statistically indistinguishable** here — the difference
# is well inside the noise of a 60-observation test set — while being radically
# different as artefacts. Ridge delivers 60 small coefficients; lasso delivers a
# handful, including the true signals.
#
# If the deliverable is a *prediction*, either will do. If it is an
# *explanation*, or a model that must be maintained and audited, seven
# coefficients is a different kind of object from sixty.
#
# ### The coefficient path
#
# Watch lasso zero things out as the penalty grows.

# %%
from sklearn.linear_model import lasso_path, ridge_regression

Xs_std = StandardScaler().fit_transform(Xs)
alpha_grid, coefs_path, _ = lasso_path(Xs_std, ys, alphas=np.logspace(-2, 1, 60))
ridge_path = np.array([ridge_regression(Xs_std, ys, alpha=a)
                       for a in np.logspace(-2, 4, 60)]).T

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for i in range(p_s):
    hot = i in (0, 3, 4)
    axes[0].plot(alpha_grid, coefs_path[i],
                 color="#9C3535" if hot else "#B8C4D0",
                 lw=1.6 if hot else 0.6, zorder=3 if hot else 1)
    axes[1].plot(np.logspace(-2, 4, 60), ridge_path[i],
                 color="#9C3535" if hot else "#B8C4D0",
                 lw=1.6 if hot else 0.6, zorder=3 if hot else 1)
for ax, title in zip(axes, ["Lasso: coefficients hit exactly zero",
                            "Ridge: shrinks, never reaches zero"]):
    ax.set(xscale="log", xlabel=r"penalty $\lambda$", ylabel="coefficient",
           title=title)
    ax.axhline(0, color="black", lw=0.6)
fig.suptitle("Red = the three genuinely non-zero coefficients", y=1.05,
             fontsize=9, color="#9C3535")
plt.show()

# %% [markdown]
# ### TASK 6.2
#
# Which penalty performs **variable selection** by setting coefficients exactly
# to zero?

# %%
penalti = None       # "ridge", "lasso" or "elasticnet"

checks.task("6.2", penalti)

# %% [markdown]
# ## 5. Logistic regression  (DEMO)
#
# Linear regression is unsuitable for a binary target: it predicts outside
# $[0,1]$ and its error structure is wrong. Model the **log-odds** instead:
#
# $$\log\frac{p}{1-p} = \mathbf{x}^\top\boldsymbol\beta
# \qquad\Longleftrightarrow\qquad
# p = \frac{1}{1 + e^{-\mathbf{x}^\top\boldsymbol\beta}}$$

# %%
fig, ax = plt.subplots(figsize=(5.5, 3.2))
z = np.linspace(-6, 6, 300)
ax.plot(z, 1 / (1 + np.exp(-z)), color="#173F5F", lw=2)
ax.axhline(0.5, color="grey", ls=":", lw=1)
ax.axvline(0, color="grey", ls=":", lw=1)
ax.set(xlabel=r"log-odds  $\mathbf{x}^\top\beta$", ylabel="probability",
       title="The logistic (sigmoid) function")
ax.annotate("nearly linear here", xy=(0, 0.5), xytext=(2.2, 0.32),
            arrowprops=dict(arrowstyle="->", color="#9C3535"), fontsize=8,
            color="#9C3535")
ax.annotate("very flat here", xy=(-4.5, 0.011), xytext=(-5.8, 0.28),
            arrowprops=dict(arrowstyle="->", color="#9C3535"), fontsize=8,
            color="#9C3535")
plt.show()

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                             confusion_matrix, roc_curve)

credit = pd.read_csv(DATA / "credit.csv")
yc = (credit["default"] == "yes").astype(int)
Xc = credit.drop(columns=["default"]).copy()
Xc["dti"] = Xc["debt"] / Xc["income"].clip(lower=1)
Xc["pti"] = Xc["monthly_payment"] / (Xc["income"] / 12).clip(lower=1)
Xc["util"] = Xc["balance"] / Xc["credit_limit"].clip(lower=1)

num_c = Xc.select_dtypes("number").columns.tolist()
cat_c = Xc.select_dtypes("object").columns.tolist()

prep = ColumnTransformer([
    ("n", Pipeline([("i", SimpleImputer(strategy="median")),
                    ("s", StandardScaler())]), num_c),
    ("c", Pipeline([("i", SimpleImputer(strategy="constant",
                                        fill_value="Unknown")),
                    ("o", OneHotEncoder(handle_unknown="ignore",
                                        sparse_output=False))]), cat_c),
], verbose_feature_names_out=False)

clf = Pipeline([("prep", prep),
                ("lr", LogisticRegression(max_iter=5000, C=1.0))])

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25,
                                              stratify=yc, random_state=42)
clf.fit(Xc_tr, yc_tr)
proba = clf.predict_proba(Xc_te)[:, 1]
pred_c = (proba >= 0.5).astype(int)

print(f"accuracy : {accuracy_score(yc_te, pred_c):.4f}")
print(f"ROC-AUC  : {roc_auc_score(yc_te, proba):.4f}")
print(f"log loss : {log_loss(yc_te, proba):.4f}")
print(f"\nbase rate (always predict 'no default'): "
      f"{1 - yc_te.mean():.4f}  <- compare with accuracy!")
print("\nconfusion matrix (rows=true, cols=predicted):")
print(confusion_matrix(yc_te, pred_c))

# %% [markdown]
# Accuracy 0.837 sounds respectable until you notice the base rate is 0.800.
# The model buys only 3.7 accuracy points over "always predict no default" —
# while having a genuinely useful ROC-AUC of 0.82. Accuracy is a poor lens here
# because it depends entirely on the 0.5 threshold; ROC-AUC integrates over all
# thresholds. This is the base-rate trap from week 1, and it is why we read the
# confusion matrix and choose a threshold from costs (section 6).
#
# ### Coefficients as odds ratios

# %%
names = clf.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(clf.named_steps["lr"].coef_[0], index=names)
odds = np.exp(coefs)

top = pd.DataFrame({"coef": coefs, "odds_ratio": odds})
top = top.reindex(top["coef"].abs().sort_values(ascending=False).index).head(10)
print("Largest effects (features standardised, so these are comparable):\n")
print(top.round(4).to_string())

# %% [markdown]
# $e^{\beta_j}$ is the multiplicative change in the **odds** for a one-SD
# increase — *not* in the probability. The distinction matters enormously:

# %%
beta = 0.693        # exactly doubles the odds
for p0 in (0.05, 0.20, 0.50, 0.80):
    odds0 = p0 / (1 - p0)
    p1 = (odds0 * np.exp(beta)) / (1 + odds0 * np.exp(beta))
    print(f"baseline p = {p0:.2f}  ->  p = {p1:.4f}   "
          f"(change {100*(p1-p0):+.2f} percentage points)")

print("\nThe SAME doubling of the odds moves the probability by 8 points at")
print("p=0.05 and by 17 points at p=0.50. Never say 'the probability doubles'.")

# %% [markdown]
# ## 6. Choosing the threshold from costs  (DEMO)
#
# A classifier outputs a probability. Converting it to a decision needs a
# threshold, and **0.5 is a default, not an answer**.
#
# Acting positive costs $(1-p)C_{FP}$; acting negative costs $p\,C_{FN}$.
# Acting positive is preferable when
#
# $$p > \frac{C_{FP}}{C_{FP} + C_{FN}}$$

# %%
def total_cost(y_true, proba, thr, c_fp, c_fn):
    tn, fp, fn, tp = confusion_matrix(y_true,
                                      (proba >= thr).astype(int)).ravel()
    return fp * c_fp + fn * c_fn


C_FP, C_FN = 1.0, 10.0       # missing a default costs 10x a needless rejection
theory = C_FP / (C_FP + C_FN)

grid = np.linspace(0.02, 0.98, 97)
costs = [total_cost(yc_te, proba, t, C_FP, C_FN) for t in grid]
best = grid[int(np.argmin(costs))]

print(f"cost ratio C_FN/C_FP = {C_FN/C_FP:.0f}")
print(f"theoretical threshold = {theory:.4f}")
print(f"empirical minimum     = {best:.4f}")
print(f"\ncost at threshold 0.5  : {total_cost(yc_te, proba, 0.5, C_FP, C_FN):,.0f}")
print(f"cost at threshold {best:.2f} : "
      f"{total_cost(yc_te, proba, best, C_FP, C_FN):,.0f}")
print(f"saving: "
      f"{1 - total_cost(yc_te, proba, best, C_FP, C_FN) / total_cost(yc_te, proba, 0.5, C_FP, C_FN):.1%}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

axes[0].plot(grid, costs, color="#173F5F")
axes[0].axvline(best, color="#356B3F", ls="--",
                label=f"empirical min {best:.2f}")
axes[0].axvline(theory, color="#9C3535", ls=":",
                label=f"theory {theory:.2f}")
axes[0].axvline(0.5, color="grey", ls="-.", label="default 0.5")
axes[0].set(xlabel="threshold", ylabel="total cost",
            title=f"Cost curve (C_FN = {C_FN:.0f} x C_FP)")
axes[0].legend(fontsize=8, frameon=False)

fpr, tpr, _ = roc_curve(yc_te, proba)
axes[1].plot(fpr, tpr, color="#2E6F9E",
             label=f"AUC = {roc_auc_score(yc_te, proba):.3f}")
axes[1].plot([0, 1], [0, 1], ls=":", color="grey")
axes[1].set(xlabel="false positive rate", ylabel="true positive rate",
            title="ROC curve")
axes[1].legend(fontsize=8, frameon=False)
plt.show()

# %% [markdown]
# ### TASK 6.3
#
# If a false negative costs **10×** a false positive, what is the theoretically
# optimal threshold?

# %%
ambang = None

checks.task("6.3", ambang)

# %% [markdown]
# > This calculation requires **calibrated** probabilities: among instances
# > predicted at 0.3, roughly 30% must actually be positive. Logistic regression
# > is usually well calibrated by construction; tree ensembles and SVMs are not,
# > and need `CalibratedClassifierCV` first.

# %%
# A quick calibration check: bin the predictions and compare with reality.
bins = pd.qcut(proba, 10, duplicates="drop")
cal = pd.DataFrame({"predicted": proba, "actual": yc_te.to_numpy(),
                    "bin": bins}).groupby("bin", observed=True).agg(
    mean_predicted=("predicted", "mean"),
    observed_rate=("actual", "mean"),
    n=("actual", "size"))
print(cal.round(4).to_string())

fig, ax = plt.subplots(figsize=(4.2, 4))
ax.plot([0, cal["mean_predicted"].max()], [0, cal["mean_predicted"].max()],
        ls=":", color="grey", label="perfect calibration")
ax.plot(cal["mean_predicted"], cal["observed_rate"], marker="o", ms=5,
        color="#2E6F9E", label="logistic regression")
ax.set(xlabel="mean predicted probability", ylabel="observed frequency",
       title="Calibration plot")
ax.legend(fontsize=8, frameon=False)
plt.show()

# %% [markdown]
# ## 7. Take-home  (assessed)
#
# **A. Derive it.** Derive $\hat\beta_1 = \text{Cov}(x,y)/\text{Var}(x)$ from the
# two normal equations, showing every step. Then show the residuals sum to zero
# whenever an intercept is included, and explain the **geometric** meaning.
#
# **B. Diagnose four violations.** For each, state the consequence for the
# coefficients, for their standard errors, and the appropriate remedy:
# (a) heteroscedastic errors; (b) an omitted variable correlated with an
# included one; (c) VIF of 25 on two predictors; (d) heavy-tailed residuals at
# $n = 5000$. Which one is the *serious* case, and why?
#
# **C. Sparsity and lasso.** Reproduce section 4, then change the number of
# truly relevant predictors from 3 to 40 and rerun. Report which method now wins
# and explain the result in terms of the sparsity assumption underlying lasso.
#
# **D. Odds vs probability.** A logistic model of default on income (in millions
# of rupiah) gives $\beta = -0.35$. Compute the odds ratio and interpret it in
# one sentence. Then compute the change in default *probability* for a
# one-million increase for applicants at baseline (a) 0.05 and (b) 0.40, and
# explain why the two answers differ.
#
# **E. Cost-sensitive deployment.** A fraud model's costs are: investigating a
# legitimate transaction Rp 50,000; missing a fraudulent one Rp 4,000,000.
# Compute the optimal threshold, then state what additional property the model's
# outputs must have for that threshold to be valid — and how you would check it.
#
# ---
# **Next week:** the classification toolkit, compared fairly.
