# %% [markdown]
# # Praktikum 5 — Feature Engineering and Dimensionality Reduction
#
# **Data Science · Magister Ilmu Komputer, FMIPA UGM**
# Companion to **Chapter 5**. Assignment weight: **3%**.
#
# ---
#
# ## What you will do
#
# 1. Encode categorical variables four ways, and break one of them on purpose.
# 2. See which models need scaling and which do not.
# 3. Build RFM features from a transaction log.
# 4. Select features by filter, wrapper and embedded methods.
# 5. Derive PCA numerically and read its loadings.
# 6. **Measure the cost of data leakage** — the most important cell in this course.
#
# **The datasets.** `data/transactions.csv` (30,000 transactions) and
# `data/credit.csv` (4,000 loan applications with a default flag).

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
import matplotlib.pyplot as plt

DATA = ROOT / "data"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titleweight": "bold"})
pd.set_option("display.width", 110)

# %% [markdown]
# ## 0. Why representation is the problem  (DEMO)
#
# Every model class has an **inductive bias** — a set of relationships it can
# express easily. Feature engineering is re-expressing the data so the
# relationship you care about is one the model can see.
#
# Here the truth is that risk depends on `debt / income`. Watch what one
# division does.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
n = 2000
income = rng.lognormal(16, 0.5, n)
debt = income * rng.beta(2, 3, n) * 3
risk = (debt / income > 0.9).astype(int)          # the truth is a RATIO

cv = StratifiedKFold(5, shuffle=True, random_state=0)
raw = pd.DataFrame({"debt": debt, "income": income})
eng = raw.assign(dti=debt / income)

for name, X in [("debt + income (raw)", raw), ("+ debt/income ratio", eng)]:
    s = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()),
                        X, risk, cv=cv, scoring="roc_auc")
    print(f"{name:<22} ROC-AUC = {s.mean():.4f}")

print("\nNo amount of hyperparameter tuning substitutes for that one division.")

# %% [markdown]
# ## 1. Encoding categorical variables  (DEMO)

# %%
credit = pd.read_csv(DATA / "credit.csv")
print(credit.shape)
print(credit.dtypes.to_string())
credit.head(3)

# %%
print("Cardinality of each categorical column:")
for c in credit.select_dtypes("object").columns:
    print(f"  {c:<18} {credit[c].nunique():>3} levels  "
          f"{sorted(credit[c].dropna().unique())}")

# %% [markdown]
# ### One-hot: the default for nominal variables

# %%
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="infrequent_if_exist",
                    min_frequency=0.01, sparse_output=False)
oh = enc.fit_transform(credit[["purpose", "housing"]].fillna("Unknown"))
print("shape:", oh.shape)
print("features:", list(enc.get_feature_names_out()))

# %% [markdown]
# Three parameters carry the practical weight:
#
# - `drop="first"` — needed **only for OLS**, where the dummies otherwise sum to
#   a column of ones and make $X^\top X$ singular. Dropping hurts tree feature
#   importances, so keep it for everything else.
# - `handle_unknown` — a level *will* appear at prediction time that was absent
#   at training time. Without this, `transform` raises and your service 500s.
# - `min_frequency` — pools rare levels, preventing a column that is 1-in-20,000
#   and pure noise.
#
# ### Ordinal: only for genuinely ordered variables

# %%
from sklearn.preprocessing import OrdinalEncoder

# WRONG: let the encoder sort alphabetically
wrong = OrdinalEncoder().fit(credit[["education"]].fillna("SMA"))
print("alphabetical order imposed:", list(wrong.categories_[0]))

# RIGHT: state the order explicitly
order = [["SMA", "D3", "S1", "S2"]]
right = OrdinalEncoder(categories=order, handle_unknown="use_encoded_value",
                       unknown_value=-1).fit(credit[["education"]].fillna("SMA"))
print("explicit order        :", list(right.categories_[0]))

print("\nAlphabetically, D3 < S1 < S2 < SMA -- which says a high-school")
print("graduate outranks a master's. Always pass `categories`.")

# %% [markdown]
# ### Target encoding: powerful, and the easiest way to leak
#
# The naive version, $\text{enc}(k) = \bar{y}_k$, **memorises**: a level
# appearing once is encoded with that row's own target value.

# %%
demo = pd.DataFrame({
    "kota": ["Yogya", "Yogya", "Solo", "Solo", "Klaten"],   # Klaten appears once
    "y":    [1, 0, 1, 1, 1],
})
naive_map = demo.groupby("kota")["y"].mean()
print("naive target encoding:")
print(naive_map.to_string())
print("\nKlaten -> 1.0, which IS that row's own label. A model reading this")
print("column achieves perfect training accuracy and learns nothing.\n")

# Smoothing shrinks small groups towards the global mean.
def smooth_encode(counts, means, global_mean, m=20):
    return (counts * means + m * global_mean) / (counts + m)

g = demo.groupby("kota")["y"].agg(["count", "mean"])
g["smoothed"] = smooth_encode(g["count"], g["mean"], demo["y"].mean(), m=20)
print("with smoothing (m=20):")
print(g.round(4).to_string())
print("\nKlaten now sits near the global mean: one observation is almost no evidence.")

# %%
# scikit-learn's TargetEncoder does smoothing AND cross-fitting by default.
from sklearn.preprocessing import TargetEncoder

y = (credit["default"] == "yes").astype(int)
te = TargetEncoder(smooth="auto", cv=5, random_state=0)
te_out = te.fit_transform(credit[["purpose"]].fillna("Unknown"), y)
print("target-encoded 'purpose', first 5:", np.round(te_out[:5, 0], 4))
print("global default rate              :", round(y.mean(), 4))

# %% [markdown]
# ## 2. Scaling: who needs it  (DEMO)
#
# Scaling matters when the model is **distance- or variance-based**, or
# **penalised**. It is irrelevant to trees, whose splits depend only on the
# order of values.

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X_demo = credit[["age", "income", "debt"]].fillna(credit[["age", "income", "debt"]].median())
print("Feature ranges -- income is ~7 orders of magnitude larger than age:")
print(X_demo.describe().T[["min", "max"]].round(0).to_string())

print("\nEffect of scaling on cross-validated ROC-AUC:\n")
for name, model in [("k-NN", KNeighborsClassifier(15)),
                    ("SVM (RBF)", SVC(probability=False)),
                    ("Random forest", RandomForestClassifier(
                        n_estimators=60, random_state=0))]:
    unscaled = cross_val_score(model, X_demo, y, cv=cv, scoring="roc_auc").mean()
    scaled = cross_val_score(make_pipeline(StandardScaler(), model),
                             X_demo, y, cv=cv, scoring="roc_auc").mean()
    print(f"  {name:<16} unscaled {unscaled:.4f}   scaled {scaled:.4f}   "
          f"delta {scaled - unscaled:+.4f}")

print("\nDistance-based methods are transformed by scaling. The forest does not")
print("move -- splits depend on order, which scaling preserves.")

# %% [markdown]
# ## 3. Creating features from domain reasoning  (DEMO)
#
# This is the part that cannot be automated. We build a per-customer modelling
# table from a raw transaction log — the standard route from transactional data
# to a rectangular dataset.

# %%
tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["ts"])
print(tx.shape)
tx.head()

# %%
asof = tx["ts"].max()
print("feature cutoff (as-of date):", asof)

feats = tx.groupby("customer_id").agg(
    n_tx=("amount", "size"),
    total=("amount", "sum"),
    mean_amt=("amount", "mean"),
    std_amt=("amount", "std"),
    max_amt=("amount", "max"),
    first_ts=("ts", "min"),
    last_ts=("ts", "max"),
    n_categories=("category", "nunique"),
)

# Recency, Frequency, Monetary -- the classic RFM triple
feats["recency_days"] = (asof - feats["last_ts"]).dt.days
feats["tenure_days"] = (asof - feats["first_ts"]).dt.days
feats["tx_per_month"] = feats["n_tx"] / (feats["tenure_days"] / 30).clip(lower=1)

# Dispersion relative to level: a scale-free measure of erratic spending
feats["cv_amt"] = feats["std_amt"] / feats["mean_amt"]

# Category concentration: is this a specialist or a generalist?
share = tx.pivot_table(index="customer_id", columns="category",
                       values="amount", aggfunc="sum", fill_value=0)
feats["top_cat_share"] = share.max(axis=1) / share.sum(axis=1)

# Trend: recent behaviour relative to the customer's own baseline
recent = tx[tx["ts"] > asof - pd.Timedelta(days=90)]
feats["recent_vs_overall"] = (recent.groupby("customer_id")["amount"].mean()
                              / feats["mean_amt"]).fillna(0)

cols = ["n_tx", "mean_amt", "cv_amt", "recency_days", "tx_per_month",
        "top_cat_share", "recent_vs_overall"]
print(feats[cols].describe().round(3).T.to_string())

# %% [markdown]
# Seven features from four raw columns, each motivated by a **hypothesis about
# behaviour** rather than an automatic rule. Note the two defensive details:
# `.clip(lower=1)` prevents division by zero for one-day-old customers, and
# `.fillna(0)` handles customers with no recent transactions — an omission that
# would otherwise propagate `NaN` into the model.
#
# ### Cyclical time features
#
# Hour 23 and hour 0 are adjacent, but the integers 23 and 0 are maximally far
# apart. Encode them on a circle.

# %%
tx["hour"] = tx["ts"].dt.hour
tx["hour_sin"] = np.sin(2 * np.pi * tx["hour"] / 24)
tx["hour_cos"] = np.cos(2 * np.pi * tx["hour"] / 24)


def circ_dist(h1, h2):
    p1 = np.array([np.sin(2*np.pi*h1/24), np.cos(2*np.pi*h1/24)])
    p2 = np.array([np.sin(2*np.pi*h2/24), np.cos(2*np.pi*h2/24)])
    return np.linalg.norm(p1 - p2)


print(f"raw integer distance   23:00 vs 00:00 = {abs(23-0):>6.3f}")
print(f"raw integer distance   11:00 vs 12:00 = {abs(11-12):>6.3f}   <- wrong!")
print(f"cyclical distance      23:00 vs 00:00 = {circ_dist(23,0):>6.3f}")
print(f"cyclical distance      11:00 vs 12:00 = {circ_dist(11,12):>6.3f}   <- equal")

# %% [markdown]
# ### TASK 5.1
#
# How many customers are in your feature table?

# %%
n_pelanggan = None

checks.task("5.1", n_pelanggan)

# %% [markdown]
# ### TASK 5.2
#
# Build the three credit ratios that the risk model actually depends on:
# `dti` (debt / income), `pti` (monthly_payment / monthly income) and `util`
# (balance / credit_limit). Guard every denominator.

# %%
credit_feat = credit.copy()

# TODO: add the three ratio columns, e.g.
#   credit_feat["dti"] = credit_feat["debt"] / credit_feat["income"].clip(lower=1)
#   credit_feat["pti"] = ...
#   credit_feat["util"] = ...

checks.task("5.2", credit_feat)

# %% [markdown]
# ## 4. Feature selection  (DEMO)
#
# Three families, in increasing order of cost and of fidelity to the final model.

# %%
from sklearn.feature_selection import (mutual_info_classif, f_classif,
                                       SelectKBest, RFECV)
from sklearn.impute import SimpleImputer

num_cols = credit.select_dtypes("number").columns.tolist()
X_num = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(credit[num_cols]),
    columns=num_cols)

# Add the ratios, so selection can see them
X_num["dti"] = X_num["debt"] / X_num["income"].clip(lower=1)
X_num["pti"] = X_num["monthly_payment"] / (X_num["income"] / 12).clip(lower=1)
X_num["util"] = X_num["balance"] / X_num["credit_limit"].clip(lower=1)

# --- filter: fast, model-agnostic, blind to interactions -------------
f_scores = pd.Series(f_classif(X_num, y)[0], index=X_num.columns)
mi_scores = pd.Series(mutual_info_classif(X_num, y, random_state=0),
                      index=X_num.columns)

comparison = pd.DataFrame({"anova_F": f_scores, "mutual_info": mi_scores})
comparison["rank_F"] = comparison["anova_F"].rank(ascending=False).astype(int)
comparison["rank_MI"] = comparison["mutual_info"].rank(ascending=False).astype(int)
print(comparison.sort_values("mutual_info", ascending=False).round(4).to_string())

# %% [markdown]
# Mutual information is the more useful filter because it detects **any**
# dependence, not only monotone ones. A feature with correlation 0.0 and high
# mutual information — a U-shaped relationship — is invisible to a correlation
# filter and obvious to this one.

# %%
# --- embedded: selection happens as part of fitting ------------------
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

lasso_sel = SelectFromModel(
    LassoCV(cv=5, random_state=0, max_iter=20000)
).fit(StandardScaler().fit_transform(X_num), y)
kept = X_num.columns[lasso_sel.get_support()]
print(f"L1 (lasso) kept {len(kept)} of {X_num.shape[1]} features:")
print("  ", list(kept))

# %%
# --- permutation importance: measures what you actually care about ---
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

Xtr, Xte, ytr, yte = train_test_split(X_num, y, test_size=0.25,
                                      stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=0,
                            ).fit(Xtr, ytr)

perm = permutation_importance(rf, Xte, yte, n_repeats=10,
                              random_state=0, scoring="roc_auc")
imp = pd.DataFrame({
    "impurity": pd.Series(rf.feature_importances_, index=X_num.columns),
    "permutation": pd.Series(perm.importances_mean, index=X_num.columns),
}).sort_values("permutation", ascending=False)
print(imp.round(4).to_string())

print("\nImpurity importance is biased towards high-cardinality and continuous")
print("features. Permutation importance measures the drop in the score you")
print("actually report -- prefer it.")

# %% [markdown]
# ## 5. Principal component analysis  (DEMO)
#
# PCA finds an orthogonal basis in which the first axis captures the greatest
# possible variance. The components are the **eigenvectors of the covariance
# matrix**, ordered by eigenvalue, and component $j$ explains
# $\lambda_j / \sum_l \lambda_l$ of the total variance.
#
# First, verify that claim numerically.

# %%
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

Xbc, ybc = load_breast_cancer(return_X_y=True, as_frame=True)
Xs = StandardScaler().fit_transform(Xbc)

# Route 1: eigendecomposition of the covariance matrix, by hand
cov = np.cov(Xs, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
eigvals = eigvals[::-1]                       # eigh returns ascending

# Route 2: scikit-learn's PCA (which uses the SVD)
pca = PCA().fit(Xs)

print("first 5 eigenvalues, by hand :", np.round(eigvals[:5], 4))
print("first 5 from sklearn         :", np.round(pca.explained_variance_[:5], 4))
print("identical:", np.allclose(eigvals[:5], pca.explained_variance_[:5]))

# %%
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)

print(f"original dimensions: {Xbc.shape[1]}")
for k in (1, 2, 3, 5, 10):
    print(f"  {k:>2} components: {cum[k-1]:.3%} of variance")
print(f"\ncomponents for 95% variance: {np.searchsorted(cum, 0.95) + 1}")
print(f"eigenvalues > 1 (Kaiser)   : {(pca.explained_variance_ > 1).sum()}")
print("\nThe two criteria disagree. Neither is wrong; they answer different questions.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].bar(range(1, 16), evr[:15], color="#2E6F9E")
axes[0].axhline(1 / len(evr), color="grey", ls=":", lw=1)
axes[0].set(title="Scree plot", xlabel="component", ylabel="variance explained")
axes[1].plot(range(1, len(cum) + 1), cum, marker="o", ms=3, color="#173F5F")
axes[1].axhline(0.95, color="#9C3535", ls="--", lw=1)
axes[1].set(title="Cumulative variance", xlabel="components",
            ylabel="cumulative proportion")
plt.show()

# %%
# Loadings are the only route to interpreting what a component MEANS.
loadings = pd.Series(pca.components_[0], index=Xbc.columns)
print("PC1, largest loadings:")
print(loadings.abs().nlargest(6).round(3).to_string())
print("\nPC1 is dominated by concavity and concave-point measures -- it is")
print("essentially a 'boundary irregularity' factor.")

# %% [markdown]
# **Four limitations, frequently ignored:**
#
# - PCA is **unsupervised**. It maximises variance in $X$ with no reference to
#   $y$; a low-variance component may carry all your signal.
# - PCA **requires scaling**. Without it, whichever variable has the largest
#   units determines the components.
# - PCA is **linear**. Curved manifolds need kernel PCA, t-SNE or UMAP — and the
#   latter two are visualisation tools, not stable transforms for new data.
# - PCA **destroys interpretability**. Component 1 is a weighted combination of
#   all 30 original variables.

# %%
# Demonstrating limitation 2: PCA without standardising.
pca_unscaled = PCA().fit(Xbc.to_numpy())
cum_unscaled = np.cumsum(pca_unscaled.explained_variance_ratio_)
print(f"components for 95%, standardised   : {np.searchsorted(cum, 0.95) + 1}")
print(f"components for 95%, NOT standardised: {np.searchsorted(cum_unscaled, 0.95) + 1}")
top_unscaled = pd.Series(pca_unscaled.components_[0],
                         index=Xbc.columns).abs().nlargest(3)
print("\nUnscaled PC1 is essentially one variable:")
print(top_unscaled.round(4).to_string())
print("\nThat is an apparently excellent reduction which has discarded")
print("everything not correlated with tumour area.")

# %% [markdown]
# ## 6. Leakage — the most important section in this course  (DEMO)
#
# **Data leakage** is the use, during training, of information unavailable at
# prediction time. It produces validation scores that are optimistic and cannot
# be reproduced in production.
#
# Here is its cost, measured on data where the target is **pure noise** and no
# feature has any predictive value whatsoever.

# %%
from sklearn.pipeline import Pipeline

rng2 = np.random.default_rng(0)
n_rows, p_cols = 200, 5000
X_noise = rng2.normal(size=(n_rows, p_cols))       # pure noise
y_noise = rng2.integers(0, 2, n_rows)             # unrelated: true AUC is 0.5
cv_leak = StratifiedKFold(5, shuffle=True, random_state=1)

# WRONG: select features on ALL the data, then cross-validate the classifier
X_sel = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_noise,
                        cv=cv_leak, scoring="roc_auc").mean()

# RIGHT: selection inside the pipeline, refitted on each training fold
pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)),
                 ("clf", LogisticRegression(max_iter=1000))])
honest = cross_val_score(pipe, X_noise, y_noise, cv=cv_leak,
                         scoring="roc_auc").mean()

print(f"leaky selection : ROC-AUC = {leaky:.3f}")
print(f"inside pipeline : ROC-AUC = {honest:.3f}")
print(f"truth           : ROC-AUC = 0.500")

# %% [markdown]
# There is **no signal in this data at all**. The leaky procedure reports 0.91,
# which would be an excellent model if it were real.
#
# The mechanism is simple: with 5,000 noise features and 200 rows, some features
# correlate with the target by chance; the selector finds exactly those; and
# cross-validation then measures a model built using the very rows it is being
# validated against.
#
# This is not a contrived edge case. **Any** workflow with more features than
# rows — genomics, text, sensor arrays, or a spreadsheet after aggressive
# feature generation — is exposed to it.

# %%
# How the illusion scales with the number of noise features.
results = []
for p in (50, 200, 1000, 5000):
    Xn = rng2.normal(size=(n_rows, p))
    yn = rng2.integers(0, 2, n_rows)
    lk = cross_val_score(LogisticRegression(max_iter=1000),
                         SelectKBest(f_classif, k=min(20, p)).fit_transform(Xn, yn),
                         yn, cv=cv_leak, scoring="roc_auc").mean()
    hn = cross_val_score(
        Pipeline([("sel", SelectKBest(f_classif, k=min(20, p))),
                  ("clf", LogisticRegression(max_iter=1000))]),
        Xn, yn, cv=cv_leak, scoring="roc_auc").mean()
    results.append({"n_noise_features": p, "leaky_AUC": lk, "honest_AUC": hn})

leak_table = pd.DataFrame(results)
print(leak_table.round(3).to_string(index=False))
print("\nThe more features you have, the bigger the lie.")

# %% [markdown]
# ### TASK 5.3
#
# Report the ROC-AUC from the **leaky** procedure at p = 5000.

# %%
auc_bocor = None

checks.task("5.3", auc_bocor)

# %% [markdown]
# ### TASK 5.4
#
# Report the ROC-AUC from the **honest** pipeline at p = 5000.

# %%
auc_jujur = None

checks.task("5.4", auc_jujur)

# %% [markdown]
# ## 7. The leak-proof pipeline  (DEMO)
#
# `Pipeline` and `ColumnTransformer` make the correct version the *easy* one:
# every fitted quantity — the imputation median, the scaler's mean and standard
# deviation, the encoder's categories, the selected features — is learned on the
# training fold alone, automatically, including inside cross-validation.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class CreditFeatures(BaseEstimator, TransformerMixin):
    """Domain ratios, computed INSIDE the pipeline so they cannot drift
    between training and serving."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["debt_to_income"] = X["debt"] / X["income"].clip(lower=1)
        X["loan_to_income"] = X["loan_amount"] / X["income"].clip(lower=1)
        X["payment_to_income"] = X["monthly_payment"] / (X["income"] / 12).clip(lower=1)
        X["credit_utilisation"] = X["balance"] / X["credit_limit"].clip(lower=1)
        X["income_per_dependent"] = X["income"] / (1 + X["dependents"])
        X["age_at_maturity"] = X["age"] + X["term_months"] / 12
        return X


numeric = ["age", "income", "debt", "loan_amount", "term_months", "balance",
           "credit_limit", "dependents", "monthly_payment",
           "debt_to_income", "loan_to_income", "payment_to_income",
           "credit_utilisation", "income_per_dependent", "age_at_maturity"]
nominal = ["purpose", "housing", "employment_type"]
ordinal = ["education"]
edu_order = [["SMA", "D3", "S1", "S2"]]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), numeric),
    ("nom", Pipeline([("imp", SimpleImputer(strategy="constant",
                                            fill_value="Unknown")),
                      ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                           min_frequency=0.01,
                                           sparse_output=False))]), nominal),
    ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oe", OrdinalEncoder(categories=edu_order,
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1)),
                      ("sc", StandardScaler())]), ordinal),
], remainder="drop", verbose_feature_names_out=False)

full = Pipeline([
    ("features", CreditFeatures()),
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
])

# %%
# Split FIRST. Everything else happens inside the pipeline.
X_all = credit.drop(columns=["default"])
Xtr2, Xte2, ytr2, yte2 = train_test_split(X_all, y, test_size=0.2,
                                          stratify=y, random_state=42)

scores = cross_val_score(full, Xtr2, ytr2, cv=cv, scoring="roc_auc")
print(f"CV ROC-AUC (train only): {scores.mean():.4f} +/- {scores.std():.4f}")

full.fit(Xtr2, ytr2)
from sklearn.metrics import roc_auc_score
held_out = roc_auc_score(yte2, full.predict_proba(Xte2)[:, 1])
print(f"held-out ROC-AUC       : {held_out:.4f}")
print("\nThe two agree, which is what an honest pipeline looks like.")

# %%
# Did the engineered ratios earn their place?
names = full.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(full.named_steps["clf"].coef_[0], index=names)
print("Largest |coefficient| (standardised, so comparable):")
print(coefs.abs().nlargest(10).round(3).to_string())

# %% [markdown]
# ## 8. Take-home  (assessed)
#
# **A. Encoding trade-offs.** A categorical feature has 3,000 levels, of which
# the top 20 cover 85% of rows. Compare one-hot, target and hashing encodings:
# state for each the resulting dimensionality, the principal risk, and the
# circumstance in which you would choose it.
#
# **B. Leakage at scale.** Reproduce section 6, then vary $p$ over
# $\{50, 500, 5000\}$ and **plot** leaky and honest ROC-AUC against $p$. Explain
# the shape of each curve.
#
# **C. PCA without scaling.** Reproduce the breast-cancer PCA *without*
# standardising. Report how many components now reach 95%, examine PC1's
# loadings, and explain mechanically what happened.
#
# **D. Three more features.** Add three features of your own design to the RFM
# table of section 3: one capturing weekday-vs-weekend behaviour, one capturing
# spending volatility over time, and one capturing category switching. State the
# behavioural hypothesis behind each, then test whether they improve a model.
#
# **E. Audit a leak.** A colleague's churn model scores ROC-AUC 0.98 in
# cross-validation and 0.61 in production. Their features include
# `days_since_last_login`, `support_tickets_open`, `account_status` and
# `final_invoice_amount`. Identify the likely culprit, classify the **type** of
# leakage, and describe the audit you would run over the remaining features.
#
# ---
# **Next week:** our first models.
