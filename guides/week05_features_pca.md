# Week 5 — `week05_features_pca.ows`

Companion notes for the Orange workflow. Book: chapter 5. Notebook twin:
`notebooks/week05_features.ipynb`.

---

## 1. What this workflow is

Two **controlled experiments on the canvas**, one above the other. In each, the
branches receive the same data and differ in exactly one thing, so every
difference downstream is caused by that one thing.

**Part A — try things with features (top of the canvas).**

```
                 ┌──▶ Rank: raw columns
                 │
                 ├──▶ Feature Constructor ──┬──▶ Rank: with ratios
                 │    (dti, pti, util)      ├──▶ Box Plot: dti by default
                 │                          ├──▶ Select Columns: ratios only ──▶ Test and Score: ratios only
File: credit.tab ┤                          └────────────────────────────────▶ Test and Score: + ratios
                 ├──────────────────────────────────────────────────────────▶ Test and Score: raw columns
                 │
                 └──▶ Continuize ──▶ Encoded data

Preprocess: impute + scale + one-hot ──(Preprocessor)──┬──▶ Logistic Regression (scaled)
                                                       └──▶ kNN (k=15, scaled)
kNN (k=15)  ·  Random Forest       ── all four learners feed all three Test and Score widgets
```

Same four learners, same 5 folds, three different sets of columns. The *only*
thing that changes between the three Test and Score widgets is the features.

**Part B — PCA needs scaling (bottom of the canvas).**

```
File: credit.tab ──┬──▶ Preprocess: normalise ──▶ PCA (scaled) ────▶ PC1 vs PC2 (scaled)
                   └────────────────────────────▶ PCA (UNSCALED) ──▶ PC1 vs PC2 (unscaled)
```

**Dataset** — `data/credit.tab`, the same 4,000 loan applications, with the money
columns in raw rupiah. That matters in both parts: `loan_amount` has a standard
deviation around 29 million, `age` around 13. Any method built on distances or
variances cannot ignore a factor of two million.

**The ideas being taught.**

- *Part A:* a well-chosen feature can be worth more than a better model — but
  whether a model can *use* it depends on the model. Distance-based models also
  need the feature on a comparable scale, and are hurt by irrelevant columns.
- *Part B:* PCA maximises *variance*, and variance is not scale-free. Feed it raw
  rupiah and it reports the column with the biggest units.

**Watch the channel names.** Each is a hint about what the widget produces:

- Preprocess emits **`Preprocessor`** (a recipe, sent *into* a learner) and
  **`Preprocessed Data`** (a transformed table). Part A uses the first, Part B the
  second — the same widget, used two different ways.
- Rank emits **`Reduced Data`**, not `Data`.
- PCA emits **`Transformed Data`** (the projected coordinates) as well as
  `Components` and `Data`.

**Pre-configured, and why.**

- *Feature Constructor* already holds the three ratios, with guarded
  denominators: `debt / max(income, 1)`, `monthly_payment / max(income / 12, 1)`,
  `balance / max(credit_limit, 1)`.
- *Select Columns* keeps only `dti`, `pti`, `util` as features.
- *Box Plot* shows `dti` split by `default`.
- *kNN* uses k = 15 (as in the notebook); *Random Forest* uses 100 trees and a
  fixed seed, so everyone sees the same numbers.
- *Preprocess: impute + scale + one-hot* feeds two learners. **A preprocessor sent
  to a learner replaces that learner's own defaults** (imputation and one-hot
  encoding) rather than adding to them. Send only "normalise" and logistic
  regression fails on the first categorical column or missing income. So the
  chain restates the defaults around the scaling. Passing the preprocessor
  *into* the learner is also the leak-free wiring: it is refitted on each
  training fold, exactly like the notebook's `Pipeline`.
- *Part B:* the Preprocess widget has *Normalize Features* enabled (mean 0,
  sd 1), and **both PCA widgets have their own *Normalize variables* checkbox
  switched off**. Orange's PCA widget normalises by default; left on, it would
  quietly scale the "UNSCALED" branch too and both branches would report
  identical variance. With it off, Preprocess is the only difference. Toggle
  Preprocess off and watch the contrast collapse — or tick *Normalize variables*
  in the UNSCALED PCA — to prove the effect to yourself.

---

## 2. What to do

**Part A — features**

1. **Compare the two Rank widgets.** Where do `dti`, `pti` and `util` land once
   they exist? Where were `debt` and `income` before?
2. **Box Plot.** Does `dti` look different for defaulters? Switch the variable to
   `debt`, then `income`.
3. **Open the three Test and Score widgets**, sort by AUC, and fill in a 4 × 3
   table. Which model gains most from the ratios? Which gains least?
4. **Unscaled kNN** scores exactly the same with and without the ratios, while
   scaled kNN gains the most. Why can unscaled kNN not see them?
5. **Ratios only** — 3 features instead of 16. Unscaled kNN suddenly jumps. What
   did removing the rupiah columns do to its distances?
6. **Try your own.** Add a fourth expression in Feature Constructor, e.g.
   `loan_amount / max(income, 1)`, and watch all three Test and Score widgets
   update.
7. **Continuize.** Switch the treatment of categorical columns (one-hot,
   ordinal, remove) and look at the *Encoded data* table.

**Part B — PCA**

1. **Preprocess** — confirm *Normalize Features* is on.
2. **Open both PCA widgets.** How many components are needed to reach 95% of the
   variance in each?
3. **Compare the two scatter plots.** One of them is essentially a plot of a
   single variable. Which, and why? See book §5.6.4, limitation 2.

---

*Versi mahasiswa. Pembahasan, hal yang perlu diperhatikan, dan jawaban setiap langkah disampaikan di kelas setelah praktikum.*
