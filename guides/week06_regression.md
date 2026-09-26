# Week 6 — `week06_regression.ows`

Companion notes for the Orange workflow. Book: chapter 6. Notebook twin:
`notebooks/week06_regression.ipynb`.

---

## 1. What this workflow is

**Two independent pipelines on one canvas**, deliberately laid out as two rows so
you can see that predicting a number and predicting a class use the same skeleton
with different parts.

**Top row — continuous target (`harga`):**

```
Linear Regression ──(Learner)──▶ Test and Score ◀──(Data)──┐
                  ──(Model)────▶ Predictions   ◀──(Data)───┤
                                     │                     │
                                     │            File: rumah_yogya.tab
                                (Predictions)
                                     ▼
                          Predicted vs actual (Scatter Plot)
```

**Bottom row — binary target (`default`):**

```
Logistic Regression ──(Learner)──▶ Test and Score ◀──(Data)── File: credit.tab
                    ──(Model)────▶ Nomogram              │
                                                (Evaluation Results)
                                                         ├──▶ Confusion Matrix
                                                         └──▶ ROC Analysis
```

**Datasets** — `data/rumah_yogya.tab` (2,200 houses, target `harga`) and
`data/credit.tab` (4,000 applications, target `default`).

**The idea being taught: three distinct channel types.** This is the workflow
where Orange's type system becomes visible.

| Channel | Carries | Goes from → to |
| --- | --- | --- |
| `Data` | a table of rows | File → everything |
| **`Learner`** | an *untrained* algorithm | Linear/Logistic Regression → Test and Score |
| **`Model`** | a *trained* predictor | Linear/Logistic Regression → Predictions, Nomogram |
| **`Evaluation Results`** | per-fold predictions + truth | Test and Score → Confusion Matrix, ROC |

A model widget emits both `Learner` and `Model`, and which one you connect
decides what happens. Send the `Learner` to Test and Score and it is re-fitted
inside every fold — honest. Send the `Model` and it was fitted on everything,
including the rows it is about to be tested on — leakage. **The wire you choose is
the difference between a valid score and a fraudulent one.**

The Nomogram is the payoff widget: it turns a logistic regression into something
a loan officer can read without knowing what a log-odds is.

---

## 2. What to do

1. **Test and Score (top)** — set *Cross validation, 5 folds*. Note RMSE, MAE, R².
2. **Predicted vs actual** — is the spread even across the range, or does it fan
   out? That fan is heteroscedasticity (book §6.4.1).
3. **Logistic Regression** — try Ridge (L2) with C = 1, then Lasso (L1). Watch
   coefficients vanish.
4. **Nomogram** — each feature is a line; the length of its bar is its influence.
   Drag the blue markers to score an applicant.
5. **Confusion Matrix** — which error does the model make more often? Is that the
   cheap one?

---

*Versi mahasiswa. Pembahasan, hal yang perlu diperhatikan, dan jawaban setiap langkah disampaikan di kelas setelah praktikum.*
