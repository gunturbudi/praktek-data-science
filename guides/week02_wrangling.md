# Week 2 — `week02_wrangling.ows`

Companion notes for the Orange workflow. Book: chapter 2. Notebook twin:
`notebooks/week02_wrangling.ipynb`.

---

## 1. What this workflow is

A **cleaning pipeline**, left to right, with a Data Table tapped off at two
points so you can see what each stage changed.

```
File: credit.csv ──┬──▶ 1. Profile   (Feature Statistics)
                   └──▶ 2. Impute ──┬──▶ After impute   (Data Table)
                                    └──▶ 3. Outliers ──┬──▶ After outliers (Data Table)
                                                       └──▶ 4. Select Rows ──▶ 5. Save Data
```

**Dataset** — `data/credit.csv` again (4,000 rows), because you already know from
week 1 that `income` and `employment_type` have holes in them.

**The idea being taught: order matters, and every step is a decision.**
Profiling comes first because you cannot choose an imputation strategy for a
column you have not looked at. Imputation comes before outlier detection because
most outlier detectors cannot handle `NaN` at all. Row filtering comes last
because filtering first would change what "an outlier" means.

Note the channel names on the last three links — they are not all `Data`:

- Outliers emits **`Inliers`** (the rows it kept) and **`Outliers`** (the rows it
  flagged). The workflow uses `Inliers`.
- Select Rows emits **`Matching Data`** and **`Unmatched Data`**. The workflow
  uses `Matching Data`.

Reading channel names is how you tell what a link actually carries.

---

## 2. What to do

1. **Profile** (Feature Statistics). Confirm what you found in week 1: `income`
   342 missing, `employment_type` 161.
2. **Impute.** Run it twice. First with *Average / Most frequent*, then with
   *Model-based imputer (simple tree)*. Open "After impute" each time and compare
   what landed in the `income` column.
3. **Outliers.** Switch between *One Class SVM* and *Local Outlier Factor*.
   Read the row count in "After outliers" each time. How many rows are dropped?
4. **Select Rows.** Add the condition `age ≥ 25`.
5. **Save Data.** Write the result to `credit_clean.tab`.
6. **Answer the canvas question:** is dropping the outliers here defensible?

---

*Versi mahasiswa. Pembahasan, hal yang perlu diperhatikan, dan jawaban setiap langkah disampaikan di kelas setelah praktikum.*
