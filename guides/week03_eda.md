# Week 3 — `week03_eda.ows`

Companion notes for the Orange workflow. Book: chapter 3. Notebook twin:
`notebooks/week03_eda.ipynb`.

---

## 1. What this workflow is

A **fan-out of five views** on one table, plus one widget-to-widget link that is
not a `Data` link.

```
                        ┌──▶ Distributions
                        ├──▶ Feature Statistics
File: rumah_yogya.tab ──┼──▶ Correlations ──(Features)──┐
                        ├──▶ Box Plot                   │
                        ├──▶ Heat Map                   │
                        └──▶ Scatter Plot ◀─────────────┘
```

**Dataset** — `data/rumah_yogya.tab`, 2,200 house listings around Yogyakarta.
Six continuous predictors (`luas_bangunan`, `luas_tanah`, `kamar_tidur`,
`kamar_mandi`, `umur_bangunan`, `jarak_pusat_km`), two categorical
(`kecamatan`, `sertifikat`), and the target `harga` in millions of rupiah.
`harga` is declared as the **class variable** in the `.tab` file — remember that,
it changes what Correlations shows you.

**The idea being taught: a second channel type.** The Scatter Plot has *two*
inputs wired: `Data` from File and **`Features` from Correlations**. Clicking a
row in Correlations sends that pair of variables down the `Features` channel, and
the Scatter Plot re-draws itself on those two axes. This is the first workflow
where a widget steers another widget rather than just feeding it rows.
Double-click the Correlations → Scatter Plot link to see it.

---

## 2. What to do

1. **Distributions** — set the variable to `harga`. Is it skewed? Then set
   *Split by* to `kecamatan` and look again.
2. **Correlations** — which pair scores highest? Click it and watch the Scatter
   Plot follow.
3. **Box Plot** — `harga` grouped by `sertifikat`.
4. **Simpson hunt** — in Box Plot, put `harga` by `kecamatan`, then add a
   Subgroup. Does any comparison reverse once you condition on it?
5. Compare everything with `notebooks/week03_eda.ipynb`.

---

*Versi mahasiswa. Pembahasan, hal yang perlu diperhatikan, dan jawaban setiap langkah disampaikan di kelas setelah praktikum.*
