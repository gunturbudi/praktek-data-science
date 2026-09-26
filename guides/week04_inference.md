# Week 4 — `week04_inference.ows`

Companion notes for the Orange workflow. Book: chapter 4. Notebook twin:
`notebooks/week04_inference.ipynb`.

---

## 1. What this workflow is

Five visual widgets that show **shape** and **association**, one Select Rows
branch that isolates a single segment, and one Python Script that performs
every **test** Orange has no widget for.

```
                      ┌──▶ Feature Statistics                    § 2
                      ├──▶ Distributions                         § 2, § 0
                      ├──▶ Box Plot  (prints a t-test)           § 5
File: ab_test.tab ────┼──▶ Sieve Diagram                         § 1, § 5
                      ├──▶ Mosaic Display                        § 7
                      ├──▶ Select Rows ──▶ Sieve (one segment)   § 6
                      └──▶ Python Script ──▶ Data Table          § 6, § 7
```

The § references point at the matching section of the notebook, which follows
chapter 4's own nine-section order.

**Dataset** — `data/ab_test.tab`, 70,000 sessions from an A/B test.
`variant` (A/B), `device` (mobile/desktop), `day` (1–14), `session_seconds`,
`pages_viewed`, and the target `converted` (0/1).

**The idea being taught: know the limits of your tool.** Orange is a visual
exploration environment, not a statistics package. It can show you that two bars
differ; it will not hand you a z-statistic, a p-value or a confidence interval for
a two-proportion test. So the workflow does what a real analyst does — it uses the
visual widgets to *find* the effect and drops into Python to *quantify* it.

**Pre-configured:** the Python Script widget already contains the analysis, so
the workflow runs as shipped. The same script lives at
`orange/scripts/ab_test.py` and runs standalone with `python orange/scripts/ab_test.py`,
which is how it is tested.

---

## 2. What to do

**Shape** — what kind of variable is each column?

1. **Feature Statistics** — read the thumbnail histogram beside each column.
   Which are symmetric and which have a long right tail?
2. **Distributions** — `pages_viewed`. A count of events in one visit: that is a
   Poisson shape. Then `session_seconds`: the long right tail is log-normal.
3. **Distributions** — `converted`, split by `variant`. Eyeball the difference.
   Is it convincing?

**Association** — do two variables move together?

4. **Box Plot** — `session_seconds` by `variant`. Orange prints a t-test at the
   bottom of this widget; it is the only test it performs for you.
5. **Sieve Diagram** — `variant` × `converted`. Colour shows the deviation from
   independence; grid density shows the cell frequency.
6. **Mosaic Display** — `variant`, `converted` **and** `device`. Two questions:
   does the lift hold in both devices, and is the device *mix* the same in both
   arms? The second decides whether `device` could confound anything at all.

**One segment at a time**

7. **Select Rows** — set the condition `device = mobile` and read the Sieve
   downstream. Then switch it to `desktop` and read it again. That is two
   hypothesis tests, chosen after seeing the data.

**The numbers Orange will not give you**

8. **Python Script** — run it. It prints, in the order a real analysis runs: the
   sample-ratio check, the lift with a 95% interval, the sample size that was
   actually required, and the per-device segments with a Holm correction.

---

*Versi mahasiswa. Pembahasan, hal yang perlu diperhatikan, dan jawaban setiap langkah disampaikan di kelas setelah praktikum.*
