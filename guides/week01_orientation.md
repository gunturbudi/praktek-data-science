# Week 1 — `week01_orientation.ows`

Companion notes for the Orange workflow. Read this beside the canvas, not
instead of it. Book: chapter 1. Notebook twin: `notebooks/week01_foundations.ipynb`.

---

## 1. What this workflow is

The smallest workflow that still teaches something: **one source, three views**.

```
File: credit.csv ──┬──▶ Data Table
                   ├──▶ Feature Statistics
                   └──▶ Distributions
```

**Dataset** — `data/credit.csv`, 4,000 loan applications, 13 predictor columns
plus the target `default` (yes/no). Money columns are in rupiah.

**The idea being taught: widgets and channels.** A widget is a small program with
inputs and outputs. A link between two widgets is a *channel* carrying a typed
object — here every link carries `Data` (an `Orange.data.Table`). Nothing is
"run"; changing anything upstream pushes new data downstream immediately. That
push model is the whole mental model of Orange, and everything in weeks 2–7 is
just a longer chain of it.

Three widgets hang off one File on purpose: one source can feed many consumers,
and each consumer answers a different question about the same table.

| Widget | The question it answers |
| --- | --- |
| Data Table | *What do the rows actually look like?* |
| Feature Statistics | *What is each column, and how broken is it?* |
| Distributions | *What is the shape of one variable?* |

---

## 2. What to do

1. **Open the file.** `orange-canvas week01_orientation.ows`. The File widget is
   pre-pointed at `data/credit.csv` — do not re-select it unless it shows an error.
2. **Double-click a link.** The channel dialog opens and shows `Data → Data`.
   Close it. You have now seen the thing that makes Orange a dataflow tool.
3. **Data Table.** How many rows and columns? Click a column header to sort.
4. **Feature Statistics.** Which column has the most missing values? Which
   columns, if any, are identifiers rather than features?
5. **Distributions.** Set the variable to `default`. What is the base rate of
   default? Write it down — week 7 will not make sense without it.

---

*Versi mahasiswa. Pembahasan, hal yang perlu diperhatikan, dan jawaban setiap langkah disampaikan di kelas setelah praktikum.*
