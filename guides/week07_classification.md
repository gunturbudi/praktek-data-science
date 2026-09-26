# Week 7 — `week07_classification.ows`

Companion notes for the Orange workflow. Book: chapter 7. Notebook twin:
`notebooks/week07_classification.ipynb`. **This is the last practicum before the
UTS.**

---

## 1. What this workflow is

The **fair comparison** pattern: six learners, one Test and Score, four evaluation
views, and a drill-down into the mistakes.

```
                    ┌─▶ Constant (baseline) ─┐
                    ├─▶ Logistic Regression ─┤
                    ├─▶ Tree ────────────────┤ (Learner)
File: credit.tab ───┼─▶ Naive Bayes ─────────┼───────────▶ Test and Score
                    ├─▶ kNN ─────────────────┤                    │
                    ├─▶ SVM ─────────────────┘                    │
                    └─▶ (Data) ────────────────────────────────────┤
                                                                   │ (Evaluation Results)
     Tree ──(Model)──▶ Tree Viewer                                 │
                                                                   ├─▶ Confusion Matrix
                                                                   ├─▶ ROC Analysis
                                                                   ├─▶ Lift Curve
                                                                   └─▶ Calibration Plot

     Confusion Matrix ──(Selected Data)──▶ Misclassified rows (Data Table)
```

**Dataset** — `data/credit.tab`. 4,000 applications, 20% default rate.

**The idea being taught: what makes a comparison fair.** Every learner enters the
*same* Test and Score widget, so every one is trained and tested on identical
folds with identical preprocessing. Six separate Test and Score widgets would give
six numbers that cannot be compared, because each would have drawn its own random
folds. One widget, six inputs, one table — that layout *is* the methodology.

**Constant is not filler.** It predicts the majority class always. It is the line
below which a model has demonstrably learned nothing, and it belongs on every
classification canvas you ever build.

**The `Selected Data` channel.** Click a cell in the Confusion Matrix and it emits
the rows in that cell down `Selected Data` to the Data Table. That is how you get
from "the model is X% accurate" to "here are the people it was wrong about" —
the single most useful move in the whole workflow.

---

## 2. What to do

1. **Test and Score** — *Cross validation, 5 folds*. Sort by AUC. Does every model
   beat Constant?
2. Set **Target class = `yes`** (the minority class). Watch precision and recall
   change completely.
3. **Confusion Matrix** — select the false-negative cell and inspect those rows in
   the Data Table. What do the missed defaulters have in common?
4. **Tree Viewer** — set depth to 3. Can you read the rules aloud?
5. **Calibration Plot** — compare Naive Bayes with Logistic Regression.

**UTS prep: be able to explain every number here.**

---

*Versi mahasiswa. Pembahasan, hal yang perlu diperhatikan, dan jawaban setiap langkah disampaikan di kelas setelah praktikum.*
