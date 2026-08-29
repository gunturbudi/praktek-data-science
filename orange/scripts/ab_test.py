# Week 4 — the formal test Orange does not do for you.
#
# This script is embedded into the Python Script widget of
# week04_inference.ows. It also runs standalone:
#
#     python scripts/ab_test.py
#
# Inside Orange, `in_data` is supplied by the widget. Standalone, we load the
# table ourselves.
import numpy as np
from scipy import stats
from Orange.data import Table

try:
    in_data          # provided by the Python Script widget
except NameError:    # running standalone
    from pathlib import Path
    in_data = Table(str(Path(__file__).parent.parent.parent / "data" / "ab_test.tab"))

VARIANT = "variant"
TARGET = "converted"


def column(table, name):
    """Return a column by name as a list of strings, wherever it lives."""
    domain = table.domain
    for source, var_list in ((table.X, domain.attributes),
                             (table.Y.reshape(-1, 1), [domain.class_var]),
                             (table.metas, domain.metas)):
        for i, var in enumerate(var_list):
            if var is not None and var.name == name:
                col = source[:, i]
                if var.is_discrete:
                    return [var.values[int(v)] for v in col]
                return col
    raise KeyError(name)


variant = np.array(column(in_data, VARIANT))
converted = np.array(column(in_data, TARGET))

arms = sorted(set(variant))
positive = sorted(set(converted))[-1]        # "1" or "yes"

n = np.array([(variant == a).sum() for a in arms], dtype=float)
conv = np.array([((variant == a) & (converted == positive)).sum()
                 for a in arms], dtype=float)
rate = conv / n

# ---- sample ratio check: do this BEFORE trusting anything else ----------
_, p_srm = stats.chisquare(n)

# ---- two-proportion z test ---------------------------------------------
pool = conv.sum() / n.sum()
se_pool = np.sqrt(pool * (1 - pool) * (1 / n[0] + 1 / n[1]))
z = (rate[1] - rate[0]) / se_pool
p = 2 * (1 - stats.norm.cdf(abs(z)))

se_unpooled = np.sqrt(rate[0] * (1 - rate[0]) / n[0]
                      + rate[1] * (1 - rate[1]) / n[1])
diff = rate[1] - rate[0]
lo, hi = diff - 1.96 * se_unpooled, diff + 1.96 * se_unpooled

print("A/B TEST RESULT")
print("=" * 48)
print("  sample ratio check: p = {:.3f}  {}".format(
    p_srm, "OK" if p_srm > 0.01 else "BROKEN - stop here"))
print("")
for arm, r, ni in zip(arms, rate, n):
    print("  arm {}: {:.4%}   (n = {:,})".format(arm, r, int(ni)))
print("")
print("  absolute lift {:+.4%}".format(diff))
print("  95% CI        [{:+.4%}, {:+.4%}]".format(lo, hi))
print("  relative lift {:+.2%}".format(diff / rate[0]))
print("  z = {:.3f},  p = {:.6f}".format(z, p))
print("")
print("  " + ("REJECT the null: the two arms differ."
              if p < 0.05 else "Cannot reject the null."))
print("")
print("  Now check the same split by device (Mosaic Display).")
print("  Does the lift hold in BOTH devices, or is it Simpson's paradox?")

out_data = in_data
