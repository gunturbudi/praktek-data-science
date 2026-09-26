# Week 4 — the formal tests Orange does not do for you.
#
# This script is embedded into the Python Script widget of
# week04_inference.ows. It also runs standalone:
#
#     python scripts/ab_test.py
#
# It mirrors sections 6 and 7 of notebooks/week04_inference.ipynb: the
# sample-ratio check, the primary two-proportion test with an interval, the
# sample size that was actually required, and the per-device segment tests
# with a multiplicity correction.
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
SEGMENT = "device"


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


def two_proportion(conv, n):
    """(diff, lo, hi, z, p) for a two-proportion z-test. conv/n are length 2."""
    rate = conv / n
    pool = conv.sum() / n.sum()
    se_pool = np.sqrt(pool * (1 - pool) * (1 / n[0] + 1 / n[1]))
    z = (rate[1] - rate[0]) / se_pool
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = np.sqrt(rate[0] * (1 - rate[0]) / n[0]
                          + rate[1] * (1 - rate[1]) / n[1])
    diff = rate[1] - rate[0]
    return diff, diff - 1.96 * se_unpooled, diff + 1.96 * se_unpooled, z, p


def holm(pvals):
    """Holm step-down adjusted p-values. Dominates Bonferroni; prefer it."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def n_per_arm(p1, rel_mde, alpha=0.05, power=0.80):
    """Sample size PER ARM to detect a relative lift, via Cohen's h.

    Two arms of n each give var(h_hat) = 2/n, hence the leading 2. Omitting it
    is the classic slip: it under-orders the sample by half, and the resulting
    experiment has about 50% power while its designer believes it has 80%.
    """
    p2 = p1 * (1 + rel_mde)
    h = abs(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2 * ((z_a + z_b) / h) ** 2))


variant = np.array(column(in_data, VARIANT))
converted = np.array(column(in_data, TARGET))
device = np.array(column(in_data, SEGMENT))

arms = sorted(set(variant))
positive = sorted(set(converted))[-1]        # "1" or "yes"

n = np.array([(variant == a).sum() for a in arms], dtype=float)
conv = np.array([((variant == a) & (converted == positive)).sum()
                 for a in arms], dtype=float)
rate = conv / n

# ---- step 0: sample ratio check -- do this BEFORE trusting anything else --
_, p_srm = stats.chisquare(n)

print("A/B TEST RESULT")
print("=" * 58)
print("  sample ratio check: p = {:.3f}  {}".format(
    p_srm, "OK" if p_srm > 0.01 else "BROKEN - stop here"))
if p_srm <= 0.01:
    print("")
    print("  Randomisation or logging is broken. Nothing below is")
    print("  interpretable. Fix the assignment mechanism and rerun.")

# ---- step 1: the primary metric -----------------------------------------
diff, lo, hi, z, p = two_proportion(conv, n)

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
print("  Report the interval, not the verdict. '+0.72 pp, 95% CI")
print("  [+0.41, +1.03]' is actionable; 'p < 0.001' is not.")

# ---- step 2: was the test big enough? -----------------------------------
print("")
print("SAMPLE SIZE REQUIRED (80% power, alpha = 0.05)")
print("-" * 58)
for mde in (0.05, 0.10, 0.12, 0.20):
    need = n_per_arm(rate[0], mde)
    print("  {:>4.0%} relative lift: {:>9,} per arm   (have {:,})  {}".format(
        mde, need, int(n[0]),
        "OK" if n[0] >= need else "UNDERPOWERED"))

# ---- step 3: segments, with the multiplicity correction -----------------
segments = sorted(set(device))
rows, pvals = [], []
for seg in segments:
    mask = device == seg
    n_s = np.array([(mask & (variant == a)).sum() for a in arms], dtype=float)
    c_s = np.array([(mask & (variant == a) & (converted == positive)).sum()
                    for a in arms], dtype=float)
    d_s, lo_s, hi_s, z_s, p_s = two_proportion(c_s, n_s)
    rows.append((seg, int(n_s.sum()), c_s[0] / n_s[0], c_s[1] / n_s[1],
                 d_s, lo_s, hi_s, z_s, p_s))
    pvals.append(p_s)

adj = holm(pvals)

print("")
print("BY SEGMENT ({}), Holm-corrected for {} tests".format(
    SEGMENT, len(segments)))
print("-" * 58)
print("  {:<9}{:>8}{:>9}{:>9}{:>10}{:>11}{:>9}".format(
    "segment", "n", "rate A", "rate B", "lift", "raw p", "Holm p"))
for (seg, n_s, ra, rb, d_s, lo_s, hi_s, z_s, p_s), a_p in zip(rows, adj):
    print("  {:<9}{:>8,}{:>9.2%}{:>9.2%}{:>+10.2%}{:>11.5f}{:>9.5f}{}".format(
        seg, n_s, ra, rb, d_s, p_s, a_p, "  *" if a_p < 0.05 else ""))

print("")
print("  95% CI per segment:")
for seg, n_s, ra, rb, d_s, lo_s, hi_s, z_s, p_s in rows:
    print("    {:<9} {:+.2%}  [{:+.2%}, {:+.2%}]".format(seg, d_s, lo_s, hi_s))

# ---- what to conclude, and what NOT to call it --------------------------
balance = np.array([[(mask_a & (device == s)).sum() for s in segments]
                    for mask_a in [variant == a for a in arms]], dtype=float)
_, p_balance, _, _ = stats.chi2_contingency(balance)

print("")
print("INTERPRETATION")
print("-" * 58)
print("  {} balance across arms: chi2 p = {:.3f}".format(SEGMENT, p_balance))
print("  The arms have the same device mix, so device is NOT a confounder,")
print("  and a variable that is not a confounder cannot reverse anything.")
print("")
print("  What you are looking at is EFFECT HETEROGENEITY, not Simpson's")
print("  paradox: the lift holds in sign everywhere but differs hugely in")
print("  size. Simpson's paradox requires the aggregate conclusion to FLIP")
print("  when you condition, and here it does not. Use the right name.")
print("")
print("  Note also that these segment tests were not free: they are extra")
print("  hypotheses, which is why the Holm column exists. Notebook section 6")
print("  runs ten of them and shows what happens when you do not correct.")

out_data = in_data
