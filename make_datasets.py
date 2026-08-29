"""
make_datasets.py — regenerate every dataset used by the practicum.

Run once:      python make_datasets.py
Output:        data/*.csv   (for the notebooks)
               data/*.tab   (Orange native, with types and class flags declared)

Everything is generated from fixed seeds, so the files are byte-identical on
every machine and the numbers in the notebooks are reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def solve_intercept(score, target_rate, lo=-12.0, hi=12.0, tol=1e-6):
    """Bisect for the intercept that makes mean sigmoid(intercept + score) equal
    target_rate. Lets us set a base rate directly instead of guessing."""
    for _ in range(200):
        mid = (lo + hi) / 2
        rate = float(np.mean(1 / (1 + np.exp(-(mid + score)))))
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- Orange .tab
def write_tab(df: pd.DataFrame, path: Path, types: dict, flags: dict | None = None):
    """Write an Orange native .tab file: names / types / flags / rows."""
    flags = flags or {}
    cols = list(df.columns)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(cols) + "\n")
        fh.write("\t".join(types.get(c, "") for c in cols) + "\n")
        fh.write("\t".join(flags.get(c, "") for c in cols) + "\n")
        for row in df.itertuples(index=False):
            out = []
            for v in row:
                if pd.isna(v):
                    out.append("?")
                elif isinstance(v, float):
                    out.append(f"{v:.4f}".rstrip("0").rstrip("."))
                else:
                    out.append(str(v))
            fh.write("\t".join(out) + "\n")
    print(f"  wrote {path.name:<28} {len(df):>6,} rows x {len(cols)} cols")


def write_csv(df: pd.DataFrame, path: Path, **kw):
    df.to_csv(path, index=False, **kw)
    print(f"  wrote {path.name:<28} {len(df):>6,} rows x {df.shape[1]} cols")


# ================================================================= WEEK 2
def make_sales():
    """A deliberately messy retail extract + a clean reference table."""
    rng = np.random.default_rng(7)
    n = 5000

    regions_messy = ["Yogyakarta", "yogyakarta ", " YOGYAKARTA", "Jawa Tengah",
                     "jawa tengah", "Jawa Timur", "Jawa Barat"]
    channels = ["toko", "online", "reseller"]

    df = pd.DataFrame({
        "order_id":    np.arange(1, n + 1),
        "store_id":    [f"{i:05d}" for i in rng.integers(1, 60, n)],
        "customer_id": rng.integers(1000, 1800, n),
        "region":      rng.choice(regions_messy, n),
        "channel":     rng.choice(channels, n, p=[0.5, 0.35, 0.15]),
        "order_date":  pd.to_datetime("2023-01-01")
                       + pd.to_timedelta(rng.integers(0, 730, n), unit="D"),
        "quantity":    rng.integers(1, 12, n),
        "unit_price":  np.round(rng.lognormal(11.5, 0.6, n), -2),
    })

    # ---- inject realistic defects -------------------------------------
    df.loc[rng.choice(n, 300, replace=False), "unit_price"] = np.nan   # missing
    df.loc[rng.choice(n, 80, replace=False), "quantity"] = -999        # sentinel
    df.loc[rng.choice(n, 40, replace=False), "region"] = ""            # blank
    big = rng.choice(n, 25, replace=False)
    df.loc[big, "unit_price"] = df.loc[big, "unit_price"] * 500        # outliers

    clean = df.copy()                       # keep a pristine copy before formatting

    # European number format, dd/mm/yyyy dates, and duplicated rows
    df["order_date"] = df["order_date"].dt.strftime("%d/%m/%Y")
    df["unit_price"] = df["unit_price"].map(
        lambda v: "" if pd.isna(v)
        else f"{v:,.2f}".replace(",", "@").replace(".", ",").replace("@", "."))
    df = pd.concat([df, df.sample(120, random_state=1)], ignore_index=True)
    df = df.sample(frac=1, random_state=2).reset_index(drop=True)
    write_csv(df, DATA / "sales_raw.csv", sep=";")

    # ---- the cleaned version, so week 3 does not depend on week 2 ------
    clean["region"] = (clean["region"].str.strip().str.title()
                                      .replace({"": np.nan}))
    clean.loc[clean["quantity"] <= 0, "quantity"] = np.nan
    clean["unit_price"] = clean.groupby("region")["unit_price"].transform(
        lambda s: s.fillna(s.median()))
    clean = clean.dropna(subset=["quantity", "region"]).copy()
    lo, hi = clean["unit_price"].quantile([0.001, 0.999])
    clean["unit_price"] = clean["unit_price"].clip(lo, hi)
    clean["quantity"] = clean["quantity"].astype(int)
    clean["amount"] = (clean["quantity"] * clean["unit_price"]).round(0)
    clean["month"] = clean["order_date"].dt.month
    clean["weekday"] = clean["order_date"].dt.day_name()
    clean["order_date"] = clean["order_date"].dt.strftime("%Y-%m-%d")
    write_csv(clean, DATA / "sales_clean.csv")

    write_tab(
        clean.drop(columns=["order_id", "store_id", "customer_id"]),
        DATA / "sales_clean.tab",
        types={"region": "discrete", "channel": "discrete", "order_date": "time",
               "quantity": "continuous", "unit_price": "continuous",
               "amount": "continuous", "month": "continuous",
               "weekday": "discrete"},
    )
    return clean


def make_customers():
    rng = np.random.default_rng(11)
    ids = np.arange(1000, 1800)
    df = pd.DataFrame({
        "customer_id": ids,
        "segment":     rng.choice(["retail", "grosir", "korporat"], len(ids),
                                  p=[0.6, 0.3, 0.1]),
        "city":        rng.choice(["Yogyakarta", "Sleman", "Bantul", "Klaten",
                                   "Magelang", "Solo"], len(ids)),
        "signup_date": (pd.to_datetime("2020-01-01")
                        + pd.to_timedelta(rng.integers(0, 1400, len(ids)), unit="D")
                        ).strftime("%Y-%m-%d"),
        "is_member":   rng.choice([0, 1], len(ids), p=[0.65, 0.35]),
    })
    # 30 customers deliberately absent, so the join has unmatched rows
    df = df[~df["customer_id"].isin(rng.choice(ids, 30, replace=False))]
    write_csv(df, DATA / "customers.csv")


# ================================================================= WEEK 4
def make_ab_test():
    """An A/B test with a real but modest effect, plus a device confounder."""
    rng = np.random.default_rng(2024)
    # Sized so that a 12% relative lift on a ~4.6% baseline is detectable at
    # roughly 90% power -- students should find the effect, not a null result.
    n = 70_000
    variant = rng.choice(["A", "B"], n)
    device = rng.choice(["mobile", "desktop"], n, p=[0.68, 0.32])

    base = np.where(device == "mobile", 0.038, 0.058)      # mobile converts less
    lift = np.where(variant == "B", 1.12, 1.0)             # true effect: +12% rel.
    converted = (rng.random(n) < base * lift).astype(int)

    secs = rng.lognormal(4.7, 0.55, n) + np.where(variant == "B", 4.0, 0.0)
    df = pd.DataFrame({
        "user_id":   np.arange(1, n + 1),
        "variant":   variant,
        "device":    device,
        "day":       rng.integers(1, 15, n),
        "session_seconds": secs.round(1),
        "pages_viewed":    rng.poisson(np.where(variant == "B", 4.3, 4.1), n),
        "converted": converted,
    })
    write_csv(df, DATA / "ab_test.csv")
    write_tab(
        df.drop(columns=["user_id"]),
        DATA / "ab_test.tab",
        types={"variant": "discrete", "device": "discrete", "day": "continuous",
               "session_seconds": "continuous", "pages_viewed": "continuous",
               "converted": "discrete"},
        flags={"converted": "class"},
    )


# ================================================================= WEEK 5
def make_transactions():
    """A transaction log, to be aggregated into per-customer RFM features."""
    rng = np.random.default_rng(5)
    n = 30_000
    cust = rng.integers(1, 2001, n)
    df = pd.DataFrame({
        "customer_id": cust,
        "ts": (pd.Timestamp("2024-01-01")
               + pd.to_timedelta(rng.integers(0, 365 * 24, n), unit="h")
               ).strftime("%Y-%m-%d %H:%M:%S"),
        "amount":   rng.lognormal(12, 0.8, n).round(-2),
        "category": rng.choice(["makanan", "transport", "belanja", "tagihan"],
                               n, p=[0.4, 0.2, 0.28, 0.12]),
    })
    write_csv(df, DATA / "transactions.csv")


# ============================================================= WEEKS 5, 7
def make_credit():
    """Loan applications with a default flag. Signal is real but not trivial."""
    rng = np.random.default_rng(42)
    n = 4000

    age = rng.integers(21, 65, n)
    edu = rng.choice(["SMA", "D3", "S1", "S2"], n, p=[0.35, 0.2, 0.35, 0.10])
    edu_lvl = pd.Series(edu).map({"SMA": 0, "D3": 1, "S1": 2, "S2": 3}).to_numpy()

    income = np.round(np.exp(15.0 + 0.018 * age + 0.22 * edu_lvl
                             + rng.normal(0, 0.45, n)), -3)
    debt = np.round(income * rng.beta(2, 5, n) * 4.5, -3)
    loan_amount = np.round(income * rng.uniform(0.5, 6.0, n), -3)
    term = rng.choice([12, 24, 36, 48, 60], n, p=[0.15, 0.25, 0.3, 0.2, 0.1])
    credit_limit = np.round(income * rng.uniform(0.8, 3.0, n), -3)
    balance = np.round(credit_limit * rng.beta(2, 3, n), -3)
    dependents = rng.poisson(1.3, n).clip(0, 6)
    monthly_payment = np.round(loan_amount / term * 1.12, -2)

    purpose = rng.choice(["modal_usaha", "renovasi", "pendidikan",
                          "kendaraan", "konsumtif"], n,
                         p=[0.28, 0.22, 0.15, 0.2, 0.15])
    housing = rng.choice(["milik", "sewa", "keluarga"], n, p=[0.5, 0.3, 0.2])
    employment = rng.choice(["tetap", "kontrak", "wiraswasta"], n,
                            p=[0.5, 0.22, 0.28])

    # The true risk model: ratios matter, raw amounts mostly do not.
    # Each ratio is centred, so the intercept alone sets the base rate.
    dti = debt / income
    pti = monthly_payment / (income / 12)
    util = balance / credit_limit
    score = (1.15 * (dti - np.nanmean(dti))
             + 0.95 * (pti - np.nanmean(pti))
             + 1.30 * (util - np.nanmean(util))
             - 0.028 * (age - age.mean())
             - 0.24 * (edu_lvl - edu_lvl.mean())
             + 0.13 * (dependents - dependents.mean())
             + np.where(employment == "kontrak", 0.45, 0)
             + np.where(employment == "wiraswasta", 0.30, 0)
             + np.where(housing == "sewa", 0.28, 0)
             + np.where(purpose == "konsumtif", 0.35, 0)
             + rng.normal(0, 0.55, n))
    intercept = solve_intercept(score, target_rate=0.20)
    default = (rng.random(n) < 1 / (1 + np.exp(-(intercept + score)))).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income, "debt": debt,
        "loan_amount": loan_amount, "term_months": term,
        "balance": balance, "credit_limit": credit_limit,
        "dependents": dependents, "monthly_payment": monthly_payment,
        "education": edu, "purpose": purpose, "housing": housing,
        "employment_type": employment,
        "default": np.where(default == 1, "yes", "no"),
    })
    # a little realistic missingness (MAR: older applicants omit income more)
    p_miss = np.clip((age - 21) / 260, 0, 0.25)
    df.loc[rng.random(n) < p_miss, "income"] = np.nan
    df.loc[rng.random(n) < 0.04, "employment_type"] = np.nan

    write_csv(df, DATA / "credit.csv")
    write_tab(
        df, DATA / "credit.tab",
        types={c: "continuous" for c in
               ["age", "income", "debt", "loan_amount", "term_months", "balance",
                "credit_limit", "dependents", "monthly_payment"]}
              | {c: "discrete" for c in
                 ["education", "purpose", "housing", "employment_type", "default"]},
        flags={"default": "class"},
    )
    print(f"    default rate = {(df['default'] == 'yes').mean():.1%}")


# ================================================================= WEEK 6
def make_housing():
    """Yogyakarta-flavoured house prices. Log-linear in area, with a ceiling."""
    rng = np.random.default_rng(99)
    n = 2200

    kec = rng.choice(["Depok", "Ngaglik", "Kasihan", "Sewon", "Mlati",
                      "Umbulharjo", "Gamping"], n,
                     p=[0.18, 0.15, 0.15, 0.14, 0.14, 0.12, 0.12])
    kec_premium = pd.Series(kec).map({
        "Depok": 0.22, "Ngaglik": 0.10, "Kasihan": 0.02, "Sewon": -0.02,
        "Mlati": 0.06, "Umbulharjo": 0.14, "Gamping": -0.06}).to_numpy()

    luas_bangunan = np.round(rng.lognormal(4.5, 0.42, n)).clip(21, 600)
    luas_tanah = np.round(luas_bangunan * rng.uniform(0.9, 2.6, n)).clip(30, 1500)
    kamar_tidur = np.clip((luas_bangunan / 32 + rng.normal(0, 0.8, n)
                           ).round(), 1, 7).astype(int)
    kamar_mandi = np.clip((kamar_tidur / 1.8 + rng.normal(0, 0.5, n)
                           ).round(), 1, 5).astype(int)
    umur = rng.integers(0, 40, n)
    jarak = np.round(rng.gamma(3.2, 1.9, n).clip(0.4, 24), 1)
    sertifikat = rng.choice(["SHM", "HGB", "Girik"], n, p=[0.68, 0.25, 0.07])
    sert_adj = pd.Series(sertifikat).map(
        {"SHM": 0.0, "HGB": -0.09, "Girik": -0.26}).to_numpy()

    log_price = (2.63
                 + 0.62 * np.log(luas_bangunan)
                 + 0.28 * np.log(luas_tanah)
                 + 0.045 * kamar_tidur
                 - 0.0055 * umur
                 - 0.031 * jarak
                 + kec_premium + sert_adj
                 + rng.normal(0, 0.20, n))
    harga = np.exp(log_price)                  # already in million rupiah
    harga = np.minimum(harga, 5_000.0)         # market ceiling: Rp 5 miliar

    df = pd.DataFrame({
        "luas_bangunan": luas_bangunan.astype(int),
        "luas_tanah": luas_tanah.astype(int),
        "kamar_tidur": kamar_tidur,
        "kamar_mandi": kamar_mandi,
        "umur_bangunan": umur,
        "jarak_pusat_km": jarak,
        "kecamatan": kec,
        "sertifikat": sertifikat,
        "harga": harga.round(1),                           # in million rupiah
    })
    write_csv(df, DATA / "rumah_yogya.csv")
    write_tab(
        df, DATA / "rumah_yogya.tab",
        types={"luas_bangunan": "continuous", "luas_tanah": "continuous",
               "kamar_tidur": "continuous", "kamar_mandi": "continuous",
               "umur_bangunan": "continuous", "jarak_pusat_km": "continuous",
               "kecamatan": "discrete", "sertifikat": "discrete",
               "harga": "continuous"},
        flags={"harga": "class"},
    )
    print(f"    price: median {df['harga'].median():.0f} juta, "
          f"max {df['harga'].max():.0f} juta, "
          f"at ceiling: {(df['harga'] >= 4999).sum()}")


if __name__ == "__main__":
    print("Generating practicum datasets into", DATA)
    make_sales()
    make_customers()
    make_ab_test()
    make_transactions()
    make_credit()
    make_housing()
    print("Done.")
