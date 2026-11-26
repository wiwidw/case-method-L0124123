"""
Case Method Data Mining - Aturan Asosiasi dengan algoritma SETM (SET-oriented Mining)
Dataset: Bakery / Bread Basket London (kolom: Transaction, Item, date_time, period_day, weekday_weekend)

Alur:
  1. Preprocessing  : strip nama item, buang item kosong/NONE, buang item non-menu, hapus duplikat item per transaksi
  2. SETM           : R1 -> L1 -> (join, sort, count, filter) berulang sampai L_k kosong
  3. Aturan         : support, confidence, lift
  4. Segmen         : bandingkan morning vs afternoon dan weekday vs weekend
  5. Sanity check   : hasil SETM dicocokkan dengan hitung brute-force

Cara pakai:
  python setm_bakery.py                                   (default: min support 1%, min confidence 30%)
  python setm_bakery.py --min-support 0.005 --min-conf 0.4
  python setm_bakery.py --csv "bread basket in london.csv" --out hasil_setm
"""

import argparse
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

# Item yang bukan menu makanan/minuman (barang retail, penyesuaian kasir, dll).
# Silakan ubah daftar ini sesuai pertimbanganmu, dan tulis alasannya di laporan.
NON_MENU = {
    "adjustment", "gift voucher", "postcard", "tshirt", "valentine's card",
    "nomad bag", "fairy doors", "basket", "christmas common",
    "coffee granules", "drinking chocolate spoons", "art tray",
}


# --------------------------------------------------------------------------
# 1. PREPROCESSING
# --------------------------------------------------------------------------
def load_and_clean(path):
    raw = pd.read_csv(path)
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    print(f"Data mentah        : {len(raw):,} baris | {raw['Transaction'].nunique():,} transaksi "
          f"| {raw['Item'].nunique()} item unik")

    df = raw.copy()
    df["Item"] = df["Item"].astype(str).str.strip()

    n = len(df)
    df = df[(df["Item"] != "") & (df["Item"].str.upper() != "NONE")]
    print(f"Item kosong/NONE   : {n - len(df):,} baris dibuang")

    n = len(df)
    df = df[~df["Item"].str.lower().isin(NON_MENU)]
    print(f"Item non-menu      : {n - len(df):,} baris dibuang")

    n = len(df)
    df = df.drop_duplicates(subset=["Transaction", "Item"])
    print(f"Duplikat item/trx  : {n - len(df):,} baris dibuang")

    df = df.reset_index(drop=True)
    sizes = df.groupby("Transaction").size()
    print(f"Data bersih        : {len(df):,} baris | {df['Transaction'].nunique():,} transaksi "
          f"| {df['Item'].nunique()} item unik")
    print(f"Item per transaksi : rata-rata {sizes.mean():.2f}, maks {sizes.max()}, "
          f"1 item saja = {(sizes == 1).sum():,} transaksi")
    print("\n10 item terlaris:")
    print(df["Item"].value_counts().head(10).to_string())
    return df


# --------------------------------------------------------------------------
# 2. SETM
# --------------------------------------------------------------------------
def setm(trans, min_count):
    """
    trans: DataFrame dengan kolom ['tid', 'item'] (satu item per transaksi, tidak duplikat).
    Mengembalikan dict {k: DataFrame L_k} dengan kolom item1..itemk dan count.

    Tabel yang dipakai (sesuai paper SETM):
      R1  : (tid, item1)                 semua item yang frequent
      R'k : R(k-1) join R1 pada tid      dengan syarat item baru > item terakhir
      Ck  : R'k di-sort lalu group-by itemset, dihitung kemunculannya
      Lk  : Ck yang count >= min_count
      Rk  : R'k yang itemset-nya ada di Lk (diurutkan per tid)
    """
    R1 = trans[["tid", "item"]].rename(columns={"item": "item1"})
    R1 = R1.sort_values(["item1", "tid"]).reset_index(drop=True)

    # Iterasi 1: C1 -> L1, lalu buang baris item yang tidak frequent
    C1 = R1.groupby("item1").size().rename("count").reset_index()
    L1 = C1[C1["count"] >= min_count].reset_index(drop=True)
    levels = {1: L1}
    R1 = R1.merge(L1[["item1"]], on="item1")
    Rk = R1.sort_values(["tid", "item1"]).reset_index(drop=True)
    base = R1.rename(columns={"item1": "new_item"})

    k = 2
    while True:
        # JOIN: R(k-1) x R1 pada tid yang sama, item baru harus lebih besar dari item terakhir
        Rp = Rk.merge(base, on="tid")
        Rp = Rp[Rp[f"item{k-1}"] < Rp["new_item"]].rename(columns={"new_item": f"item{k}"})
        if Rp.empty:
            break

        # SORT + COUNT: urutkan berdasarkan itemset, lalu hitung dengan group-by
        cols = [f"item{i}" for i in range(1, k + 1)]
        Rp = Rp.sort_values(cols).reset_index(drop=True)
        Ck = Rp.groupby(cols).size().rename("count").reset_index()

        # FILTER: hanya yang lolos min support
        Lk = Ck[Ck["count"] >= min_count].reset_index(drop=True)
        if Lk.empty:
            break
        levels[k] = Lk

        # Buang baris R'k yang itemset-nya tidak frequent -> Rk, urutkan per tid
        Rk = Rp.merge(Lk[cols], on=cols).sort_values(["tid"] + cols).reset_index(drop=True)
        k += 1
    return levels


def itemset_counts(levels):
    """Ubah hasil SETM menjadi dict {frozenset(itemset): count}."""
    counts = {}
    for k, Lk in levels.items():
        for row in Lk.itertuples(index=False):
            counts[frozenset(row[:k])] = row[k]
    return counts


# --------------------------------------------------------------------------
# 3. ATURAN ASOSIASI
# --------------------------------------------------------------------------
def build_rules(counts, n_trans, min_conf):
    rows = []
    for itemset, cnt_xy in counts.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for r in range(1, len(items)):
            for ant in combinations(items, r):
                ant_set = frozenset(ant)
                cons_set = itemset - ant_set
                cnt_x, cnt_y = counts.get(ant_set), counts.get(cons_set)
                if not cnt_x or not cnt_y:
                    continue
                conf = cnt_xy / cnt_x
                if conf < min_conf:
                    continue
                rows.append({
                    "antecedent": ", ".join(ant),
                    "consequent": ", ".join(sorted(cons_set)),
                    "support_count": cnt_xy,
                    "support": cnt_xy / n_trans,
                    "confidence": conf,
                    "lift": conf / (cnt_y / n_trans),
                })
    cols = ["antecedent", "consequent", "support_count", "support", "confidence", "lift"]
    rules = pd.DataFrame(rows, columns=cols)
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. SANITY CHECK (hitung brute-force, tanpa SETM)
# --------------------------------------------------------------------------
def verify_bruteforce(trans, levels, min_count):
    max_k = max(levels) + 1
    cnt = Counter()
    for _, items in trans.groupby("tid")["item"]:
        items = sorted(items)
        for k in range(1, min(len(items), max_k) + 1):
            cnt.update(combinations(items, k))
    expected = {frozenset(s): c for s, c in cnt.items() if c >= min_count}
    return expected == itemset_counts(levels)


# --------------------------------------------------------------------------
# JALANKAN SETM + SIMPAN HASIL
# --------------------------------------------------------------------------
def run_setm(df, min_support, min_conf, label, outdir, verify=True):
    trans = df[["Transaction", "Item"]].rename(columns={"Transaction": "tid", "Item": "item"})
    n_trans = trans["tid"].nunique()
    min_count = max(1, int(-(-min_support * n_trans // 1)))  # pembulatan ke atas

    print("\n" + "=" * 60)
    print(f"SETM - {label}")
    print("=" * 60)
    print(f"Transaksi: {n_trans:,} | min support {min_support:.2%} (= {min_count} transaksi) "
          f"| min confidence {min_conf:.0%}")

    t0 = time.perf_counter()
    levels = setm(trans, min_count)
    elapsed = time.perf_counter() - t0

    counts = itemset_counts(levels)
    for k, Lk in levels.items():
        print(f"  L{k}: {len(Lk)} itemset frequent")
    print(f"Waktu SETM: {elapsed:.3f} detik")

    if verify:
        ok = verify_bruteforce(trans, levels, min_count)
        print("Sanity check brute-force:", "SAMA (OK)" if ok else "BEDA, cek kode!")

    rules = build_rules(counts, n_trans, min_conf)
    print(f"Jumlah aturan (confidence >= {min_conf:.0%}): {len(rules)}")
    if len(rules):
        show = rules.head(10).copy()
        for c in ["support", "confidence"]:
            show[c] = (show[c] * 100).round(2).astype(str) + "%"
        show["lift"] = show["lift"].round(2)
        print("\nTop 10 aturan (urut lift):")
        print(show.to_string(index=False))

    # simpan CSV
    tag = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    rows = []
    for itemset, c in counts.items():
        rows.append({"itemset": " + ".join(sorted(itemset)), "ukuran": len(itemset),
                     "count": c, "support": c / n_trans})
    freq = pd.DataFrame(rows).sort_values(["ukuran", "count"], ascending=[True, False])
    freq.to_csv(outdir / f"frequent_itemsets_{tag}.csv", index=False)
    rules.to_csv(outdir / f"rules_{tag}.csv", index=False)
    return levels, counts, rules, n_trans


# --------------------------------------------------------------------------
# VISUALISASI (opsional, perlu matplotlib)
# --------------------------------------------------------------------------
def make_plots(df, rules, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib tidak terpasang, grafik dilewati. Install: pip install matplotlib)")
        return

    top = df["Item"].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.values)
    ax.set_title("10 Item Terlaris")
    ax.set_xlabel("Jumlah transaksi")
    fig.tight_layout()
    fig.savefig(outdir / "top_items.png", dpi=150)
    plt.close(fig)

    if len(rules):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sc = ax.scatter(rules["support"] * 100, rules["confidence"] * 100,
                        c=rules["lift"], s=60, cmap="viridis")
        fig.colorbar(sc, label="Lift")
        ax.set_xlabel("Support (%)")
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Aturan Asosiasi: Support vs Confidence")
        fig.tight_layout()
        fig.savefig(outdir / "rules_support_confidence.png", dpi=150)
        plt.close(fig)
    print(f"Grafik disimpan di folder {outdir}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SETM pada dataset Bakery / Bread Basket")
    ap.add_argument("--csv", default="bread basket in london.csv")
    ap.add_argument("--min-support", type=float, default=0.01, help="contoh 0.01 = 1%%")
    ap.add_argument("--min-conf", type=float, default=0.3, help="contoh 0.3 = 30%%")
    ap.add_argument("--out", default="hasil_setm")
    ap.add_argument("--no-segments", action="store_true", help="lewati perbandingan segmen")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(args.csv)

    # SETM pada seluruh data
    _, _, rules, _ = run_setm(df, args.min_support, args.min_conf, "Semua data", outdir)
    if not args.no_plot:
        make_plots(df, rules, outdir)

    # Perbandingan segmen (evening/night diabaikan karena datanya sangat sedikit)
    if not args.no_segments:
        attr = df.groupby("Transaction")[["period_day", "weekday_weekend"]].first()
        if (df.groupby("Transaction")[["period_day", "weekday_weekend"]].nunique() > 1).any().any():
            print("\nPeringatan: ada transaksi dengan period_day/weekday_weekend yang tidak konsisten.")
        df = df.join(attr, on="Transaction", rsuffix="_trx")
        segmen = [
            ("Morning", df["period_day_trx"] == "morning"),
            ("Afternoon", df["period_day_trx"] == "afternoon"),
            ("Weekday", df["weekday_weekend_trx"] == "weekday"),
            ("Weekend", df["weekday_weekend_trx"] == "weekend"),
        ]
        for nama, mask in segmen:
            run_setm(df[mask], args.min_support, args.min_conf, nama, outdir, verify=False)

    print(f"\nSemua file hasil (CSV) ada di folder: {outdir}")


if __name__ == "__main__":
    main()
