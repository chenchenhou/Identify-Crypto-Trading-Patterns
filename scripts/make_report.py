# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import glob, os
plt.rcParams["figure.dpi"] = 120

RES = Path("./data/results")
OUT = Path("./reports"); OUT.mkdir(parents=True, exist_ok=True)

def r2_rmse_panels():
    df = pd.read_csv(RES/"summary.csv")
    fig = plt.figure(figsize=(10,4))
    # 用 in-sample 指标
    plt.subplot(1,2,1)
    df["r2_in"].hist(bins=40)
    plt.title("In-sample R²")

    plt.subplot(1,2,2)
    q99 = df["rmse_in"].quantile(0.99)
    df.loc[df["rmse_in"] <= q99, "rmse_in"].hist(bins=40)
    plt.title("RMSE (<=99%, in-sample)")

    plt.tight_layout()
    plt.savefig(OUT/"dist_r2_rmse.png")
    plt.close()


def top_pairs_bar(n=20):
    from collections import Counter
    cnt = Counter()
    for fp in glob.glob(str(RES/"coef_*.csv")):
        c = pd.read_csv(fp)
        c = c[c["abs_coef"]>1e-8].sort_values("abs_coef", ascending=False).head(5)
        for f in c["feature"]:
            if f.endswith("_ret"): cnt[f.replace("_ret","")] += 1
    rk = pd.DataFrame(cnt.items(), columns=["pair","count"]).sort_values("count", ascending=False).head(n)
    rk.plot(kind="barh", x="pair", y="count", figsize=(8,6)); plt.title("Most-picked pairs (Top-5 per user)")
    plt.tight_layout(); plt.savefig(OUT/"top_pairs.png"); plt.close()

def per_user_report(uid):
    fp = RES/f"coef_{uid}.csv"
    if not Path(fp).exists(): return
    c = pd.read_csv(fp).sort_values("abs_coef", ascending=False)
    fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot(2,1,1)
    top = c.head(20)
    ax1.barh(top["feature"][::-1], top["coef_std"][::-1]); ax1.set_title(f"Top Coeff (std) - {uid}")
    ax2 = plt.subplot(2,1,2)
    top_raw = c.sort_values("coef_raw", key=lambda s: s.abs(), ascending=False).head(20)
    ax2.barh(top_raw["feature"][::-1], top_raw["coef_raw"][::-1]); ax2.set_title(f"Top Coeff (raw) - {uid}")
    plt.tight_layout(); plt.savefig(OUT/f"user_{uid}_coef.png"); plt.close()

def main():
    r2_rmse_panels()
    top_pairs_bar()
    # 也给 R² 前 10 的用户各画一张
    df = pd.read_csv(RES/"summary.csv").sort_values("r2_in", ascending=False).head(10)

    for uid in df["user_id"]:
        per_user_report(uid)
    print(f"done. see {OUT}")

if __name__ == "__main__":
    main()
