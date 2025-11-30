# -*- coding: utf-8 -*-
"""
批量抓取主要交易对的日频收盘价（Binance Vision 月度K线），并可选抓取UM永续的资金费率。
用法示例：
  # 期货(UM) + 资金费率
  python fetch_major_pairs.py --market um --with-funding
  # 现货
  python fetch_major_pairs.py --market spot
  # 指定币种
  python fetch_major_pairs.py --market um --symbols BTCUSDT ETHUSDT SOLUSDT

说明：
- 价格来源（按 --market）：
    spot -> /data/spot/monthly/klines
    um   -> /data/futures/um/monthly/klines
- 资金费率（仅 --market um 且 --with-funding 才抓）：
    /data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-YYYY-MM.zip
  若某月或某币对无文件，会自动跳过。
"""

import os
import io
import re
import time
import zipfile
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone

# ==================== 默认配置（可改） ====================
MAJOR_SYMBOLS = [
    "BTCUSDT","ETHUSDT",
    "BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","TRXUSDT",
    "TONUSDT","AVAXUSDT","LINKUSDT","LTCUSDT",
]
INTERVAL = "1d"
START_TS = 1596240000  # 2020-08-01 00:00:00 UTC
END_TS   = 1761782400  # 2025-10-30 00:00:00 UTC
OUT_DIR  = "./data/warehouse"
SLEEP_BETWEEN = 0.03
TIMEOUT = 20
BASE_HOST = "https://data.binance.vision"
# =========================================================

def month_range_utc(start_ts: int, end_ts: int):
    s = datetime.fromtimestamp(start_ts, tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    e = datetime.fromtimestamp(end_ts,   tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = []
    y, m = s.year, s.month
    while (y < e.year) or (y == e.year and m <= e.month):
        months.append((y, m))
        y, m = (y+1, 1) if m == 12 else (y, m+1)
    return months

def _get(url: str) -> requests.Response:
    try:
        return requests.get(url, timeout=TIMEOUT)
    except Exception as e:
        print(f"[WARN] GET failed: {url} -> {e}")
        r = requests.Response()
        r.status_code = 404
        r._content = b""
        return r

def kline_base_path(market: str) -> str:
    if market == "spot":
        return "/data/spot/monthly/klines"
    elif market == "um":
        return "/data/futures/um/monthly/klines"
    else:
        raise ValueError("market must be 'spot' or 'um'")

def fetch_kline_month(symbol: str, market: str, year: int, month: int) -> pd.DataFrame:
    base = kline_base_path(market)
    fname = f"{symbol}-{INTERVAL}-{year:04d}-{month:02d}.zip"
    url   = f"{BASE_HOST}{base}/{symbol}/{INTERVAL}/{fname}"
    resp  = _get(url)
    if resp.status_code == 404:
        return pd.DataFrame(columns=["timestamp","close"])
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] {url} -> {e}")
        return pd.DataFrame(columns=["timestamp","close"])

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        inner = [n for n in zf.namelist() if n.endswith(".csv")]
        if not inner:
            return pd.DataFrame(columns=["timestamp","close"])
        with zf.open(inner[0]) as f:
            df = pd.read_csv(
                f, header=None,
                names=[
                    "open_time","open","high","low","close","volume",
                    "close_time","quote_asset_volume","number_of_trades",
                    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
                ],
            )
    # 对齐到日频 00:00:00 UTC，仅保留 timestamp/close
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close"]     = pd.to_numeric(df["close"],     errors="coerce")
    df = df.dropna(subset=["open_time","close"])
    dt = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.floor("D")
    df["timestamp"] = (dt.astype("int64") // 10**9).astype("int64")
    out = (df[["timestamp","close"]]
           .drop_duplicates("timestamp")
           .sort_values("timestamp")
           .reset_index(drop=True))
    return out

# ---- 资金费率（仅 UM 永续） ----
def fetch_funding_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    """
    下载 {symbol}-fundingRate-YYYY-MM.zip
    典型CSV列：symbol, fundingTime(ms), fundingRate, markPrice(可选)
    输出：timestamp(日频00:00UTC), funding_rate(按日聚合：求和或均值都可，这里默认“按日求和”)
    """
    # 资金费率归档：/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-YYYY-MM.zip
    fname = f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
    url   = f"{BASE_HOST}/data/futures/um/monthly/fundingRate/{symbol}/{fname}"
    resp  = _get(url)
    if resp.status_code == 404:
        return pd.DataFrame(columns=["timestamp","funding_rate"])
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] {url} -> {e}")
        return pd.DataFrame(columns=["timestamp","funding_rate"])

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csvs = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csvs:
            return pd.DataFrame(columns=["timestamp","funding_rate"])
        with zf.open(csvs[0]) as f:
            # 常见字段名可能为: "symbol","fundingTime","fundingRate","markPrice"
            raw = pd.read_csv(f)
    # 归一化字段名（兼容大小写/不同命名）
    cols = {c.lower(): c for c in raw.columns}
    ftime_col = cols.get("fundingtime") or cols.get("time") or "fundingTime"
    frate_col = cols.get("fundingrate") or "fundingRate"
    if ftime_col not in raw.columns or frate_col not in raw.columns:
        return pd.DataFrame(columns=["timestamp","funding_rate"])

    df = raw[[ftime_col, frate_col]].copy()
    # fundingTime 可能为毫秒或秒，做健壮转换
    t = pd.to_numeric(df[ftime_col], errors="coerce")
    # 猜测是否毫秒级：大于 10^12 基本可判为 ms
    unit = "ms" if t.dropna().median() > 10**12 else "s"
    dt = pd.to_datetime(t, unit=unit, utc=True).dt.floor("D")
    df["timestamp"] = (dt.astype("int64") // 10**9).astype("int64")
    df["funding_rate"] = pd.to_numeric(df[frate_col], errors="coerce")
    df = df.dropna(subset=["timestamp","funding_rate"])

    # 8小时一笔 -> 聚合到“日”（可以 sum 或 mean；这里给出 sum，更接近“日度总资金费”）
    daily = (df.groupby("timestamp", as_index=False)["funding_rate"]
               .sum()
               .sort_values("timestamp")
               .reset_index(drop=True))
    return daily

def fetch_symbol(market: str, symbol: str, with_funding: bool, start_ts: int, end_ts: int):
    # 价格
    parts = []
    for y, m in month_range_utc(start_ts, end_ts):
        dfm = fetch_kline_month(symbol, market, y, m)
        if not dfm.empty:
            parts.append(dfm)
        time.sleep(SLEEP_BETWEEN)
    price = (pd.concat(parts, ignore_index=True) if parts else
             pd.DataFrame(columns=["timestamp","close"]))
    price = price[(price["timestamp"] >= start_ts) & (price["timestamp"] <= end_ts)]
    price = price.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # 资金费率
    funding = None
    if with_funding and market == "um":
        fparts = []
        for y, m in month_range_utc(start_ts, end_ts):
            dfm = fetch_funding_month(symbol, y, m)
            if not dfm.empty:
                fparts.append(dfm)
            time.sleep(SLEEP_BETWEEN)
        if fparts:
            funding = (pd.concat(fparts, ignore_index=True)
                       .drop_duplicates("timestamp")
                       .sort_values("timestamp")
                       .reset_index(drop=True))

    return price, funding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["spot","um"], default="um",
                        help="数据市场：spot=现货, um=U本位永续期货")
    parser.add_argument("--with-funding", action="store_true",
                        help="(仅um有效) 抓取资金费率并按日聚合")
    parser.add_argument("--symbols", nargs="*", default=MAJOR_SYMBOLS,
                        help="自定义币对列表，默认抓主要币种")
    parser.add_argument("--start-ts", type=int, default=START_TS)
    parser.add_argument("--end-ts",   type=int, default=END_TS)
    parser.add_argument("--out-dir",  type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    merged = None
    merged_f = None

    print(f"[INFO] market={args.market}, with_funding={args.with_funding}")
    print(f"[INFO] symbols={args.symbols}")

    for sym in args.symbols:
        print(f"[INFO] Fetching {sym} {INTERVAL} {args.market} from {args.start_ts} to {args.end_ts} ...")
        price, funding = fetch_symbol(args.market, sym, args.with_funding, args.start_ts, args.end_ts)

        # 保存单币种明细
        p_parquet = os.path.join(args.out_dir, f"{sym.lower()}_{INTERVAL}_{args.market}.parquet")
        p_csv     = os.path.join(args.out_dir, f"{sym.lower()}_{INTERVAL}_{args.market}.csv")
        price.to_parquet(p_parquet, index=False)
        price.to_csv(p_csv, index=False)
        print(f"[INFO] saved -> {p_parquet}")
        print(f"[INFO] saved -> {p_csv}")

        # 合并价格到宽表
        colp = f"{sym}_close"
        px = price.rename(columns={"close": colp})
        merged = px if merged is None else pd.merge(merged, px, on="timestamp", how="outer")

        # 资金费率
        if funding is not None:
            f_parquet = os.path.join(args.out_dir, f"{sym.lower()}_funding_daily.parquet")
            f_csv     = os.path.join(args.out_dir, f"{sym.lower()}_funding_daily.csv")
            funding.to_parquet(f_parquet, index=False)
            funding.to_csv(f_csv, index=False)
            print(f"[INFO] saved -> {f_parquet}")
            print(f"[INFO] saved -> {f_csv}")

            colf = f"{sym}_funding_sum"
            fx = funding.rename(columns={"funding_rate": colf})
            merged_f = fx if merged_f is None else pd.merge(merged_f, fx, on="timestamp", how="outer")

    # 输出合并后的宽表
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    wide_parquet = os.path.join(args.out_dir, f"major_pairs_{INTERVAL}_{args.market}_wide.parquet")
    wide_csv     = os.path.join(args.out_dir, f"major_pairs_{INTERVAL}_{args.market}_wide.csv")
    merged.to_parquet(wide_parquet, index=False)
    merged.to_csv(wide_csv, index=False)
    print(f"[INFO] merged price saved -> {wide_parquet}")
    print(f"[INFO] merged price saved -> {wide_csv}")

    if args.with_funding and args.market == "um" and merged_f is not None:
        merged_f = merged_f.sort_values("timestamp").reset_index(drop=True)
        f_parquet = os.path.join(args.out_dir, f"major_pairs_funding_daily_wide.parquet")
        f_csv     = os.path.join(args.out_dir, f"major_pairs_funding_daily_wide.csv")
        merged_f.to_parquet(f_parquet, index=False)
        merged_f.to_csv(f_csv, index=False)
        print(f"[INFO] merged funding saved -> {f_parquet}")
        print(f"[INFO] merged funding saved -> {f_csv}")

if __name__ == "__main__":
    main()
