"""
scripts/fetch_btc_daily.py

只下载 BTCUSDT 的 1d K 线（U本位期货，月度打包），
合并为日频（timestamp=当天 00:00:00 UTC 的秒），只保留 timestamp 与 close，
并保存为 Parquet/CSV。

如需现货，把 KLINE_BASE_PATH 改为 "/data/spot/monthly/klines" 即可。
"""

import os
import io
import time
import zipfile
import requests
import pandas as pd
from datetime import datetime, timezone

# ======== 配置 ========
# 时间范围（Unix 秒, UTC）
START_TS = 1596240000  # 2020-08-01 00:00:00 UTC
END_TS   = 1761782400  # 2025-10-30 00:00:00 UTC

SYMBOL = "BTCUSDT"
INTERVAL = "1d"

BASE_HOST = "https://data.binance.vision"
# U本位期货月度K线：
KLINE_BASE_PATH = "/data/futures/um/monthly/klines"
# 如需现货：
# KLINE_BASE_PATH = "/data/spot/monthly/klines"

OUT_DIR = "../data/warehouse"
OUT_PARQUET = os.path.join(OUT_DIR, "btc_usdt_daily.parquet")
OUT_CSV     = os.path.join(OUT_DIR, "btc_usdt_daily.csv")

TIMEOUT = 20
SLEEP_BETWEEN = 0.03  # 每个请求间隔，防止偶发超时
# =====================


def month_range_utc(start_ts: int, end_ts: int):
    """生成 [start_ts, end_ts] 覆盖到的 (YYYY, MM) 列表（UTC）。"""
    s = datetime.fromtimestamp(start_ts, tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    e = datetime.fromtimestamp(end_ts, tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = []
    y, m = s.year, s.month
    while (y < e.year) or (y == e.year and m <= e.month):
        months.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return months


def _get(url: str) -> requests.Response:
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        return resp
    except Exception as e:
        print(f"[WARN] GET failed: {url} -> {e}")
        r = requests.Response()
        r.status_code = 404
        r._content = b""
        return r


def fetch_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    """
    下载 {symbol}-{INTERVAL}-{YYYY}-{MM}.zip，返回列：timestamp(秒, 日度00:00UTC)、close
    """
    fname = f"{symbol}-{INTERVAL}-{year:04d}-{month:02d}.zip"
    url = f"{BASE_HOST}{KLINE_BASE_PATH}/{symbol}/{INTERVAL}/{fname}"
    resp = _get(url)
    if resp.status_code == 404:
        return pd.DataFrame(columns=["timestamp", "close"])
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] {url} -> {e}")
        return pd.DataFrame(columns=["timestamp", "close"])

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        inner = [n for n in zf.namelist() if n.endswith(".csv")]
        if not inner:
            return pd.DataFrame(columns=["timestamp", "close"])
        with zf.open(inner[0]) as f:
            # K线无表头；但仍做兼容处理
            df = pd.read_csv(
                f,
                header=None,
                names=[
                    "open_time","open","high","low","close","volume",
                    "close_time","quote_asset_volume","number_of_trades",
                    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
                ],
            )

    # 强制数值化 + 日度对齐到 00:00:00 UTC
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close"]     = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["open_time","close"])

    dt = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.floor("D")
    df["timestamp"] = (dt.astype("int64") // 10**9).astype("int64")

    out = (
        df[["timestamp","close"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out


def fetch_btc_daily(start_ts: int, end_ts: int) -> pd.DataFrame:
    parts = []
    for y, m in month_range_utc(start_ts, end_ts):
        dfm = fetch_month(SYMBOL, y, m)
        if not dfm.empty:
            parts.append(dfm)
        time.sleep(SLEEP_BETWEEN)
    if not parts:
        return pd.DataFrame(columns=["timestamp","close"])
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] Fetching {SYMBOL} {INTERVAL} from {START_TS} to {END_TS} (UTC)...")
    df = fetch_btc_daily(START_TS, END_TS)
    print(f"[INFO] rows={len(df)}; first={df['timestamp'].min()} last={df['timestamp'].max()}")

    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] saved -> {OUT_PARQUET}")
    print(f"[INFO] saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
