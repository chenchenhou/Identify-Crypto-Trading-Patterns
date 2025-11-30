# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import ElasticNet, Lasso


from pathlib import Path

# ========= 路径配置（改成你的实际路径） =========
PRICE_WIDE_CSV   = "./data/warehouse/major_pairs_1d_um_wide.csv"
FUND_WIDE_CSV    = "./data/warehouse/major_pairs_funding_daily_wide.csv"  # 若不用资金费，可设为 None
USER_PNL_CSV     = "./data/raw/subsampled_users.csv"  # 需要包含 columns: [user_id, timestamp, pnl]

USE_FUNDING      = False   # 首版可先关掉；稳健性再开
ALPHA_GRID       = np.logspace(-4, 1, 30)  # Lasso alpha 搜索网格
N_SPLITS         = 5       # TimeSeriesSplit 折数（>=3）
ROLLING_WINDOW   = None    # 如设为 90，滚动训练 90 天、前瞻预测；None 则一次性拟合
OUT_DIR          = "./data/results"
os.makedirs(OUT_DIR, exist_ok=True)
# ==============================================
# ---- 模型与调参网格 ----
USE_ELASTICNET = True          # 用 ElasticNet（更抗共线）
L1_RATIO_GRID  = [0.7, 0.85, 1.0]
ALPHA_GRID     = np.logspace(-4, 0, 20)   # 下限放到 1e-4，减小“全零”的概率
# ---- 抽样与清洗参数 ----
SAMPLE_FRAC   = 0.001      # 随机抽样比例：1%~2% 之间改这里
MIN_OBS       = 120       # 清洗阈值：用户最少有效样本数（建议 >=120 天）
VAR_EPS       = 1e-10     # 清洗阈值：逐期 pnl 方差下限（近零视为无信息）
RANDOM_SEED   = 2025      # 抽样随机种子（可改，便于复现）

def list_done_user_ids(out_dir=OUT_DIR):
    """读取已产出的 metrics_*.txt，返回已完成的 user_id 集合（断点续跑用）"""
    done = set()
    for p in Path(out_dir).glob("metrics_*.txt"):
        # 文件名形如 metrics_<user_id>.txt
        uid = p.stem[len("metrics_"):]
        done.add(uid)
    return done

def winsorize_series(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

def compute_log_returns(df_price):
    # df_price: columns = [timestamp, XXX_close, YYY_close, ...]
    df = df_price.sort_values("timestamp").copy()
    price_cols = [c for c in df.columns if c.endswith("_close")]
    for c in price_cols:
        df[c] = np.log(df[c]).diff()
    df = df.dropna(subset=price_cols, how="all")
    # 重命名为 *_ret
    df = df.rename(columns={c: c.replace("_close", "_ret") for c in price_cols})
    return df

def load_features():
    # 价格 → 对数收益
    px = pd.read_csv(PRICE_WIDE_CSV)
    # 强制 int64 时间戳
    px["timestamp"] = pd.to_numeric(px["timestamp"], errors="coerce").astype("Int64")
    px = px.dropna(subset=["timestamp"])
    px["timestamp"] = px["timestamp"].astype(np.int64)

    ret_df = compute_log_returns(px)

    if USE_FUNDING and FUND_WIDE_CSV and os.path.exists(FUND_WIDE_CSV):
        fr = pd.read_csv(FUND_WIDE_CSV)
        fr["timestamp"] = pd.to_numeric(fr["timestamp"], errors="coerce").astype("Int64")
        fr = fr.dropna(subset=["timestamp"])
        fr["timestamp"] = fr["timestamp"].astype(np.int64)
        # 合并（外连接）
        feat = pd.merge(ret_df, fr, on="timestamp", how="outer")
    else:
        feat = ret_df

    # 排序、去重
    feat = feat.sort_values("timestamp").drop_duplicates("timestamp")
    return feat  # 列：timestamp, <sym>_ret..., [可选 funding 列...]

def load_user_pnl(csv_path=USER_PNL_CSV):
    import pandas as pd, numpy as np
    # 你的列：id, timestamp, pnl(累计), margin(可选)
    use_cols = ["id", "timestamp", "pnl"]  # margin 暂时不用
    df = pd.read_csv(csv_path, usecols=use_cols).sort_values(["id", "timestamp"]).copy()

    # timestamp -> int64 (UTC 秒)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # 累计 -> 逐期：按用户分组差分；第一条用原值（等价于从0起算）
    df["pnl_period"] = df.groupby("id")["pnl"].diff().fillna(df["pnl"])

    # 轻度 winsorize 防极端（可按需调整/关闭）
    q1, q99 = df["pnl_period"].quantile([0.01, 0.99])
    df["pnl_w"] = df["pnl_period"].clip(q1, q99)

    # 与后续代码约定的列名对齐：user_id, timestamp, pnl_w
    df = df.rename(columns={"id": "user_id"})
    return df[["user_id", "timestamp", "pnl_w"]]

def build_Xy_for_user(feat, pnl_user):
    # 对齐到用户时间段；外连接再在 y 上筛
    df = pd.merge(pnl_user[["timestamp","pnl_w"]], feat, on="timestamp", how="left")
    # 去掉特征全空的行
    Xcols = [c for c in df.columns if c.endswith("_ret") or c.endswith("_funding_sum")]
    df = df.dropna(subset=Xcols, how="all")
    # 进一步：若任意特征 NaN，先用列均值填充或直接剔除该行（保守起见，这里剔除）
    df = df.dropna(subset=Xcols, how="any")
    X = df[Xcols].values
    y = df["pnl_w"].values
    t = df["timestamp"].values
    return X, y, t, Xcols

def time_series_split_idx(n, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n)))

def fit_lasso_time_series(X, y, alpha_grid=ALPHA_GRID, n_splits=N_SPLITS):
    splits = time_series_split_idx(len(y), n_splits=n_splits)
    if USE_ELASTICNET:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ls", ElasticNet(max_iter=200000, tol=1e-3))
        ])
        gs = GridSearchCV(
            pipe,
            param_grid={"ls__alpha": alpha_grid, "ls__l1_ratio": L1_RATIO_GRID},
            cv=splits,
            n_jobs=None
        )
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ls", Lasso(max_iter=200000, tol=1e-3))
        ])
        gs = GridSearchCV(
            pipe,
            param_grid={"ls__alpha": alpha_grid},
            cv=splits,
            n_jobs=None
        )
    gs.fit(X, y)
    return gs

def evaluate_oos(model, X, y, splits):
    # 用最后一折做 OOS 评估（或把每折验证期拼起来评估也行）
    tr_idx, te_idx = splits[-1]
    y_pred = model.predict(X[te_idx])
    return {
        "r2_oos": r2_score(y[te_idx], y_pred),
        "rmse_oos": np.sqrt(mean_squared_error(y[te_idx], y_pred)),
        "n_test": int(te_idx.size),
        "best_alpha": model.best_params_.get("ls__alpha", None)
    }

def run_for_user(user_id, feat, pnl):
    pnl_u = pnl[pnl["user_id"] == user_id].sort_values("timestamp").copy()
    if pnl_u.empty:
        return None

    X, y, ts, Xcols = build_Xy_for_user(feat, pnl_u)
    if len(y) < 100:
        print(f"[WARN] user {user_id}: too few samples ({len(y)}).")
        return None

    # 可选：滚动窗口
    if ROLLING_WINDOW:
        results = []
        W = int(ROLLING_WINDOW)
        for end in range(W, len(y)):
            X_tr, y_tr = X[end-W:end], y[end-W:end]
            X_te, y_te = X[end:end+1], y[end:end+1]
            model = fit_lasso_time_series(X_tr, y_tr, ALPHA_GRID, max(3, min(N_SPLITS, W//20)))
            yhat = model.predict(X_te)[0]
            results.append((ts[end], y_te[0], yhat))
        df_pred = pd.DataFrame(results, columns=["timestamp","y_true","y_pred"])
        df_pred.to_csv(os.path.join(OUT_DIR, f"pred_{user_id}.csv"), index=False)
        return {"user_id": user_id, "rolling_points": len(df_pred)}
    else:
        # === 训练完成后（model 是 GridSearchCV 返回的） ===
        splits = time_series_split_idx(len(y), n_splits=N_SPLITS)
        model  = fit_lasso_time_series(X, y, ALPHA_GRID, N_SPLITS)
        metrics = evaluate_oos(model, X, y, splits)

        # ====== 从最优管线中取出 scaler 与 lasso ======
        best_pipe = model.best_estimator_                  # Pipeline(scaler -> lasso)
        scaler    = best_pipe.named_steps["scaler"]        # StandardScaler
        lasso     = best_pipe.named_steps["ls"]            # Lasso

        # ---- 1) 标准化空间下的系数（与模型拟合时一致） ----
        beta_std = lasso.coef_.copy()
        b0_std   = lasso.intercept_

        # ---- 2) 还原到“原始量纲”的系数（可选，更易解读） ----
        #   X_raw -> 标准化:  X_std = (X_raw - mean_) / scale_
        #   y 未标准化，因此 beta_raw = beta_std / scale_
        #   截距还原: b0_raw = b0_std - sum( mean_/scale_ * beta_std )
        import numpy as np
        scale = np.asarray(scaler.scale_, dtype=float)
        mean  = np.asarray(scaler.mean_,  dtype=float)

        # 避免除以 0（极少数常量列）
        safe_scale = np.where(scale == 0.0, 1.0, scale)

        beta_raw = beta_std / safe_scale
        b0_raw   = b0_std - np.sum((mean / safe_scale) * beta_std)

        # ====== 组装并导出 ======
        import pandas as pd
        df_coef = pd.DataFrame({
            "feature":  Xcols,
            "coef_std": beta_std,           # 标准化空间的系数（用于稀疏选择/比较相对重要性）
            "coef_raw": beta_raw,           # 原始量纲的系数（用于可解释性/与PnL量纲关联）
            "abs_coef": np.abs(beta_std)    # 习惯上用标准化系数做“强度排序”
        }).sort_values("abs_coef", ascending=False)

        df_coef.to_csv(os.path.join(OUT_DIR, f"coef_{user_id}.csv"), index=False)
        with open(os.path.join(OUT_DIR, f"metrics_{user_id}.txt"), "w") as f:
            f.write(str({**metrics, "best_alpha": model.best_params_.get("ls__alpha", None)}))

        # Top-N 非零（用标准化系数判断稀疏性更稳）
        top_nonzero = df_coef[df_coef["abs_coef"] > 1e-8].head(10)["feature"].tolist()

        return {
            "user_id": user_id,
            "top_features": top_nonzero,
            **metrics
        }

def main():
    # 1) 载入特征 & 用户逐期 pnl
    feat = load_features()
    pnl  = load_user_pnl()  # 列：user_id, timestamp, pnl_w（已做累计->逐期与 winsorize）

    # 2) 先做“用户层清洗”：样本数 & 方差
    stats = pnl.groupby("user_id").agg(
        n=("pnl_w","size"),
        v=("pnl_w","var")
    ).reset_index()
    eligible = stats[(stats["n"] >= MIN_OBS) & (stats["v"].fillna(0) >= VAR_EPS)]["user_id"].tolist()

    # 3) 跳过已完成的用户（断点续跑）
    done_ids   = list_done_user_ids(OUT_DIR)
    remaining  = [u for u in eligible if u not in done_ids]

    # 4) 随机抽样 1–2%（至少留 1 个）
    import numpy as np
    rng = np.random.RandomState(RANDOM_SEED)
    k   = max(1, int(len(remaining) * SAMPLE_FRAC))
    if k < len(remaining):
        sampled = sorted(rng.choice(remaining, size=k, replace=False).tolist())
    else:
        sampled = remaining  # 剩余很少时就全跑

    print(f"[INFO] users_total={pnl['user_id'].nunique()} "
          f"eligible={len(eligible)} remaining={len(remaining)} sampled={len(sampled)}")

    # 5) 逐用户训练（沿用你现有的 Lasso+CV 流程）
    summary = []
    for i, uid in enumerate(sampled, 1):
        res = run_for_user(uid, feat, pnl)
        if res:
            summary.append(res)
        if i % 100 == 0:
            # 期间性落盘，防止意外中断丢进度
            pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    # 6) 汇总落盘
    if summary:
        df_sum = pd.DataFrame(summary)
        df_sum.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
        print(df_sum.head(10))
    else:
        print("[WARN] no results after sampling")


if __name__ == "__main__":
    main()
