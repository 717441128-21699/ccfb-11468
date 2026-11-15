"""生成曝光/点击/下单三份日志 + 登录、退款、设备表,内置真实因果设定。

真实因果设定(ground truth):
    用户转化 logit = base + Σ_c effect_c * (用户触点中渠道 c 的次数)
    effect_c 即 config.CHANNELS[*]["effect"], 归因的"真实份额"按各渠道
    效应贡献在订单回溯窗口内归一化得到(见 pipeline.build_truth)。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .config import (CAMPAIGNS, CHANNEL_CTR, CHANNEL_EFFECT, CHANNEL_MIX,
                     CHANNEL_NAMES, Config)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: Config) -> dict:
    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    t0, span = cfg.t0, cfg.span

    # ---------- 用户与设备 ----------
    U = cfg.n_users
    dpu = rng.choice([1, 2, 3], size=U, p=[0.55, 0.30, 0.15])
    dev_user = np.repeat(np.arange(U, dtype=np.int32), dpu)
    D = len(dev_user)
    Du = cfg.n_unknown_devices
    n_dev = D + Du
    devices = pl.DataFrame({
        "device_idx": np.arange(n_dev, dtype=np.int32),
        "user_idx": np.concatenate([dev_user, np.full(Du, -1, dtype=np.int32)]),
        "device_type": rng.choice(["手机", "电脑", "平板"], n_dev, p=[0.6, 0.3, 0.1]),
    })

    # ---------- 曝光 ----------
    N = cfg.n_impressions
    ch_idx = rng.choice(len(CHANNEL_NAMES), N, p=CHANNEL_MIX)
    ch_arr = np.array(CHANNEL_NAMES, dtype="U8")[ch_idx]
    maxc = max(len(v) for v in CAMPAIGNS.values())
    camp_matrix = np.array(
        [CAMPAIGNS[c] + [CAMPAIGNS[c][0]] * (maxc - len(CAMPAIGNS[c])) for c in CHANNEL_NAMES],
        dtype="U8",
    )
    camp_arr = camp_matrix[ch_idx, rng.integers(0, maxc, N)]
    impressions = pl.DataFrame({
        "imp_id": np.arange(N, dtype=np.int64),
        "device_idx": rng.integers(0, n_dev, N, dtype=np.int32),
        "ts": t0 + rng.integers(0, span, N, dtype=np.int64),
        "channel": ch_arr,
        "campaign": camp_arr,
    })

    # ---------- 点击(含渠道重复上报) ----------
    ctr = np.array(CHANNEL_CTR)[ch_idx]
    is_click = rng.random(N) < ctr
    clk_imp = impressions.filter(pl.Series("m", is_click)).drop("imp_id")
    M = clk_imp.height
    clicks = clk_imp.with_columns([
        pl.Series("click_id", np.arange(M, dtype=np.int64)),
        pl.Series("event_id", np.arange(M, dtype=np.int64)),
        pl.col("ts") + pl.Series("d", rng.integers(0, 1800, M, dtype=np.int64)),
    ])
    n_dup = int(M * cfg.dup_click_rate)
    dup_idx = rng.choice(M, n_dup, replace=False)
    dups = clicks.filter(pl.Series("d", np.isin(np.arange(M), dup_idx))).with_columns([
        pl.Series("event_id", M + np.arange(n_dup, dtype=np.int64)),
        pl.col("ts") + pl.Series("d2", rng.integers(1, 120, n_dup, dtype=np.int64)),
    ])
    clicks_raw = pl.concat([clicks, dups]).sort("event_id")

    # ---------- 登录事件(设备 <-> 用户 关联边) ----------
    has_login = rng.random(D) < cfg.login_rate
    logins = pl.DataFrame({
        "device_idx": np.flatnonzero(has_login).astype(np.int32),
        "user_idx": dev_user[has_login],
        "ts": t0 + rng.integers(0, span, int(has_login.sum()), dtype=np.int64),
    })

    # ---------- 下单(真实因果模型) ----------
    imp_u = (impressions.join(devices.select("device_idx", "user_idx"), on="device_idx")
             .filter(pl.col("user_idx") >= 0)
             .select("user_idx", "ts", "channel"))
    counts = imp_u.group_by("user_idx", "channel").len()
    piv = counts.pivot(values="len", index="user_idx", on="channel").fill_null(0)
    piv = piv.select(["user_idx"] + [c for c in CHANNEL_NAMES if c in piv.columns])
    users_with_imp = piv["user_idx"].to_numpy()
    mat = piv.select([c for c in piv.columns if c != "user_idx"]).to_numpy().astype(np.float64)
    eff = np.array([CHANNEL_EFFECT[CHANNEL_NAMES.index(c)] for c in piv.columns if c != "user_idx"])
    raw = mat @ eff  # Σ effect_c * n_c

    # 反解基准 logit, 使期望转化率 = target_convert_rate
    lo, hi = -12.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _sigmoid(0.55 * raw + mid).mean() < cfg.target_convert_rate:
            lo = mid
        else:
            hi = mid
    base = mid
    p_conv = _sigmoid(0.55 * raw + base)
    conv_users = users_with_imp[rng.random(len(users_with_imp)) < p_conv]

    # 订单时间锚定在该用户某次真实曝光之后 0~2 小时, 保证窗口内通常有触点
    anchors = (imp_u.with_columns(pl.Series("r", rng.random(imp_u.height)))
               .sort("r")
               .group_by("user_idx", maintain_order=True)
               .agg(pl.col("ts").first().alias("anchor_ts")))
    orders = (pl.DataFrame({"user_idx": conv_users})
              .join(anchors, on="user_idx", how="left")
              .with_columns([
                  (pl.col("anchor_ts") + pl.Series("d", rng.integers(0, 7200, len(conv_users), dtype=np.int64))).alias("ts"),
                  pl.Series("amount", np.round(np.exp(rng.normal(np.log(180), 0.8, len(conv_users))), 2)),
              ])
              .drop("anchor_ts")
              .filter(pl.col("ts") < t0 + span))
    # 复购
    n2 = int(orders.height * cfg.second_order_rate)
    second = orders.sample(n=n2, seed=cfg.seed).with_columns(
        pl.col("ts") + pl.Series("d", rng.integers(86400, 10 * 86400, n2, dtype=np.int64)),
        pl.Series("amount", np.round(np.exp(rng.normal(np.log(180), 0.8, n2)), 2)),
    ).filter(pl.col("ts") < t0 + span)
    orders = (pl.concat([orders, second])
              .sort("ts")
              .with_row_index("order_id")
              .select("order_id", "user_idx", "ts", "amount"))

    # ---------- 退款 ----------
    n_ref = int(orders.height * cfg.refund_rate)
    ref = orders.sample(n=n_ref, seed=cfg.seed + 1)
    frac = np.where(rng.random(n_ref) < 0.8, 1.0, 0.5)
    refunds = ref.select([
        "order_id",
        (pl.col("amount") * pl.Series("f", frac)).round(2).alias("refund_amount"),
        (pl.col("ts") + pl.Series("d", rng.integers(86400, 8 * 86400, n_ref, dtype=np.int64))).alias("refund_ts"),
    ])

    # ---------- 落盘 ----------
    frames = {
        "impressions": impressions, "clicks_raw": clicks_raw, "orders": orders,
        "refunds": refunds, "logins": logins, "devices": devices,
    }
    for name, df in frames.items():
        df.write_parquet(cfg.data_dir / f"{name}.parquet", compression="zstd")

    truth = {
        "channel_effect": dict(zip(CHANNEL_NAMES, CHANNEL_EFFECT)),
        "base_logit": round(float(base), 4),
        "model": "logit = base + 0.55 * Σ_c effect_c * n_c;  n_c = 用户全周期渠道触点数",
    }
    (cfg.data_dir / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "seconds": round(time.perf_counter() - t_start, 2),
        "rows": {k: v.height for k, v in frames.items()},
        "base_logit": round(float(base), 4),
    }
    (cfg.data_dir / "gen_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
