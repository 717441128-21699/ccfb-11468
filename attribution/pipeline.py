"""归因流水线:数据接入 -> 身份合并 -> 点击去重 -> 窗口串联 -> 五模型归因
-> 退款冲回 -> 路径分析 -> 渠道报表 -> 与真实因果设定对照。"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import polars as pl

from . import models
from .config import CHANNEL_EFFECT, CHANNEL_NAMES, MODEL_NAMES, Config
from .identity import build_identity


@contextmanager
def _step(timings: dict, name: str):
    t = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - t, 2)


def _date_str(t0: int, day: int) -> str:
    return datetime.fromtimestamp(t0 + day * 86400, tz=timezone.utc).strftime("%m-%d")


def run(cfg: Config) -> dict:
    timings: dict[str, float] = {}
    d = cfg.data_dir

    with _step(timings, "读取日志"):
        impressions = pl.read_parquet(d / "impressions.parquet")
        clicks_raw = pl.read_parquet(d / "clicks_raw.parquet")
        orders = pl.read_parquet(d / "orders.parquet")
        refunds = pl.read_parquet(d / "refunds.parquet")
        logins = pl.read_parquet(d / "logins.parquet")
        devices = pl.read_parquet(d / "devices.parquet")

    # ---------- 用户识别:设备图谱合并 ----------
    with _step(timings, "身份图谱合并"):
        ident = build_identity(devices, logins)
        device_uid = ident["device_uid"].select("device_idx", "uid")
        user_uid = ident["user_uid"]

    # ---------- 点击去重(同一 click_id 渠道重复上报,保留最早一条) ----------
    with _step(timings, "点击去重"):
        clicks = clicks_raw.sort("ts").unique(subset=["click_id"], keep="first")
        n_click_raw, n_click = clicks_raw.height, clicks.height

    # ---------- 统一触点流(曝光+点击, 挂到统一身份) ----------
    with _step(timings, "触点归一化"):
        imp_t = (impressions.join(device_uid, on="device_idx")
                 .select("uid", "ts", "channel", "campaign")
                 .with_columns(pl.lit("曝光").alias("touch_type")))
        clk_t = (clicks.join(device_uid, on="device_idx")
                 .select("uid", "ts", "channel", "campaign")
                 .with_columns(pl.lit("点击").alias("touch_type")))
        touches = pl.concat([imp_t, clk_t])
        orders_u = orders.join(user_uid, on="user_idx").rename({"ts": "order_ts"})

    # ---------- 曝光/点击/下单 按用户+时间串联(回溯窗口可配) ----------
    with _step(timings, "窗口串联"):
        window_s = cfg.window_days * 86400
        ot = (orders_u.join(touches, on="uid")
              .filter((pl.col("ts") <= pl.col("order_ts"))
                      & (pl.col("order_ts") - pl.col("ts") <= window_s))
              .with_columns(dt_days=(pl.col("order_ts") - pl.col("ts")) / 86400.0)
              .select("order_id", "order_ts", "amount", "ts", "channel",
                      "campaign", "touch_type", "dt_days"))

    # ---------- 退款冲回系数 ----------
    with _step(timings, "退款冲回"):
        ref = (refunds.group_by("order_id")
               .agg(pl.col("refund_amount").sum()))
        orders_u = (orders_u.join(ref, on="order_id", how="left")
                    .with_columns(pl.col("refund_amount").fill_null(0.0))
                    .with_columns(netf=(1.0 - pl.col("refund_amount") / pl.col("amount")).clip(0.0, 1.0))
                    .with_columns(day=(pl.col("order_ts") - cfg.t0) // 86400))
        order_dim = orders_u.select("order_id", "amount", "netf", "day")

    # ---------- 五种模型 + 真实份额 ----------
    with _step(timings, "归因计算"):
        effect = dict(zip(CHANNEL_NAMES, CHANNEL_EFFECT))
        credit_frames = {
            "末次点击": models.credit_last_click(ot),
            "首次点击": models.credit_first_click(ot),
            "线性": models.credit_linear(ot),
            "时间衰减": models.credit_time_decay(ot, cfg.half_life_days),
            "位置加权": models.credit_position(ot),
        }
        truth_cr = models.credit_truth(ot, effect)
        credits = pl.concat(
            [df.with_columns(pl.lit(m).alias("model")) for m, df in credit_frames.items()]
            + [truth_cr.with_columns(pl.lit("真实份额").alias("model"))]
        ).join(order_dim, on="order_id")
        credits = credits.with_columns([
            (pl.col("w") * pl.col("amount")).alias("gross_rev"),
            (pl.col("w") * pl.col("netf")).alias("net_conv"),
            (pl.col("w") * pl.col("amount") * pl.col("netf")).alias("net_rev"),
        ]).rename({"w": "gross_conv"})

    # ---------- 汇总:模型 × 渠道 ----------
    with _step(timings, "报表汇总"):
        agg_cols = [
            pl.col("gross_conv").sum(), pl.col("gross_rev").sum(),
            pl.col("net_conv").sum(), pl.col("net_rev").sum(),
        ]
        by_model_ch = credits.group_by("model", "channel").agg(agg_cols)
        by_model_ch_camp = credits.group_by("model", "channel", "campaign").agg(agg_cols)
        by_day_model_ch = credits.group_by("model", "day", "channel").agg(agg_cols)

        # 渠道自报(平台口径:窗口内有其触点即全量认领该订单)
        claimed = (ot.unique(subset=["order_id", "channel"])
                   .join(order_dim, on="order_id")
                   .group_by("channel")
                   .agg(pl.len().alias("self_conv"),
                        pl.col("amount").sum().alias("self_rev"),
                        (pl.col("netf")).sum().alias("self_conv_net")))

        # 触点侧按天/渠道/活动
        touch_daily = (touches.with_columns(day=(pl.col("ts") - cfg.t0) // 86400)
                       .group_by("day", "channel", "touch_type").len())
        touch_camp = (touches.group_by("channel", "campaign", "touch_type").len())
        order_daily = (orders_u.group_by("day")
                       .agg(pl.len().alias("orders"), pl.col("amount").sum().alias("gmv"),
                            pl.col("netf").sum().alias("net_orders")))

    # ---------- 路径分析 ----------
    with _step(timings, "路径分析"):
        seqs = (ot.sort("order_id", "ts")
                .group_by("order_id")
                .agg([pl.col("channel").alias("seq"),
                      pl.len().alias("touches"),
                      (pl.col("touch_type") == "点击").sum().alias("clicks"),
                      ((pl.col("ts").max() - pl.col("ts").min()) / 3600.0).alias("span_h")]))
        path_rows: dict[str, list] = {}
        len_dist = np.zeros(11, dtype=np.int64)
        n_orders_with_touch = seqs.height
        tot_touches = tot_clicks = 0
        tot_span = 0.0
        single_touch = 0
        for seq, nt, nc, sp in zip(seqs["seq"].to_list(), seqs["touches"].to_list(),
                                   seqs["clicks"].to_list(), seqs["span_h"].to_list()):
            comp = [seq[0]] if seq else []
            for c in seq[1:]:
                if c != comp[-1]:
                    comp.append(c)
            key = " > ".join(comp)
            path_rows[key] = [path_rows.get(key, [0])[0] + 1]
            tot_touches += nt
            tot_clicks += nc
            tot_span += sp
            single_touch += int(nt == 1)
            len_dist[min(nt, 10)] += 1
        top_paths = sorted(path_rows.items(), key=lambda kv: -kv[1][0])[:20]
        path_stats = {
            "orders_with_touch": n_orders_with_touch,
            "avg_touches": round(tot_touches / max(n_orders_with_touch, 1), 2),
            "avg_clicks": round(tot_clicks / max(n_orders_with_touch, 1), 2),
            "avg_span_h": round(tot_span / max(n_orders_with_touch, 1), 1),
            "single_touch_pct": round(100 * single_touch / max(n_orders_with_touch, 1), 1),
            "len_dist": len_dist.tolist(),
        }

    # ---------- 组装结果 ----------
    with _step(timings, "组装输出"):
        result = _assemble(cfg, timings, ident["stats"], impressions, clicks_raw,
                           clicks, orders_u, refunds, touches, by_model_ch,
                           by_model_ch_camp, by_day_model_ch, claimed, touch_daily,
                           touch_camp, order_daily, top_paths, path_stats)
    return result


def _assemble(cfg, timings, ident_stats, impressions, clicks_raw, clicks, orders_u,
              refunds, touches, by_model_ch, by_model_ch_camp, by_day_model_ch,
              claimed, touch_daily, touch_camp, order_daily, top_paths, path_stats):
    t0 = cfg.t0
    n_orders = orders_u.height
    total_net = float(orders_u["netf"].sum())

    # 模型 × 渠道 透视
    def _pivot(df, val):
        out = {}
        for row in df.iter_rows(named=True):
            out.setdefault(row["model"], {})[row["channel"]] = row[val]
        return out

    net_conv = _pivot(by_model_ch, "net_conv")
    gross_conv = _pivot(by_model_ch, "gross_conv")
    net_rev = _pivot(by_model_ch, "net_rev")

    all_models = MODEL_NAMES + ["真实份额"]
    shares, deviations = {}, {}
    for m in all_models:
        row = net_conv.get(m, {})
        tot = sum(row.values())
        unattrib = max(total_net - tot, 0.0)
        sh = {c: 100.0 * row.get(c, 0.0) / total_net for c in CHANNEL_NAMES}
        sh["未归因/自然"] = 100.0 * unattrib / total_net
        shares[m] = sh
    for m in MODEL_NAMES:
        dev = {c: round(shares[m][c] - shares["真实份额"][c], 2) for c in CHANNEL_NAMES}
        dev["未归因/自然"] = round(shares[m]["未归因/自然"] - shares["真实份额"]["未归因/自然"], 2)
        deviations[m] = dev
    mae = {m: round(float(np.mean([abs(deviations[m][c]) for c in CHANNEL_NAMES])), 3)
           for m in MODEL_NAMES}
    max_dev = {m: round(max(abs(deviations[m][c]) for c in CHANNEL_NAMES), 2)
               for m in MODEL_NAMES}

    # 每模型 总量:毛/冲回/净
    model_totals = {}
    for m in MODEL_NAMES:
        g = sum(gross_conv.get(m, {}).values())
        n = sum(net_conv.get(m, {}).values())
        model_totals[m] = {"gross": round(g, 1), "reversed": round(g - n, 1), "net": round(n, 1)}
    model_totals["真实份额"] = {
        "gross": round(sum(gross_conv.get("真实份额", {}).values()), 1),
        "reversed": round(sum(gross_conv.get("真实份额", {}).values()) - sum(net_conv.get("真实份额", {}).values()), 1),
        "net": round(sum(net_conv.get("真实份额", {}).values()), 1),
    }

    # 渠道报表(按渠道):触点 + 自报 + 各模型净转化
    self_rep = {r["channel"]: r for r in claimed.iter_rows(named=True)}
    ch_touch = {}
    for row in touch_camp.iter_rows(named=True):
        e = ch_touch.setdefault(row["channel"], {"曝光": 0, "点击": 0})
        e[row["touch_type"]] += row["len"]
    lc_net = net_conv.get("末次点击", {})
    truth_net = net_conv.get("真实份额", {})
    channel_table = []
    for c in CHANNEL_NAMES:
        imps = ch_touch.get(c, {}).get("曝光", 0)
        clks = ch_touch.get(c, {}).get("点击", 0)
        sr = self_rep.get(c, {})
        lc = lc_net.get(c, 0.0)
        channel_table.append({
            "channel": c,
            "impressions": int(imps),
            "clicks": int(clks),
            "ctr": round(100 * clks / max(imps, 1), 2),
            "self_conv": int(sr.get("self_conv", 0)),
            "self_conv_net": round(float(sr.get("self_conv_net", 0.0)), 1),
            "attr_conv": {m: round(net_conv.get(m, {}).get(c, 0.0), 1) for m in MODEL_NAMES},
            "truth_conv": round(truth_net.get(c, 0.0), 1),
            "net_rev": round(net_rev.get("末次点击", {}).get(c, 0.0), 0),
            "self_over_attr": round(float(sr.get("self_conv_net", 0.0)) / max(lc, 1e-9), 2),
        })
    self_total = float(sum(float(r.get("self_conv_net", 0.0)) for r in self_rep.values()))

    # 活动级报表
    camp_touch: dict = {}
    for row in touch_camp.iter_rows(named=True):
        e = camp_touch.setdefault((row["channel"], row["campaign"]), {"曝光": 0, "点击": 0})
        e[row["touch_type"]] += row["len"]
    camp_attr: dict = {}
    for row in by_model_ch_camp.iter_rows(named=True):
        camp_attr.setdefault((row["channel"], row["campaign"]), {})[row["model"]] = row["net_conv"]
    campaign_table = []
    for (ch, cp), t in sorted(camp_touch.items()):
        campaign_table.append({
            "channel": ch, "campaign": cp,
            "impressions": int(t["曝光"]), "clicks": int(t["点击"]),
            "attr_conv": {m: round(camp_attr.get((ch, cp), {}).get(m, 0.0), 1) for m in MODEL_NAMES},
        })

    # 按天:触点 + 订单 + 各模型净转化
    days = list(range(cfg.days))
    daily_touch = {(r["day"], r["channel"], r["touch_type"]): r["len"]
                   for r in touch_daily.iter_rows(named=True)}
    daily_order = {r["day"]: r for r in order_daily.iter_rows(named=True)}
    daily_attr: dict = {}
    for row in by_day_model_ch.iter_rows(named=True):
        daily_attr.setdefault((row["model"], row["channel"]), {})[row["day"]] = row["net_conv"]
    daily = []
    for day in days:
        rec = {"date": _date_str(t0, day),
               "impressions": sum(daily_touch.get((day, c, "曝光"), 0) for c in CHANNEL_NAMES),
               "clicks": sum(daily_touch.get((day, c, "点击"), 0) for c in CHANNEL_NAMES),
               "orders": int(daily_order.get(day, {}).get("orders", 0)),
               "by_channel": {}}
        for c in CHANNEL_NAMES:
            rec["by_channel"][c] = {
                "impressions": int(daily_touch.get((day, c, "曝光"), 0)),
                "clicks": int(daily_touch.get((day, c, "点击"), 0)),
                "attr": {m: round(daily_attr.get((m, c), {}).get(day, 0.0), 1) for m in MODEL_NAMES},
            }
        daily.append(rec)

    # 退款
    n_refund_orders = refunds["order_id"].n_unique()
    refund_stats = {
        "orders": n_orders,
        "refunded_orders": int(n_refund_orders),
        "refund_rate": round(100 * n_refund_orders / max(n_orders, 1), 1),
        "refund_amount": round(float(refunds["refund_amount"].sum()), 0),
        "gmv": round(float(orders_u["amount"].sum()), 0),
    }

    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "window_days": cfg.window_days,
            "half_life_days": cfg.half_life_days,
            "days": cfg.days,
            "seed": cfg.seed,
            "timings": timings,
            "pipeline_seconds": round(sum(timings.values()), 2),
            "channels": CHANNEL_NAMES,
            "models": MODEL_NAMES,
            "channel_effect": dict(zip(CHANNEL_NAMES, CHANNEL_EFFECT)),
        },
        "ingest": {
            "impressions": impressions.height,
            "clicks_raw": clicks_raw.height,
            "clicks_dedup": clicks.height,
            "clicks_dup_removed": clicks_raw.height - clicks.height,
            "orders": n_orders,
            "refunds": refunds.height,
            "logins": int(ident_stats["login_links"]),
            "devices": int(ident_stats["devices"]),
            "users": int(ident_stats["users"]),
            "touches": touches.height,
        },
        "identity": ident_stats,
        "refunds": refund_stats,
        "models": {
            "shares": {m: {k: round(v, 2) for k, v in shares[m].items()} for m in all_models},
            "deviations": deviations,
            "mae": mae,
            "max_dev": max_dev,
            "totals": model_totals,
            "total_net_orders": round(total_net, 1),
        },
        "report": {
            "channels": channel_table,
            "campaigns": campaign_table,
            "daily": daily,
            "self_report_total": round(self_total, 1),
            "self_over_actual": round(self_total / max(total_net, 1e-9), 2),
        },
        "paths": {
            "top": [{"path": k, "count": v[0],
                     "pct": round(100 * v[0] / max(path_stats["orders_with_touch"], 1), 2)}
                    for k, v in top_paths],
            **path_stats,
        },
    }
    return result
