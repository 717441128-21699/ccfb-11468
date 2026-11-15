"""五种归因模型 + 真实份额,输入统一为 order_touches(订单 × 窗口内触点)。"""
from __future__ import annotations

import polars as pl


def credit_last_click(ot: pl.DataFrame) -> pl.DataFrame:
    """末次点击:窗口内最后一次点击独得;无点击的订单不计入(未归因)。"""
    c = ot.filter(pl.col("touch_type") == "点击")
    c = c.filter(pl.col("ts") == pl.col("ts").max().over("order_id"))
    return (c.unique(subset=["order_id"], keep="first")
             .select("order_id", "channel", "campaign", pl.lit(1.0).alias("w")))


def credit_first_click(ot: pl.DataFrame) -> pl.DataFrame:
    """首次点击:窗口内第一次点击独得;无点击的订单不计入(未归因)。"""
    c = ot.filter(pl.col("touch_type") == "点击")
    c = c.filter(pl.col("ts") == pl.col("ts").min().over("order_id"))
    return (c.unique(subset=["order_id"], keep="first")
             .select("order_id", "channel", "campaign", pl.lit(1.0).alias("w")))


def credit_linear(ot: pl.DataFrame) -> pl.DataFrame:
    """线性:窗口内全部触点(曝光+点击)均分。"""
    return (ot.with_columns(w=1.0 / pl.len().over("order_id"))
              .select("order_id", "channel", "campaign", "w"))


def credit_time_decay(ot: pl.DataFrame, half_life_days: float) -> pl.DataFrame:
    """时间衰减:权重 = 0.5^(距转化天数/半衰期),按订单归一化。"""
    w0 = pl.lit(0.5) ** (pl.col("dt_days") / half_life_days)
    return (ot.with_columns(_w0=w0)
              .with_columns(w=pl.col("_w0") / pl.col("_w0").sum().over("order_id"))
              .select("order_id", "channel", "campaign", "w"))


def credit_position(ot: pl.DataFrame) -> pl.DataFrame:
    """位置加权:首末各 40%,中间均分 20%;单触点 100%,双触点各 50%。"""
    n = pl.len().over("order_id")
    rank = pl.col("ts").rank("ordinal").over("order_id")
    w = (pl.when(n == 1).then(1.0)
           .when(n == 2).then(0.5)
           .when((rank == 1) | (rank == n)).then(0.4)
           .otherwise(0.2 / (n - 2)))
    return (ot.with_columns(w=w)
              .select("order_id", "channel", "campaign", "w"))


def credit_truth(ot: pl.DataFrame, effect: dict[str, float]) -> pl.DataFrame:
    """真实份额:按生成设定中各渠道因果效应加权,按订单归一化。"""
    eff = pl.DataFrame({"channel": list(effect), "eff": list(effect.values())})
    return (ot.join(eff, on="channel")
              .with_columns(w=pl.col("eff") / pl.col("eff").sum().over("order_id"))
              .select("order_id", "channel", "campaign", "w"))
