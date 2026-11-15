"""全局配置:渠道设定(含真实因果效应)、数据规模、归因参数。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 渠道设定:mix=曝光占比, ctr=点击率, effect=真实因果效应(logit 系数, 每个触点)
CHANNELS: dict[str, dict] = {
    "搜索广告": {"mix": 0.22, "ctr": 0.060, "effect": 0.90, "campaigns": ["品牌词", "通用词"]},
    "信息流":   {"mix": 0.28, "ctr": 0.030, "effect": 0.50, "campaigns": ["信息流A", "信息流B"]},
    "社交媒体": {"mix": 0.18, "ctr": 0.025, "effect": 0.35, "campaigns": ["社交A", "社交B"]},
    "视频广告": {"mix": 0.12, "ctr": 0.015, "effect": 0.25, "campaigns": ["视频前贴片"]},
    "邮件营销": {"mix": 0.08, "ctr": 0.050, "effect": 0.60, "campaigns": ["周刊", "促销"]},
    "联盟营销": {"mix": 0.12, "ctr": 0.020, "effect": 0.15, "campaigns": ["联盟A"]},
}
CHANNEL_NAMES = list(CHANNELS)
CHANNEL_MIX = [CHANNELS[c]["mix"] for c in CHANNEL_NAMES]
CHANNEL_CTR = [CHANNELS[c]["ctr"] for c in CHANNEL_NAMES]
CHANNEL_EFFECT = [CHANNELS[c]["effect"] for c in CHANNEL_NAMES]
CAMPAIGNS = {c: CHANNELS[c]["campaigns"] for c in CHANNEL_NAMES}

MODEL_NAMES = ["末次点击", "首次点击", "线性", "时间衰减", "位置加权"]


@dataclass
class Config:
    seed: int = 42
    days: int = 30
    t0: int = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())
    # 数据规模
    n_impressions: int = 10_000_000
    n_users: int = 1_000_000
    n_unknown_devices: int = 200_000      # 未登录匿名设备
    login_rate: float = 0.90              # 已知设备中出现登录关联的比例
    dup_click_rate: float = 0.04          # 渠道重复上报点击的比例
    target_convert_rate: float = 0.24     # 目标转化率(自动反解基准 logit)
    second_order_rate: float = 0.15       # 复购比例
    refund_rate: float = 0.10             # 退款订单比例(其中 80% 全额, 20% 半额)
    # 归因参数
    window_days: int = 7                  # 回溯窗口(可配)
    half_life_days: float = 3.0           # 时间衰减半衰期(可配)
    # 路径
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    web_dir: Path = field(default_factory=lambda: ROOT / "web")

    @property
    def span(self) -> int:
        return self.days * 86400
