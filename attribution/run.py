"""入口:生成数据(可跳过) -> 跑归因流水线 -> 导出网页看板。

用法:
  py -m attribution.run                      # 全流程(数据已存在则复用)
  py -m attribution.run --regen              # 重新生成数据
  py -m attribution.run --window-days 14     # 调整回溯窗口
  py -m attribution.run --impressions 2000000 --regen   # 小规模试跑
"""
from __future__ import annotations

import argparse
import json

from .config import Config
from .export_web import export
from .generate import generate
from .pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impressions", type=int, default=10_000_000)
    ap.add_argument("--users", type=int, default=1_000_000)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--half-life-days", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regen", action="store_true", help="强制重新生成日志")
    args = ap.parse_args()

    cfg = Config(n_impressions=args.impressions, n_users=args.users, days=args.days,
                 window_days=args.window_days, half_life_days=args.half_life_days,
                 seed=args.seed)

    meta_file = cfg.data_dir / "gen_meta.json"
    need_gen = args.regen or not meta_file.exists()
    if not need_gen:
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        need_gen = old.get("rows", {}).get("impressions") != cfg.n_impressions
    if need_gen:
        print(f"[1/3] 生成日志 ({cfg.n_impressions:,} 曝光) ...")
        meta = generate(cfg)
        print(f"      完成 {meta['seconds']}s, 行数: {meta['rows']}")
    else:
        print("[1/3] 复用已有日志数据")

    print(f"[2/3] 归因流水线 (回溯窗口 {cfg.window_days} 天) ...")
    result = run(cfg)
    print(f"      完成 {result['meta']['pipeline_seconds']}s, 分步: {result['meta']['timings']}")

    print("[3/3] 导出网页看板 ...")
    export(result, cfg)
    print(f"      打开 {cfg.web_dir / 'index.html'} 查看")

    m = result["models"]
    print("\n各模型与真实份额的 MAE(百分点):")
    for name in sorted(m["mae"], key=m["mae"].get):
        print(f"  {name:<6} MAE {m['mae'][name]:>6}  最大偏差 {m['max_dev'][name]}")


if __name__ == "__main__":
    main()
