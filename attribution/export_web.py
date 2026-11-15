"""把流水线结果写进 web/dashboard_data.json,并渲染 web/index.html。"""
from __future__ import annotations

import json

from .config import Config


def export(result: dict, cfg: Config) -> None:
    cfg.web_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    (cfg.web_dir / "dashboard_data.json").write_text(payload, encoding="utf-8")
    tpl = (cfg.web_dir / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/", payload)
    (cfg.web_dir / "index.html").write_text(html, encoding="utf-8")
