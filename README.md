# 广告归因系统

自建多渠道归因:把曝光、点击、下单三份日志按统一身份和时间串起来,五种归因模型并排对比,并与数据生成时内置的真实因果设定对照。

## 运行

```bash
py -m attribution.run                    # 全流程(数据已存在则复用), 一千万曝光
py -m attribution.run --regen            # 强制重新生成日志
py -m attribution.run --window-days 14   # 改回溯窗口(默认 7 天)
py -m attribution.run --half-life-days 5 # 改时间衰减半衰期(默认 3 天)
py -m attribution.run --impressions 1000000 --users 200000 --regen  # 小规模试跑
```

结果在 `web/index.html`(自包含,浏览器直接打开),聚合数据在 `web/dashboard_data.json`。

## 模块

| 文件 | 职责 |
|---|---|
| `attribution/config.py` | 渠道设定(曝光占比/CTR/**真实因果效应**)、规模、归因参数 |
| `attribution/generate.py` | 生成曝光/点击/下单/退款/登录日志;转化 logit = base + Σ 渠道效应×触点数 |
| `attribution/identity.py` | 登录事件构成 设备↔用户 二分图,并查集合并多设备为统一身份 |
| `attribution/models.py` | 末次点击 / 首次点击 / 线性 / 时间衰减 / 位置加权 + 真实份额 |
| `attribution/pipeline.py` | 点击去重(同 click_id)→ 窗口串联 → 归因 → 退款冲回 → 路径/报表 |
| `attribution/export_web.py` | 渲染看板 |

## 口径说明

- **回溯窗口**:触点 ≤ 下单时间 且 间隔 ≤ `--window-days`,曝光与点击都计入(末次/首次点击模型只认点击)。
- **点击去重**:同一 `click_id` 被渠道重复上报,保留最早一条。
- **退款冲回**:净归因 = 毛归因 × (1 − 退款金额/订单金额),全额退款订单归因归零。
- **渠道自报**:窗口内该渠道有触点即全量认领订单(模拟平台口径),因此各家相加远超实际。
- **真实份额**:按生成设定中各渠道因果效应对窗口内触点加权归一;偏差单位是百分点(pp)。
- **未归因/自然**:窗口内无触点(或点击模型下无点击)的订单。
