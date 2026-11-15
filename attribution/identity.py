"""用户识别:用登录事件把 设备ID 和 用户ID 连成图谱,并查集合并成统一身份。"""
from __future__ import annotations

import numpy as np
import polars as pl


def _find(parent: np.ndarray, x: int) -> int:
    r = x
    while parent[r] != r:
        r = parent[r]
    # 路径压缩
    while parent[x] != r:
        parent[x], x = r, parent[x]
    return r


def build_identity(devices: pl.DataFrame, logins: pl.DataFrame) -> dict:
    """节点 = 设备(0..D-1) + 用户(D..D+U-1);登录事件是二分图上的边。"""
    D = devices.height
    U = int(devices["user_idx"].max()) + 1
    parent = np.arange(D + U)

    src = logins["device_idx"].to_numpy()
    dst = D + logins["user_idx"].to_numpy()
    for a, b in zip(src, dst):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[ra] = rb

    # 指针跳跃整体压缩
    roots = parent.copy()
    while True:
        nxt = roots[roots]
        if (nxt == roots).all():
            break
        roots = nxt

    dev_roots = roots[:D]
    user_roots = roots[D:]
    # 压缩成连续身份ID
    all_roots = np.concatenate([dev_roots, user_roots])
    uniq, inv = np.unique(all_roots, return_inverse=True)
    dev_uid = inv[:D].astype(np.int32)
    user_uid = inv[D:].astype(np.int32)

    device_uid = devices.select("device_idx", "user_idx", "device_type").with_columns(
        pl.Series("uid", dev_uid))
    user_uid_df = pl.DataFrame({
        "user_idx": np.arange(U, dtype=np.int32),
        "uid": user_uid,
    })

    # 统计
    dev_per_ident = np.bincount(dev_uid, minlength=len(uniq))
    multi_dev_idents = int((dev_per_ident >= 2).sum())
    known = devices["user_idx"].to_numpy() >= 0
    anon_dev = int((~known).sum())
    login_dev = np.isin(np.arange(D), logins["device_idx"].to_numpy())
    no_login_dev = int((known & ~login_dev).sum())

    stats = {
        "devices": D,
        "users": U,
        "login_links": logins.height,
        "identities": int(len(uniq)),
        "multi_device_identities": multi_dev_idents,
        "anonymous_devices": anon_dev,
        "known_no_login_devices": no_login_dev,
        "devices_per_identity": {
            "1台": int((dev_per_ident == 1).sum()),
            "2台": int((dev_per_ident == 2).sum()),
            "3台及以上": int((dev_per_ident >= 3).sum()),
        },
    }
    return {"device_uid": device_uid, "user_uid": user_uid_df, "stats": stats}
