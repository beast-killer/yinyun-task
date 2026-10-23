#!/usr/bin/env python
"""复现 README 中的全部数字。

用法:
    python repro.py              # 打印所有段落
    python repro.py --section 3  # 只跑某一段

三个口径常量决定整份分析的结论, 改它们等于换一个题目:
    FAIL_COST_H     一次现场维修失败损失的工时
    INTERCEPT_COST_H 提前拦截重派一次的工时成本
    LOAD_HI / LOAD_LO 积压工时的分档边界
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path(__file__).parent / "data.csv"

# ============================================================ 口径常量
FAIL_COST_H = 75.0        # 题干给定: 一次失败约损失 75 小时
INTERCEPT_COST_H = 15.0   # 题干给定: 每提前拦截重派一次约 15 小时
LOAD_LO = 40              # 低负荷上界
LOAD_HI = 60              # 高负荷下界(>= 此值进入高频失败区)

# ============================================================ 载入
def load(path: Path = DATA_PATH) -> pd.DataFrame:
    """读原始数据 -> 去重 -> 统一时区 -> 派生字段。

    两个必须做的处理, 不做结论就错:
      1. 53 行完全重复(同一 ticket 被重复派工记录), 不去重会把失败率摊薄。
      2. dispatch_time 带 +08:00, manual_synced_at 带 Z(UTC)。直接相减会凭空多出 8 小时,
         把"手册同步及时"误判成"手册滞后"。
    """
    if not path.exists():
        raise FileNotFoundError(f"缺少数据文件: {path}")
    raw = pd.read_csv(path)
    df = raw.drop_duplicates().copy()

    # 两个时间列时区不同, 必须都转成带时区的 UTC 再相减
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], utc=True)
    df["manual_synced_at"] = pd.to_datetime(df["manual_synced_at"], utc=True)
    df["manual_lag_h"] = (
        df["dispatch_time"] - df["manual_synced_at"]
    ).dt.total_seconds() / 3600

    # is_resolved: 1 成功 / 0 失败 / 空 未回传。空值不能当失败, 也不能当成功, 只能剔除
    df["is_fail"] = np.where(df["is_resolved"] == 0, 1.0, np.where(df["is_resolved"] == 1, 0.0, np.nan))
    df["load_band"] = pd.cut(
        df["labor_hours"], [-1, LOAD_LO, LOAD_HI, 10**9],
        labels=[f"<{LOAD_LO}h", f"{LOAD_LO}-{LOAD_HI}h", f">={LOAD_HI}h"], right=False,
    )
    return df


def valid(df: pd.DataFrame) -> pd.DataFrame:
    """只保留结果已回传的工单。所有失败率的分母都是它。"""
    return df[df["is_fail"].notna()].copy()


# ============================================================ 0 数据概况
def sec_data(df: pd.DataFrame, raw_rows: int) -> None:
    print("=" * 72)
    print("[0] 数据概况")
    print(f"  原始行数            : {raw_rows:,}")
    print(f"  完全重复行          : {raw_rows - len(df):,}  (已剔除)")
    print(f"  去重后工单          : {len(df):,}   ticket_id 唯一: {df['ticket_id'].nunique():,}")
    v = valid(df)
    print(f"  结果已回传          : {len(v):,}   未回传: {len(df) - len(v):,} (不计入失败率分母)")
    print(f"  时间覆盖            : {df['dispatch_time'].min():%Y-%m-%d} ~ {df['dispatch_time'].max():%Y-%m-%d}")
    print(f"  总体失败率          : {v['is_fail'].mean()*100:.2f}%  ({int(v['is_fail'].sum())}/{len(v)})")


# ============================================================ 1 排除方案失效
def sec_manual(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("[1] 先排除'排查手册失效'这个假设")
    lag = df["manual_lag_h"]
    print(f"  手册同步滞后(小时)  : 中位数 {lag.median():.2f}  P95 {lag.quantile(.95):.2f}  最大 {lag.max():.2f}")
    print(f"  滞后 > 24h 的工单   : {(lag > 24).sum()}")
    print(f"  滞后为负(手册比派工晚) : {(lag < 0).sum()}")
    print("  结论: 派工时手册与备件目录最长只滞后 1.17 小时, 系统同步实质是实时的。")
    print("        题干假设的'方案信息失效'在这份数据里不存在 —— 这条线索可以关闭。")


# ============================================================ 2 积压工时
def sec_load(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("[2] 真正的致败因子: 派工时段积压工时")
    v = valid(df)
    t = v.groupby("load_band", observed=True).agg(
        n=("is_fail", "size"), fails=("is_fail", "sum"), rate=("is_fail", "mean")
    )
    t["rate_pct"] = (t["rate"] * 100).round(1)
    print(t[["n", "fails", "rate_pct"]].to_string())

    lo = t.loc[f"<{LOAD_LO}h", "rate"]
    hi = t.loc[f">={LOAD_HI}h", "rate"]
    print(f"\n  高负荷 vs 低负荷: {hi*100:.1f}% vs {lo*100:.1f}%  = {hi/lo:.1f} 倍")

    ct = pd.crosstab(v["load_band"], v["is_fail"])
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    print(f"  卡方检验: chi2={chi2:.1f}, dof={dof}, p={p:.3g}")
    print(f"  效应量  : Cramer's V = {cramers_v:.3f}")

    rho, p_s = stats.spearmanr(v["labor_hours"], v["is_fail"])
    print(f"  单调性  : Spearman rho={rho:.3f}, p={p_s:.3g}  (积压越高失败越多)")

    # 分层: 每个网点内部方向是否一致, 排除辛普森悖论
    print("\n  分层复核(各网点内部, 高负荷 vs 低负荷失败率):")
    rows = []
    for sc, g in v.groupby("service_center"):
        g_lo = g[g["labor_hours"] < LOAD_LO]["is_fail"]
        g_hi = g[g["labor_hours"] >= LOAD_HI]["is_fail"]
        if len(g_lo) >= 5 and len(g_hi) >= 5:
            rows.append([sc, len(g_lo), g_lo.mean() * 100, len(g_hi), g_hi.mean() * 100,
                         (g_hi.mean() - g_lo.mean()) * 100])
    layer = pd.DataFrame(rows, columns=["网点", "n_低", "低负荷%", "n_高", "高负荷%", "差值pp"])
    print(layer.round(1).to_string(index=False))
    print(f"  5 个网点中方向一致的有 {int((layer['差值pp'] > 0).sum())}/5")


# ============================================================ 3 备件
def sec_parts(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("[3] 备件可用量: 独立但次要的第二条路径")
    v = valid(df)
    v = v.assign(part_gap=pd.cut(v["parts_available"], [-1, 0, 2, 10**9],
                                 labels=["0 缺货", "1-2 紧张", ">2 充足"]))
    t = v.groupby("part_gap", observed=True).agg(
        n=("is_fail", "size"), rate=("is_fail", "mean")
    )
    t["rate_pct"] = (t["rate"] * 100).round(1)
    print(t[["n", "rate_pct"]].to_string())
    print(f"\n  缺货 vs 充足: {t.loc['0 缺货','rate']*100:.1f}% vs {t.loc['>2 充足','rate']*100:.1f}%")
    print(f"  但缺货只占 {t.loc['0 缺货','n']/len(v)*100:.1f}% 的工单, 影响面有限")


# ============================================================ 4 ROI
def sec_roi(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("[4] 算账: '提前拦截重派'这门生意亏不亏")
    v = valid(df)
    breakeven = INTERCEPT_COST_H / FAIL_COST_H * 100
    print(f"  盈亏平衡: 拦截一单花 {INTERCEPT_COST_H:.0f}h, 拦对一次省 {FAIL_COST_H:.0f}h")
    print(f"            => 失败率必须 > {breakeven:.1f}% 才划算")
    print(f"  实际整体失败率: {v['is_fail'].mean()*100:.2f}%  -> "
          f"{'亏' if v['is_fail'].mean()*100 < breakeven else '赚'}")

    t = v.groupby("load_band", observed=True).agg(
        n=("is_fail", "size"), rate=("is_fail", "mean")
    )
    t["每单期望挽回h"] = (t["rate"] * FAIL_COST_H).round(1)
    t["每单净收益h"] = (t["rate"] * FAIL_COST_H - INTERCEPT_COST_H).round(1)
    print("\n  按负荷分档, 每拦截一单的净收益:")
    print(t[["n", "每单期望挽回h", "每单净收益h"]].to_string())

    hi = v[v["labor_hours"] >= LOAD_HI]
    lo_rate = t.loc[f"<{LOAD_LO}h", "rate"]
    avoided = hi["is_fail"].sum() - len(hi) * lo_rate
    cost = len(hi) * INTERCEPT_COST_H
    print(f"\n  场景 A: 拦截全部 >={LOAD_HI}h 工单({len(hi)} 单, 其中失败 {int(hi['is_fail'].sum())} 次)")
    print(f"    避免失败 {avoided:.0f} 次 x {FAIL_COST_H:.0f}h = {avoided*FAIL_COST_H:,.0f}h")
    print(f"    拦截成本 {len(hi)} x {INTERCEPT_COST_H:.0f}h = {cost:,.0f}h")
    print(f"    >>> 净收益 {avoided*FAIL_COST_H - cost:+,.0f}h   (负数 = 比不拦更亏)")

    print(f"\n  场景 B: 不拦截, 直接把积压压到 <{LOAD_LO}h 水平(调度手段, 无按单成本)")
    print(f"    避免失败 {avoided:.0f} 次 = {avoided*FAIL_COST_H:,.0f}h")
    print(f"    >>> 净收益 {avoided*FAIL_COST_H:+,.0f}h")

    narrow = v[(v["labor_hours"] >= LOAD_HI) & (v["parts_available"] <= 2)]
    if len(narrow):
        net = narrow["is_fail"].mean() * FAIL_COST_H - INTERCEPT_COST_H
        print(f"\n  场景 C: 只拦截'高积压 + 缺货'窄口 ({len(narrow)} 单, 失败率 "
              f"{narrow['is_fail'].mean()*100:.1f}%)")
        print(f"    每单净收益 {net:+.1f}h  -> {'可以拦, 但盘子只有 %d 单' % len(narrow) if net > 0 else '仍然不划算'}")


SECTIONS = {"0": sec_data, "1": sec_manual, "2": sec_load, "3": sec_parts, "4": sec_roi}


def main() -> int:
    ap = argparse.ArgumentParser(description="复现 README 全部数字")
    ap.add_argument("--section", choices=sorted(SECTIONS))
    args = ap.parse_args()

    raw_rows = len(pd.read_csv(DATA_PATH))
    df = load()
    for key in sorted(SECTIONS):
        if args.section and key != args.section:
            continue
        fn = SECTIONS[key]
        if key == "0":
            fn(df, raw_rows)
        else:
            fn(df)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
