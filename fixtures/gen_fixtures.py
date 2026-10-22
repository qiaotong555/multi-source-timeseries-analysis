# -*- coding: utf-8 -*-
"""生成固定脱敏样例与真值标签。

所有样例由固定随机种子生成，可重复生成且结果一致；注入的质量问题与异常点
均记录真值标签，用于异常检测 Precision/Recall 评估。

用法：
    python fixtures/gen_fixtures.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260927

# ---------------- 天气样例 ----------------
WEATHER_START = "2024-01-01 00:00:00"
WEATHER_HOURS = 17544          # 2 年小时级
WEATHER_STATIONS = ("STA_A", "STA_B")


def _ar1(rng, n, phi, sigma):
    x, out = 0.0, []
    for _ in range(n):
        x = phi * x + rng.normal(0.0, sigma)
        out.append(x)
    return np.array(out)


def gen_weather(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(WEATHER_START, periods=WEATHER_HOURS, freq="h")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    t_year = (idx - idx[0]).days.to_numpy() / 365.25

    frames, labels = [], []
    for s_i, station in enumerate(WEATHER_STATIONS):
        n = WEATHER_HOURS
        phase = 0.0 if s_i == 0 else 0.6
        annual = 12.0 * np.sin(2 * np.pi * (doy - 100) / 365.25 + phase)
        diurnal = 6.0 * np.sin(2 * np.pi * (hour - 14) / 24.0 + phase)
        trend = 0.8 * t_year                      # 每年 +0.8°C 的升温趋势
        noise = _ar1(rng, n, 0.97, 0.35)
        temp = 15.0 + annual + diurnal + trend + noise + (1.5 if s_i == 0 else -1.5)

        humidity = 65.0 - 0.8 * (temp - 15.0) + _ar1(rng, n, 0.95, 2.5)
        humidity = np.clip(humidity, 5.0, 100.0)
        # 气压噪声自回归系数取 0.85：0.98 的强持续性噪声近似随机游走，
        # 会让变化点定位误差达到千点量级，无法作为可判定的真值
        pressure = 1013.0 + 8.0 * np.sin(2 * np.pi * doy / 365.25) + _ar1(rng, n, 0.85, 1.2)
        wind = np.abs(3.0 + _ar1(rng, n, 0.90, 1.1) + 1.2 * np.sin(2 * np.pi * hour / 24.0))
        rain_u = rng.random(n)
        rainfall = np.where(rain_u < 0.94, 0.0,
                            rng.exponential(2.5, n) * (rain_u - 0.93) / 0.07)

        df = pd.DataFrame({
            "timestamp": idx,
            "station": station,
            "temperature": temp,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind,
            "rainfall": rainfall,
        })
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # ---- 注入质量问题 ----
    # 0) 气压持续水平抬升 +18 hPa（非单位变化，用于变化点检测评估）
    # 幅度需显著：变化点判据要求分割后残差平方和下降 >30%，过小跳变不会被采纳
    for station in WEATHER_STATIONS:
        rows = data.index[data["station"] == station][12000:]
        data.loc[rows, "pressure"] = data.loc[rows, "pressure"] + 18.0
        labels.append({"column": "pressure", "timestamp": str(data.loc[rows[0], "timestamp"]),
                       "kind": "level_shift_start", "station": station})

    # 1) 温度异常跳点（每站 6 个）
    for s_i, station in enumerate(WEATHER_STATIONS):
        m = data["station"] == station
        base = data.loc[m, "temperature"].to_numpy()
        sd = float(np.std(base))
        pos = rng.choice(m.sum(), size=6, replace=False)
        rows = data.index[m][pos]
        signs = rng.choice([-1, 1], size=6)
        for r, sg in zip(rows, signs):
            data.loc[r, "temperature"] = data.loc[r, "temperature"] + sg * (7.0 + 2.0 * rng.random()) * sd
            labels.append({"column": "temperature", "timestamp": str(data.loc[r, "timestamp"]),
                           "kind": "anomaly", "station": station})
        # 2) 风速异常跳点（3 个）
        wsd = float(np.std(data.loc[m, "wind_speed"].to_numpy()))
        wrows = data.index[m][rng.choice(m.sum(), size=3, replace=False)]
        for r in wrows:
            data.loc[r, "wind_speed"] = data.loc[r, "wind_speed"] + (8.0 + 2.0 * rng.random()) * wsd
            labels.append({"column": "wind_speed", "timestamp": str(data.loc[r, "timestamp"]),
                           "kind": "anomaly", "station": station})

    # 3) 单位变化：STA_A 温度有 720 小时（30 天）以华氏度记录
    unit_start = 3000
    unit_end = unit_start + 720
    m_a = data["station"] == "STA_A"
    a_rows = data.index[m_a][unit_start:unit_end]
    labels.append({"column": "temperature", "timestamp": str(data.loc[a_rows[0], "timestamp"]),
                   "kind": "unit_change_start", "station": "STA_A"})
    data.loc[a_rows, "temperature"] = data.loc[a_rows, "temperature"] * 1.8 + 32.0

    # 4) 连续缺失：STA_B 温度缺失 72 小时
    m_b = data["station"] == "STA_B"
    miss_rows = data.index[m_b][5000:5072]
    labels.append({"column": "temperature", "timestamp": str(data.loc[miss_rows[0], "timestamp"]),
                   "kind": "missing_gap_start", "station": "STA_B"})
    data.loc[miss_rows, "temperature"] = np.nan

    # 5) 采样不连续：整段删除连续 10 小时（超过 3 倍中位间隔，应被判为缺口）
    gap_rows = []
    for station in WEATHER_STATIONS:
        gap_rows.extend(list(data.index[data["station"] == station][8000:8010]))
        labels.append({"column": "timestamp",
                       "timestamp": str(data.loc[data.index[data["station"] == station][8000], "timestamp"]),
                       "kind": "sampling_gap_start", "station": station})
    data = data.drop(index=gap_rows).reset_index(drop=True)

    # 6) 不规则采样：随机删除 300 行
    drop_idx = rng.choice(len(data), size=300, replace=False)
    data = data.drop(index=data.index[drop_idx]).reset_index(drop=True)

    # 6) 重复时间戳：复制 25 行追加
    dup_rows = data.iloc[8000:8025].copy()
    data = pd.concat([data, dup_rows], ignore_index=True)

    # 7) 乱序：把一段 60 行整体后移
    block = data.iloc[12000:12060].copy()
    rest = pd.concat([data.iloc[:12000], data.iloc[12060:]], ignore_index=True)
    data = pd.concat([rest.iloc[:12500], block, rest.iloc[12500:]], ignore_index=True)
    labels.append({"column": "timestamp", "timestamp": str(block.iloc[0]["timestamp"]),
                   "kind": "out_of_order_block", "station": "—"})

    data = data.sort_values(["station", "timestamp"]).reset_index(drop=True)
    # 打乱输出顺序，检验引擎是否自行排序
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return data, pd.DataFrame(labels)


# ---------------- 股票样例 ----------------
def gen_stock(seed: int = SEED, leak: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + (1 if leak else 0))
    idx = pd.bdate_range("2024-01-02", periods=500)
    n = len(idx)

    lv = np.log(0.011)
    sigma = []
    for _ in range(n):
        lv = lv + 0.03 * (np.log(0.012) - lv) + rng.normal(0.0, 0.08)
        sigma.append(min(max(np.exp(lv), 0.006), 0.028))
    sigma = np.array(sigma)

    rets = 0.00035 + sigma * rng.normal(0.0, 1.0, n)
    # 剔除有限样本的抽样噪声漂移，使已实现漂移等于理论漂移（不改变相关结构）
    rets = rets - rets.mean() + 0.00035
    close = 12.0 * np.exp(np.cumsum(rets))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    volume = 1.2e7 * np.exp(0.5 * np.abs(rets) / sigma + rng.normal(0.0, 0.15, n))

    bench_rets = 0.00025 + 0.009 * rng.normal(0.0, 1.0, n)
    benchmark = 3200.0 * np.exp(np.cumsum(bench_rets))

    df = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"),
        "symbol": "DEMO.SZ",
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "benchmark": benchmark,
    })

    # 注入：两个收盘价异常跳点（先改收盘价，复权价随后同步重算）
    labels = []
    for pos in (150, 330):
        sd = float(np.std(np.diff(np.log(df["close"].to_numpy()))))
        df.loc[df.index[pos], "close"] = df.loc[df.index[pos], "close"] * (1.0 + 9.0 * sd)
        labels.append({"column": "close", "timestamp": str(df.loc[df.index[pos], "date"]),
                       "kind": "anomaly", "station": "—"})

    # 复权因子：偶发分红/拆股；复权价严格等于 close × adjust_factor
    adjust_factor = np.ones(n)
    for k in (120, 260, 400):
        adjust_factor[k:] *= 0.985
    df["adjust_factor"] = adjust_factor
    df["adj_close"] = df["close"] * df["adjust_factor"]

    # 合法信号：20 日动量（仅用历史）
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1.0

    if leak:
        # 泄漏信号：直接由次日收益构造
        fwd = df["close"].shift(-1) / df["close"] - 1.0
        df["signal_leak"] = (fwd + rng.normal(0.0, 0.0008, n)).fillna(0.0)

    # 注入：某段时间复权价不一致（+1%），此时其他行的复权关系仍然自洽
    inc_rows = df.index[300:320]
    df.loc[inc_rows, "adj_close"] = df.loc[inc_rows, "close"] * df.loc[inc_rows, "adjust_factor"] * 1.01
    labels.append({"column": "adj_close", "timestamp": str(df.loc[inc_rows[0], "date"]),
                   "kind": "adjust_inconsistent_start", "station": "—"})

    # 注入：缺失交易日（删除 5 天）与重复日期（2 天）
    df = df.drop(index=df.index[200:205]).reset_index(drop=True)
    df = pd.concat([df, df.iloc[240:242].copy()], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df, pd.DataFrame(labels)


# ---------------- 天气评估专用集 ----------------
# 与演示集分离：演示集 weather_v1 保持不变（哈希不动），评估用更密的异常真值。
# 两项关键设计：
#   1) 异常按「分层撒点」注入，每 1/k 段一个点，保证每个 holdout 段都有足够真值；
#   2) 异常幅度按「残差稳健尺度」而非原始标准差换算——演示集的注入幅度使残差异常
#      Z 值超过 100，任何阈值都能全检出，阈值选择失去意义。
EVAL_COLUMNS = ("temperature", "humidity", "wind_speed")
EVAL_PER_COLUMN = 10                 # 每站每列的孤立点异常个数
EVAL_Z_RANGE = (5.5, 7.5)            # 目标残差稳健 Z 值区间
EVAL_RESID_SIGMA = {                 # 各列残差（去趋势去季节后）的稳健尺度估算
    "temperature": 1.5,
    "humidity": 8.0,
    "wind_speed": 1.4,
}
# 持续段异常：段长必须显著小于趋势窗口（24），否则滚动中位数会把整段当成趋势
# 而把残差拉平，导致异常无法检出。取 8 小时（窗口的 1/3）。
# 段数取 4：每折 holdout 约 5 段，避免只有 2 段时「漏 1 段 = 召回掉 50%」的剧烈波动。
EVAL_COLLECTIVE_SEGMENTS = 4
EVAL_COLLECTIVE_HOURS = 8
EVAL_COLLECTIVE_Z_RANGE = (7.0, 9.0)


def _stratified_positions(rng, n: int, k: int, exclude=None, margin: int = 1) -> list[int]:
    """把 [0, n) 等分 k 段，每段随机取一个位置，避免异常扎堆。"""
    edges = np.linspace(0, n - margin, k + 1).astype(int)
    out = []
    for i in range(k):
        lo, hi = int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))
        picked = None
        for _ in range(50):
            p = int(rng.integers(lo, hi))
            if exclude is not None and exclude[0] <= p < exclude[1]:
                continue
            picked = p
            break
        out.append(picked if picked is not None else (lo + hi) // 2)
    return out


def gen_weather_eval(seed: int = SEED + 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成评估专用天气集：基础信号同演示集，异常密度更高、幅度更接近检测边界。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(WEATHER_START, periods=WEATHER_HOURS, freq="h")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    t_year = (idx - idx[0]).days.to_numpy() / 365.25

    frames, labels = [], []
    for s_i, station in enumerate(WEATHER_STATIONS):
        n = WEATHER_HOURS
        phase = 0.0 if s_i == 0 else 0.6
        annual = 12.0 * np.sin(2 * np.pi * (doy - 100) / 365.25 + phase)
        diurnal = 6.0 * np.sin(2 * np.pi * (hour - 14) / 24.0 + phase)
        trend = 0.8 * t_year
        noise = _ar1(rng, n, 0.97, 0.35)
        temp = 15.0 + annual + diurnal + trend + noise + (1.5 if s_i == 0 else -1.5)

        humidity = 65.0 - 0.8 * (temp - 15.0) + _ar1(rng, n, 0.95, 2.5)
        humidity = np.clip(humidity, 5.0, 100.0)
        pressure = 1013.0 + 8.0 * np.sin(2 * np.pi * doy / 365.25) + _ar1(rng, n, 0.85, 1.2)
        wind = np.abs(3.0 + _ar1(rng, n, 0.90, 1.1) + 1.2 * np.sin(2 * np.pi * hour / 24.0))
        rain_u = rng.random(n)
        rainfall = np.where(rain_u < 0.94, 0.0,
                            rng.exponential(2.5, n) * (rain_u - 0.93) / 0.07)

        frames.append(pd.DataFrame({
            "timestamp": idx, "station": station,
            "temperature": temp, "humidity": humidity, "pressure": pressure,
            "wind_speed": wind, "rainfall": rainfall,
        }))

    data = pd.concat(frames, ignore_index=True)

    # ---- 异常注入（两类：孤立点 + 持续段；分层、按残差尺度换算幅度）----
    unit_span = (3000, 3720)          # 留给华氏段的区间，异常点避开
    for station in WEATHER_STATIONS:
        m = data["station"] == station
        n_st = int(m.sum())
        for col in EVAL_COLUMNS:
            sd_res = EVAL_RESID_SIGMA[col]
            span = unit_span if col == "temperature" else None

            # 类型一：孤立点异常
            pos = _stratified_positions(rng, n_st, EVAL_PER_COLUMN, exclude=span)
            for j, p in enumerate(pos):
                r = data.index[m][p]
                z = float(rng.uniform(*EVAL_Z_RANGE))
                sg = 1.0 if rng.random() < 0.5 else -1.0
                data.loc[r, col] = data.loc[r, col] + sg * z * sd_res
                labels.append({"column": col, "timestamp": str(data.loc[r, "timestamp"]),
                               "kind": "anomaly", "station": station,
                               "anomaly_type": "point", "event_id": f"{col}_{station}_p{j}"})

            # 类型二：持续段异常（段内每点都偏离，需按事件而非按点判定）
            cpos = _stratified_positions(rng, n_st, EVAL_COLLECTIVE_SEGMENTS,
                                         exclude=span, margin=EVAL_COLLECTIVE_HOURS)
            for j, p in enumerate(cpos):
                z = float(rng.uniform(*EVAL_COLLECTIVE_Z_RANGE))
                sg = 1.0 if rng.random() < 0.5 else -1.0
                eid = f"{col}_{station}_c{j}"
                for r in data.index[m][p:p + EVAL_COLLECTIVE_HOURS]:
                    jitter = 0.9 + 0.2 * float(rng.random())
                    data.loc[r, col] = data.loc[r, col] + sg * z * sd_res * jitter
                    labels.append({"column": col, "timestamp": str(data.loc[r, "timestamp"]),
                                   "kind": "anomaly", "station": station,
                                   "anomaly_type": "collective", "event_id": eid})

    # ---- 质量问题：与演示集同形态，确保清洗链路被完整覆盖 ----
    m_a = data["station"] == "STA_A"
    a_rows = data.index[m_a][unit_span[0]:unit_span[1]]
    labels.append({"column": "temperature", "timestamp": str(data.loc[a_rows[0], "timestamp"]),
                   "kind": "unit_change_start", "station": "STA_A"})
    data.loc[a_rows, "temperature"] = data.loc[a_rows, "temperature"] * 1.8 + 32.0

    m_b = data["station"] == "STA_B"
    miss_rows = data.index[m_b][5000:5072]
    labels.append({"column": "temperature", "timestamp": str(data.loc[miss_rows[0], "timestamp"]),
                   "kind": "missing_gap_start", "station": "STA_B"})
    data.loc[miss_rows, "temperature"] = np.nan

    gap_rows = []
    for station in WEATHER_STATIONS:
        gap_rows.extend(list(data.index[data["station"] == station][8000:8010]))
    data = data.drop(index=gap_rows).reset_index(drop=True)

    drop_idx = rng.choice(len(data), size=300, replace=False)
    data = data.drop(index=data.index[drop_idx]).reset_index(drop=True)

    dup_src = data.index[2400:2425]
    data = pd.concat([data, data.loc[dup_src].copy()], ignore_index=True)

    block = list(data.index[6000:6400])
    data = pd.concat([data.drop(index=block), data.loc[block]], ignore_index=True)

    return data, pd.DataFrame(labels)


def main() -> None:
    w, wl = gen_weather()
    w.to_csv(os.path.join(HERE, "weather_v1.csv"), index=False, encoding="utf-8-sig")
    wl.to_csv(os.path.join(HERE, "labels_weather_v1.csv"), index=False, encoding="utf-8-sig")

    s, sl = gen_stock()
    s.to_csv(os.path.join(HERE, "stock_v1.csv"), index=False, encoding="utf-8-sig")
    sl.to_csv(os.path.join(HERE, "labels_stock_v1.csv"), index=False, encoding="utf-8-sig")

    we, wel = gen_weather_eval()
    we.to_csv(os.path.join(HERE, "weather_eval_v1.csv"), index=False, encoding="utf-8-sig")
    wel.to_csv(os.path.join(HERE, "labels_weather_eval_v1.csv"), index=False, encoding="utf-8-sig")

    sl2, sl2l = gen_stock(leak=True)
    sl2.to_csv(os.path.join(HERE, "stock_leak_v1.csv"), index=False, encoding="utf-8-sig")
    sl2l.to_csv(os.path.join(HERE, "labels_stock_leak_v1.csv"), index=False, encoding="utf-8-sig")

    print("已生成：weather_v1.csv, weather_eval_v1.csv, stock_v1.csv, stock_leak_v1.csv 及对应标签")


if __name__ == "__main__":
    main()
