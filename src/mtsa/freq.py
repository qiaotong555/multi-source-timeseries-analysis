# -*- coding: utf-8 -*-
"""频域分析：Welch 功率谱、主频与周期、分频段能量、时频图数据。

采样率由时间轴中位间隔推算；未识别到时间轴时返回空结果并告警，不编造采样率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .models import AnalysisPlan, CleanDataset, SpectrumResult
from .metrics import periods_per_year


def sampling_rate_hz(index: pd.Index) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
        med = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 0.0
        if med > 0:
            return 1.0 / med
    return float("nan")


def _period_label(freq: float) -> str:
    if freq <= 0:
        return "无穷大（直流/趋势）"
    sec = 1.0 / freq
    if sec < 90:
        return f"{sec:.1f} 秒"
    if sec < 5400:
        return f"{sec / 60:.1f} 分钟"
    if sec < 172800:
        return f"{sec / 3600:.2f} 小时"
    return f"{sec / 86400:.2f} 天"


def analyze_spectrum(clean: CleanDataset, plan: AnalysisPlan) -> list[SpectrumResult]:
    frame = clean.frame
    fs = sampling_rate_hz(frame.index)
    out: list[SpectrumResult] = []
    if not np.isfinite(fs) or fs <= 0:
        return out

    nperseg = int(plan.parameters.get("nperseg", 0)) or int(min(256, max(32, len(frame) // 8)))
    for col in plan.target_columns:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() < nperseg * 2:
            continue
        s = s.interpolate(limit_direction="both").fillna(0.0)
        vals = s.to_numpy(dtype=float)
        freqs, power = signal.welch(vals, fs=fs, nperseg=nperseg)
        # 忽略直流与极低频趋势分量以寻找真实周期
        mask = freqs > 0
        if not mask.any():
            continue
        sub_f, sub_p = freqs[mask], power[mask]
        dom_i = int(np.argmax(sub_p))
        dom_f = float(sub_f[dom_i])
        total = float(sub_p.sum())
        if total <= 0:
            continue
        thirds = len(sub_f) // 3
        bands = {
            "低频": float(sub_p[:thirds].sum() / total),
            "中频": float(sub_p[thirds:2 * thirds].sum() / total),
            "高频": float(sub_p[2 * thirds:].sum() / total),
        }
        step = max(1, len(freqs) // 500)
        out.append(SpectrumResult(
            column=col,
            sampling_rate_hz=float(fs),
            freqs=[round(float(x), 8) for x in freqs[::step]],
            power=[round(float(x), 10) for x in power[::step]],
            dominant_freq=round(dom_f, 8),
            dominant_period=_period_label(dom_f),
            band_power={k: round(v, 6) for k, v in bands.items()},
            method=f"welch(nperseg={nperseg})",
        ))
    return out


def compute_spectrogram(series: pd.Series, fs: float, nperseg: int = 64):
    """返回（时间轴秒、频率轴、功率谱密度矩阵），供时频图使用。"""
    vals = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    f, t, S = signal.spectrogram(vals, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return t, f, 10 * np.log10(S + 1e-12)
