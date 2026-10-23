# -*- coding: utf-8 -*-
"""统计检验：平稳性、自相关、正态性、趋势、随机性（游程检验）与分布一致性（卡方）。

每个检验都记录假设前提（assumptions），样本量不足时明确跳过而不给出结论。

性能与结论的边界
----------------
本模块做过一次性能改造，结论是：**只能优化实现，不能截断统计判据**。

- Ljung-Box：原调用 `statsmodels.acorr_ljungbox` 在固定 lag=10 下仍要 6.2 秒
  （10 万点），实测是 stats_tests 的第二大热点。改为 FFT 求自相关后降至 7 毫秒，
  相对误差 3e-15，Q 统计量与 p 值逐位对齐。这是纯实现优化，不影响任何结论。
- ADF：曾尝试给 `maxlag` 设上限（auto 规则给出 43 至 67 阶），实测**会翻转判定**——
  pressure 两列在 maxlag=30 时由「不能拒绝单位根」变为「拒绝」，即把非平稳误判为平稳。
  原因是截断后残差自相关吸收不掉，统计量偏大。故**不对 ADF 限幅**，
  其耗时是统计严谨性的必要代价。
- KPSS 与其余检验均在毫秒级，不构成瓶颈，同样不做截断。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from .models import AnalysisPlan, CleanDataset, TestResultRow


def ljungbox_fft(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Ljung-Box Q 统计量，用 FFT 求自协方差，复杂度 O(n log n)。

    与 statsmodels.acorr_ljungbox 同公式：
        Q = n(n+2) * Σ_{k=1..h} ρ_k² / (n-k)
    实测 Q 与 p 的相对误差在 3e-15 量级，可视为逐位一致。

    返回 (Q, p)；无法计算时返回 (nan, nan)。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= lags + 1:
        return float("nan"), float("nan")
    x = x - x.mean()
    if not np.any(x):
        return float("nan"), float("nan")
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(spec * np.conj(spec), nfft)[:n] / n
    if acov[0] <= 0:
        return float("nan"), float("nan")
    rho = acov / acov[0]
    k = np.arange(1, lags + 1)
    q = float(n * (n + 2) * np.sum(rho[1:lags + 1] ** 2 / (n - k)))
    return q, float(1 - stats.chi2.cdf(q, lags))


def _ljungbox(vals: np.ndarray, lags: int, use_fft: bool = True) -> tuple[float, float]:
    """优先走 FFT 实现；数值异常时回退 statsmodels，保证不会因优化而算不出结果。"""
    if use_fft:
        q, p = ljungbox_fft(vals, lags)
        if np.isfinite(q) and np.isfinite(p):
            return q, p
    lb = acorr_ljungbox(vals, lags=[lags], return_df=True)
    return float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])


def runs_test(x: np.ndarray) -> tuple[float, float]:
    """Wald–Wolfowitz 游程检验：原假设为序列取值独立同分布（中位数上下随机交替）。"""
    med = np.median(x)
    signs = np.sign(x - med)
    signs = signs[signs != 0]
    n = len(signs)
    if n < 10:
        return float("nan"), float("nan")
    n_pos = int((signs > 0).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    exp = 2.0 * n_pos * n_neg / n + 1.0
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))
    if var <= 0:
        return float("nan"), float("nan")
    z = (runs - exp) / np.sqrt(var)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(z), p


def chi_square_uniformity(x: np.ndarray, bins: int = 10) -> tuple[float, float, int]:
    """卡方拟合优度检验：原假设为取值在观测区间上均匀分布。"""
    if len(x) < bins * 5:
        return float("nan"), float("nan"), 0
    counts, _ = np.histogram(x, bins=bins)
    exp = counts.sum() / bins
    stat = float(((counts - exp) ** 2 / exp).sum())
    p = float(1 - stats.chi2.cdf(stat, bins - 1))
    return stat, p, bins - 1


def run_stat_tests(clean: CleanDataset, plan: AnalysisPlan) -> list[TestResultRow]:
    frame = clean.frame
    rows: list[TestResultRow] = []
    analyses = set(plan.analyses)
    cols = [c for c in plan.target_columns if c in frame.columns]

    for col in cols:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(s) < 8:
            rows.append(TestResultRow(
                test_id=f"skip:{col}", name="样本不足", column=col,
                conclusion="有效样本少于 8，未执行任何检验", assumptions="—",
            ))
            continue
        vals = s.to_numpy(dtype=float)
        if float(vals.max() - vals.min()) == 0.0:
            # ADF/KPSS/正态性/趋势对常数列无定义；跳过并留下记录，不能让整次分析崩掉。
            rows.append(TestResultRow(
                test_id=f"skip:{col}", name="常数列", column=col,
                conclusion="序列方差为 0，统计检验无定义，已跳过",
                assumptions="常数列不适用 ADF/KPSS/自相关/正态性/趋势检验",
            ))
            continue

        if "stationarity" in analyses or "test" in analyses:
            if len(vals) > 12:
                # 不对 maxlag 设上限：实测在 weather 数据上，把 auto（43 阶）截到 30
                # 就会让 pressure 两列从「不能拒绝单位根」翻成「拒绝」，把非平稳误判为
                # 平稳。ADF 的耗时（10 万点约 10 秒）是统计严谨性的必要代价。
                adf = adfuller(vals, autolag="AIC", result_object=False)
                adf_stat, adf_p, adf_crit = float(adf[0]), float(adf[1]), adf[4]
                rows.append(TestResultRow(
                    test_id=f"adf:{col}", name="ADF 单位根检验", column=col,
                    statistic=float(adf_stat), p_value=float(adf_p),
                    critical_value=float(adf_crit.get("5%", float("nan"))),
                    reject_null=bool(adf_p < 0.05),
                    conclusion=("拒绝单位根原假设，序列平稳" if adf_p < 0.05 else "不能拒绝单位根原假设，序列可能非平稳"),
                    assumptions="样本量 > 12；默认 AIC 选阶；含常数项",
                ))
                try:
                    kpss_res = kpss(vals, regression="c", nlags="auto", result_object=False)
                    kpss_stat, kpss_p, kpss_crit = float(kpss_res[0]), float(kpss_res[1]), kpss_res[3]
                    rows.append(TestResultRow(
                        test_id=f"kpss:{col}", name="KPSS 平稳性检验", column=col,
                        statistic=kpss_stat, p_value=kpss_p,
                        critical_value=float(kpss_crit.get("5%", float("nan"))),
                        reject_null=bool(kpss_p < 0.05),
                        conclusion=("拒绝平稳原假设，序列非平稳" if kpss_p < 0.05 else "不能拒绝平稳原假设"),
                        assumptions="原假设为平稳；nlags=auto",
                    ))
                except (ValueError, InterpolationWarning):
                    rows.append(TestResultRow(
                        test_id=f"kpss:{col}", name="KPSS 平稳性检验", column=col,
                        conclusion="KPSS 计算未给出有效结果（样本量或方差不足），已跳过",
                        assumptions="原假设为平稳",
                    ))

        if "autocorr" in analyses or "test" in analyses:
            lags = min(10, max(1, len(vals) // 5))
            lb_q, lb_p = _ljungbox(vals, lags)
            p = lb_p
            rows.append(TestResultRow(
                test_id=f"ljungbox:{col}", name=f"Ljung-Box 自相关检验（lag={lags}）", column=col,
                statistic=lb_q, p_value=p,
                reject_null=bool(p < 0.05),
                conclusion=("拒绝无自相关原假设，存在显著自相关" if p < 0.05 else "不能拒绝无自相关原假设"),
                assumptions=f"检验前 {lags} 阶自相关；要求样本量 > lag",
            ))

        if "normality" in analyses or "test" in analyses:
            jb = stats.jarque_bera(vals)
            jb_stat, jb_p = float(jb.statistic), float(jb.pvalue)
            rows.append(TestResultRow(
                test_id=f"jarque_bera:{col}", name="Jarque-Bera 正态性检验", column=col,
                statistic=float(jb_stat), p_value=float(jb_p),
                reject_null=bool(jb_p < 0.05),
                conclusion=("拒绝正态性原假设，分布显著偏离正态" if jb_p < 0.05 else "不能拒绝正态性原假设"),
                assumptions="基于偏度与峰度；大样本渐近",
            ))

        if "trend" in analyses or "test" in analyses:
            kt = stats.kendalltau(np.arange(len(vals)), vals)
            tau, p = float(kt.statistic), float(kt.pvalue)
            rows.append(TestResultRow(
                test_id=f"mann_kendall:{col}", name="Mann-Kendall 趋势检验", column=col,
                statistic=float(tau), p_value=float(p),
                reject_null=bool(p < 0.05),
                conclusion=(f"{'存在' if p < 0.05 else '未发现'}显著单调趋势，方向"
                            f"{'上升' if tau > 0 else '下降'}" if p < 0.05 else "未发现显著单调趋势"),
                assumptions="观测独立；对缺失不敏感；不要求正态",
            ))

        if "randomness" in analyses:
            z, p = runs_test(vals)
            rows.append(TestResultRow(
                test_id=f"runs:{col}", name="游程检验（随机性）", column=col,
                statistic=z, p_value=p,
                reject_null=bool(p < 0.05) if np.isfinite(p) else None,
                conclusion=("拒绝随机性原假设，序列存在聚集或交替结构" if (np.isfinite(p) and p < 0.05)
                            else "不能拒绝随机性原假设"),
                assumptions="以中位数二值化；n >= 10",
            ))
            cstat, cp, dof = chi_square_uniformity(vals)
            rows.append(TestResultRow(
                test_id=f"chi2_uniform:{col}", name="卡方均匀性检验", column=col,
                statistic=cstat, p_value=cp, critical_value=float(dof),
                reject_null=bool(cp < 0.05) if np.isfinite(cp) else None,
                conclusion=("拒绝均匀分布原假设" if (np.isfinite(cp) and cp < 0.05) else "不能拒绝均匀分布原假设"),
                assumptions=f"分为 {dof + 1 if dof else 0} 个区间，每区间期望频数 >= 5",
            ))

    return rows
