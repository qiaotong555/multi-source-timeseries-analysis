# -*- coding: utf-8 -*-
"""多格式数据导入：CSV / Excel / Parquet / NPY / MAT / HDF5。

导入为只读操作；任何格式不支持都明确报错，不做静默兜底。
"""
from __future__ import annotations

import hashlib
import importlib
import os
from typing import Optional

import numpy as np
import pandas as pd

from .models import ERR_DATA_LOAD_FAILED, DatasetSpec, RawDataset, SUPPORTED_FORMATS


def sha1_of_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# 可选格式的依赖：缺失时要给「装哪个包」的指令，而不是把 ImportError 原样抛给用户
_OPTIONAL_DEP = {
    ".xlsx": ("openpyxl", "mtsa[io]"),
    ".xls": ("openpyxl", "mtsa[io]"),
    ".parquet": ("pyarrow", "mtsa[io]"),
    ".h5": ("h5py", "mtsa[io]"),
    ".hdf5": ("h5py", "mtsa[io]"),
}


def _import_optional(name: str):
    """可选依赖的唯一导入入口，便于测试模拟「没装」。"""
    return importlib.import_module(name)


def _missing_dep(fmt: str, exc: ImportError) -> ValueError:
    mod, extra = _OPTIONAL_DEP.get(fmt, (exc.name or "该格式的依赖", "mtsa[io]"))
    return ValueError(
        f"读取 {fmt} 需要可选依赖 {mod}：pip install \"{extra}\"（或直接 pip install {mod}）；"
        "只处理 CSV 时不需要装它")


def _read_table(path: str, fmt: str) -> pd.DataFrame:
    if fmt in (".csv",):
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")
    if fmt in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise _missing_dep(fmt, exc) from exc
    if fmt == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise _missing_dep(fmt, exc) from exc
    raise ValueError(f"unsupported format: {fmt}")


def _read_npy(path: str) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 1:
        return pd.DataFrame({"value": arr})
    if arr.ndim == 2:
        return pd.DataFrame(arr, columns=[f"col_{i}" for i in range(arr.shape[1])])
    raise ValueError(f"NPY 数组维度 {arr.ndim} 不支持，仅支持 1D 或 2D")


def _read_mat(path: str) -> pd.DataFrame:
    from scipy import io as sio

    raw = sio.loadmat(path)
    arrays = [(k, v) for k, v in raw.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
    if not arrays:
        raise ValueError("MAT 文件中未找到数值数组")
    name, arr = max(arrays, key=lambda kv: kv[1].size)
    if arr.ndim == 1:
        return pd.DataFrame({name: arr.ravel()})
    return pd.DataFrame(arr, columns=[f"{name}_{i}" for i in range(arr.shape[1])])


def _read_hdf5(path: str) -> pd.DataFrame:
    try:
        h5py = _import_optional("h5py")
    except ImportError as exc:
        raise _missing_dep(".h5", exc) from exc

    with h5py.File(path, "r") as f:
        keys = []

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)

        f.visititems(_visit)
        if not keys:
            raise ValueError("HDF5 文件中未找到 dataset")
        cols = {}
        for k in keys[:20]:
            data = f[k][()]
            if data.ndim == 1:
                cols[k.replace("/", "_")] = data
        if not cols:
            raise ValueError("HDF5 中未找到一维 dataset")
        return pd.DataFrame(cols)


def build_spec(path: str, **kwargs) -> DatasetSpec:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"不支持的文件格式 '{ext}'，当前支持：{', '.join(SUPPORTED_FORMATS)}"
        )
    return DatasetSpec(path=path, format=ext, **kwargs)


def load_dataset(
    path: str,
    time_column: Optional[str] = None,
    value_columns: Optional[list[str]] = None,
    group_column: Optional[str] = None,
    unit: Optional[str] = None,
    sampling_interval: Optional[str] = None,
) -> RawDataset:
    """导入数据集。失败时抛出 ValueError，由调用方转为 DATA_LOAD_FAILED。"""
    if not os.path.exists(path):
        raise ValueError(f"输入文件不存在：{path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"输入文件为空：{path}")

    spec = build_spec(
        path,
        time_column=time_column,
        value_columns=value_columns,
        group_column=group_column,
        unit=unit,
        sampling_interval=sampling_interval,
    )
    fmt = spec.format
    try:
        if fmt in (".csv", ".xlsx", ".xls", ".parquet"):
            frame = _read_table(path, fmt)
        elif fmt == ".npy":
            frame = _read_npy(path)
        elif fmt == ".mat":
            frame = _read_mat(path)
        else:
            frame = _read_hdf5(path)
    except ValueError:
        raise
    except Exception as exc:  # 明确上报，不编造替代结果
        raise ValueError(f"读取 {fmt} 文件失败：{exc}") from exc

    if frame.empty:
        raise ValueError(f"输入文件解析后为空表：{path}")

    return RawDataset(
        frame=frame,
        spec=spec,
        file_hash=sha1_of_file(path),
        row_count=int(len(frame)),
        column_dtypes={c: str(t) for c, t in frame.dtypes.items()},
    )
