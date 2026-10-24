# -*- coding: utf-8 -*-
"""导入层测试：正常、边界、失败场景。"""
from __future__ import annotations

import os

import numpy as np
import pytest

from mtsa import io_loaders
from mtsa.models import SUPPORTED_FORMATS


def test_load_csv_ok(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("timestamp,value\n2024-01-01,1.0\n2024-01-02,2.0\n", encoding="utf-8")
    raw = io_loaders.load_dataset(str(p))
    assert raw.row_count == 2
    assert raw.spec.format == ".csv"
    assert len(raw.file_hash) == 40


def test_load_missing_file(tmp_path):
    with pytest.raises(ValueError, match="输入文件不存在"):
        io_loaders.load_dataset(str(tmp_path / "nope.csv"))


def test_load_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="输入文件为空"):
        io_loaders.load_dataset(str(p))


def test_unsupported_format(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的文件格式"):
        io_loaders.load_dataset(str(p))


def test_supported_formats_constant():
    for ext in (".csv", ".xlsx", ".parquet", ".npy", ".mat", ".h5"):
        assert ext in SUPPORTED_FORMATS


def test_load_npy(tmp_path):
    p = tmp_path / "a.npy"
    np.save(str(p), np.arange(30, dtype=float))
    raw = io_loaders.load_dataset(str(p))
    assert raw.row_count == 30
    assert "value" in raw.frame.columns


def test_sha1_stable(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("a\n1\n", encoding="utf-8")
    h1 = io_loaders.sha1_of_file(str(p))
    h2 = io_loaders.sha1_of_file(str(p))
    assert h1 == h2 and len(h1) == 40


def test_load_csv_with_bom(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_text("\ufefftimestamp,value\n2024-01-01,1\n", encoding="utf-8")
    raw = io_loaders.load_dataset(str(p))
    assert list(raw.frame.columns) == ["timestamp", "value"]

def test_missing_excel_engine_reports_install_hint(tmp_path, monkeypatch):
    """没装 openpyxl 时不能只抛 ImportError，要告诉用户装什么。"""
    path = tmp_path / "a.xlsx"
    path.write_bytes(b"not-a-real-xlsx")

    def boom(*args, **kwargs):
        raise ImportError("No module named 'openpyxl'")

    monkeypatch.setattr(io_loaders.pd, "read_excel", boom)
    with pytest.raises(ValueError, match="openpyxl"):
        io_loaders.load_dataset(str(path))


def test_missing_h5py_reports_install_hint(tmp_path, monkeypatch):
    path = tmp_path / "a.h5"
    path.write_bytes(b"\x00" * 16)

    def boom(name):
        raise ImportError(f"No module named '{name}'")

    monkeypatch.setattr(io_loaders, "_import_optional", boom)
    with pytest.raises(ValueError, match="h5py"):
        io_loaders.load_dataset(str(path))
