# -*- coding: utf-8 -*-
"""B3（C 档）：包能被安装、命令能被调用。

这一组测试不验证分析逻辑，只验证「交给别人能装能跑」的前提：
元数据完整、依赖声明齐全、入口点可导入、装了之后命令真的存在。
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


def test_project_metadata_is_complete():
    doc = _project()
    assert doc["name"] == "mtsa"
    assert doc["version"]
    assert doc["requires-python"].startswith(">=3.1")
    assert doc["readme"] == "README.md"
    assert doc["description"]


def test_runtime_dependencies_are_declared():
    deps = " ".join(_project()["dependencies"]).lower()
    for pkg in ("pandas", "numpy", "scipy", "statsmodels", "plotly", "kaleido", "pydantic"):
        assert pkg in deps, f"缺少运行时依赖声明：{pkg}"
    # 生成 Word 演示文档才需要的依赖，不能塞进运行时
    assert "python-docx" not in deps
    assert "python-docx" in " ".join(_project()["optional-dependencies"]["docx"])
    # 非 CSV 格式的可选依赖，装了才支持，不能塞进运行时
    io_extra = " ".join(_project()["optional-dependencies"]["io"]).lower()
    for pkg in ("openpyxl", "pyarrow", "h5py"):
        assert pkg in io_extra, f"缺少 io 可选依赖：{pkg}"


def test_console_script_is_declared_and_importable():
    target = _project()["scripts"]["mtsa"]
    assert target == "mtsa.cli:main"
    mod_name, func_name = target.split(":")
    mod = importlib.import_module(mod_name)
    assert callable(getattr(mod, func_name))


def test_src_layout_matches_package_find():
    assert (ROOT / "src" / "mtsa" / "__init__.py").is_file()
    cfg = tomllib.load(open(ROOT / "pyproject.toml", "rb"))["tool"]["setuptools"]
    assert cfg["package-dir"][""] == "src"
    assert cfg["packages"]["find"]["where"] == ["src"]


def _console_script() -> str | None:
    """先找 PATH，再找当前解释器旁边的 Scripts（pip 安装的默认落点）。"""
    hit = shutil.which("mtsa")
    if hit:
        return hit
    beside = Path(sys.executable).parent / ("mtsa.exe" if os.name == "nt" else "mtsa")
    return str(beside) if beside.exists() else None


def test_installed_command_works_or_is_skipped():
    exe = _console_script()
    if not exe:
        pytest.skip("当前环境未安装 mtsa 命令（pip install -e . 后才会出现）")
    out = subprocess.run([exe, "list-scenes"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "weather" in out.stdout and "stock" in out.stdout

def test_license_and_third_party_files_exist():
    """课程提交物必须带授权文件；缺了就等于默认「保留所有权利」却没有任何告知。"""
    lic = ROOT / "LICENSE"
    assert lic.is_file(), "缺少 LICENSE"
    text = lic.read_text(encoding="utf-8")
    assert "保留所有权利" in text
    assert "All rights reserved" in text
    assert "课程" in text
    # 交付前不能留占位符：证书主体必须是真人/账号，否则授权对象不明
    assert "待填写" not in text
    assert "qiaotong555" in text
    assert (ROOT / "doc" / "评估授权函_模板.md").is_file(), "缺少第三方评估授权函模板"

    for name in ("DATA_SOURCES.md", "THIRD_PARTY_NOTICES.md"):
        assert (ROOT / name).is_file(), f"缺少 {name}"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("LICENSE", "DATA_SOURCES.md", "THIRD_PARTY_NOTICES.md"):
        assert name in readme, f"README 没有指向 {name}"


def test_data_sources_covers_every_fixture():
    import json
    from pathlib import Path as _P
    reg = json.loads((ROOT / "fixtures" / "real_weather" / "registry.json").read_text(encoding="utf-8"))
    rows = reg["datasets"] if isinstance(reg, dict) and "datasets" in reg else list(reg.values())
    text = (ROOT / "DATA_SOURCES.md").read_text(encoding="utf-8")
    missing = [r["id"] for r in rows if r.get("id") not in text]
    assert not missing, f"DATA_SOURCES.md 漏了：{missing}"
