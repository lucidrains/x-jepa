from pathlib import Path
from zipfile import ZipFile

from hatchling.build import build_wheel


def test_wheel_contains_import_package(tmp_path, monkeypatch):
    project_root = Path(__file__).parents[1]
    monkeypatch.chdir(project_root)

    wheel_name = build_wheel(str(tmp_path))
    wheel_path = tmp_path / wheel_name

    with ZipFile(wheel_path) as wheel:
        packaged_files = set(wheel.namelist())

    assert 'x_jepa/__init__.py' in packaged_files
    assert 'x_jepa/x_jepa.py' in packaged_files
