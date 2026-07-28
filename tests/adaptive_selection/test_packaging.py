from pathlib import Path

import toml
from setuptools import find_packages


def test_setuptools_discovery_packages_adaptive_experiments():
    root = Path(__file__).parents[2]
    configuration = toml.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    includes = configuration["tool"]["setuptools"]["packages"]["find"]["include"]

    assert "experiments*" in includes
    discovered = find_packages(where=root, include=includes)
    assert "experiments" in discovered
    assert "experiments.adaptive_selection" in discovered
