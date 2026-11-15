"""Guard the archives against omitted runtime assets and leaked notebooks."""
from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from scripts.check_distribution import REQUIRED, check


def archives(directory, *, wheel_extra=(), sdist_extra=(), omit=()):
    files = REQUIRED - set(omit)
    with zipfile.ZipFile(directory / "pymc_bot-0.1.0-py3-none-any.whl", "w") as wheel:
        for name in [*files, "pymc_bot-0.1.0.dist-info/licenses/LICENSE", *wheel_extra]:
            wheel.writestr(name, "fixture")
    with tarfile.open(directory / "pymc_bot-0.1.0.tar.gz", "w:gz") as sdist:
        for name in [*files, "LICENSE", *sdist_extra]:
            data = b"fixture"
            info = tarfile.TarInfo("pymc_bot-0.1.0/" + name)
            info.size = len(data)
            sdist.addfile(info, io.BytesIO(data))


def test_distribution_check_accepts_required_files(tmp_path):
    archives(tmp_path)
    check(tmp_path)


@pytest.mark.parametrize("where", ["wheel", "sdist"])
def test_distribution_check_rejects_notebooks(tmp_path, where):
    archives(tmp_path, **{where + "_extra": ["pynb/private.ipynb"]})
    with pytest.raises(ValueError, match="forbidden"):
        check(tmp_path)


def test_distribution_check_rejects_missing_runtime_assets(tmp_path):
    archives(tmp_path, omit=["pymc_bot/web/login.js"])
    with pytest.raises(ValueError, match="missing runtime files"):
        check(tmp_path)
