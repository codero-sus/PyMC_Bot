"""Authorship and credits: codero-sus, GitHub and the website are shown everywhere."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import pymc_bot
from pymc_bot.__main__ import main
from pymc_bot.server import create_app

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/codero-sus"
WEBSITE = "https://the-codero.web.app/"


@pytest.fixture()
def client(tmp_path: Path):
    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as test_client:
        yield test_client


def test_package_metadata_names_the_author():
    assert pymc_bot.__author__ == "codero-sus"
    assert pymc_bot.__author_url__ == GITHUB
    assert pymc_bot.__website__ == WEBSITE
    assert "codero-sus" in pymc_bot.MADE_WITH and "\u2764" in pymc_bot.MADE_WITH
    assert pymc_bot.credit_line().startswith(f"PyMC_Bot {pymc_bot.__version__} by codero-sus")


def test_api_about(client: TestClient):
    data = client.get("/api/about").json()
    assert data["author"] == "codero-sus"
    assert data["author_github"] == GITHUB
    assert data["website"] == WEBSITE
    assert data["repository"] == "https://github.com/codero-sus/PyMC_Bot"
    assert data["made_with"] == pymc_bot.MADE_WITH
    names = {credit["name"]: credit for credit in data["credits"]}
    assert names["PyMC_Bot"]["by"] == "codero-sus"
    assert names["Cortex LLMHoster"]["url"] == "https://github.com/codero-sus/Cortex_LLMHoster"


def test_openapi_docs_credit_the_author(client: TestClient):
    info = client.get("/openapi.json").json()["info"]
    assert info["contact"] == {"name": "codero-sus", "url": GITHUB}
    assert WEBSITE in info["description"] and "codero-sus" in info["description"]


def test_panel_footer_and_about_dialog(client: TestClient):
    html = client.get("/").text
    assert "Made with" in html and "codero-sus" in html
    assert f'href="{GITHUB}"' in html
    assert f'href="{WEBSITE}"' in html
    assert 'id="about-dialog"' in html
    # the About dialog is rendered server-side from pymc_bot.about()
    assert "Cortex LLMHoster" in html and "Built with" in html
    assert '<meta name="author" content="codero-sus"' in html
    assert "/static/fx.js" in html
    # Jinja placeholders are all resolved
    assert "{{" not in html and "{%" not in html


def test_effects_script_is_served(client: TestClient):
    response = client.get("/static/fx.js")
    assert response.status_code == 200
    assert "window.FX" in response.text
    assert "codero-sus" in response.text


def test_cli_about_and_version(capsys):
    assert main(["about"]) == 0
    out = capsys.readouterr().out
    assert "Made with" in out and GITHUB in out and WEBSITE in out

    assert main(["about", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["author"] == "codero-sus"

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == pymc_bot.credit_line()


def test_pyproject_credits_the_author():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["authors"] == [{"name": "codero-sus"}]
    assert project["urls"]["Homepage"] == WEBSITE
    assert project["urls"]["Author"] == GITHUB
    assert len(project["dependencies"]) >= 5  # urls table must not swallow the dependency list


def test_project_files_credit_the_author():
    package = json.loads((ROOT / "package.json").read_text())
    assert package["author"]["name"] == "codero-sus" and package["homepage"] == WEBSITE

    mod = json.loads((ROOT / "mod/src/main/resources/fabric.mod.json").read_text().replace("${version}", "0"))
    assert mod["authors"][0]["name"] == "codero-sus"
    assert mod["contact"]["homepage"] == WEBSITE

    readme = (ROOT / "README.md").read_text()
    assert "Made with" in readme and GITHUB in readme and WEBSITE in readme
    assert re.search(r"Copyright \(c\) \d{4} codero-sus", (ROOT / "LICENSE").read_text())


def test_every_panel_asset_is_packaged():
    """A pip-installed PyMC_Bot must ship the scripts, styles and icon the page loads."""
    tomllib = pytest.importorskip("tomllib")
    patterns = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["pymc_bot"]
    web = ROOT / "pymc_bot" / "web"
    packaged = {path.name for pattern in patterns for path in (ROOT / "pymc_bot").glob(pattern)}
    referenced = set(re.findall(r'/static/([\w.-]+)', (web / "index.html").read_text()))
    assert referenced, "index.html should reference static assets"
    assert referenced <= packaged, f"not packaged: {sorted(referenced - packaged)}"


def test_cortex_links_point_at_the_repository_root():
    """Cortex is referenced by its repository only, never by a ref, path or clone flag."""
    pattern = re.compile(
        r"Cortex_LLMHoster(?:\.git)?(?:/(?:tree|blob|commit|commits|compare|archive)\b|\?ref=|#|@)"
        r"|git clone[^\n]*\s(?:-b|--branch)[\s=][^\n]*Cortex_LLMHoster",
        re.IGNORECASE,
    )
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", ".venv", "node_modules", "build"} for part in path.parts):
            continue
        if path.suffix not in {".md", ".py", ".js", ".html", ".css", ".json", ".toml", ".yml", ".yaml", ".java", ".txt"}:
            continue
        text = path.read_text(errors="ignore")
        offenders += [f"{path.relative_to(ROOT)}: {m.group(0)}" for m in pattern.finditer(text)]
    assert not offenders, offenders
