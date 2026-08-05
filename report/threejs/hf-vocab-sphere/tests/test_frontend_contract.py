from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "app" / "templates" / "index.html"
JS_PATH = ROOT / "app" / "static" / "app.js"


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)


def test_frontend_control_contract_and_unique_ids() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    parser = _IdCollector()
    parser.feed(html)
    assert len(parser.ids) == len(set(parser.ids))

    required = {
        "scene",
        "label-overlay",
        "nodeLabelSizeInput",
        "edgeLabelSizeInput",
        "tokenizeTextInput",
        "tokenizeTextBtn",
        "addAllTokenizedBtn",
        "replaceTokenizedBtn",
        "arithmeticModeSelect",
        "slerpFromSelect",
        "slerpToSelect",
        "slerpFractionInput",
        "showLabelAliasesInput",
        "showLabelIdsInput",
        "showLabelMagnitudesInput",
        "showLabelParticipationInput",
        "saveSettingsBtn",
        "loadSettingsBtn",
        "settingsFileInput",
        "settingsStatus",
    }
    assert required <= set(parser.ids)


def test_javascript_dom_references_and_unbounded_label_defaults() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    javascript = JS_PATH.read_text(encoding="utf-8")
    parser = _IdCollector()
    parser.feed(html)
    ids = set(parser.ids)
    references = set(re.findall(r"\$\(['\"]([^'\"]+)['\"]\)", javascript))
    assert references <= ids

    assert "edgeLabelLimit: 0" in javascript
    assert "labelEntries = [...new Set(candidates)]" in javascript
    assert "available.slice(0, requestedLimit)" in javascript
    assert "drawLabelOverlay();" in javascript
    assert "positionBuffer.needsUpdate = true" in javascript
    assert "colorBuffer.needsUpdate = true" in javascript
    assert "edgeMaterial.needsUpdate = true" not in javascript


def test_label_component_and_settings_contract() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    javascript = JS_PATH.read_text(encoding="utf-8")

    assert "showLabelAliases: true" in javascript
    assert "showLabelIds: true" in javascript
    assert "showLabelMagnitudes: false" in javascript
    assert "showLabelParticipation: false" in javascript
    assert "show_label_magnitudes: state.showLabelMagnitudes" in javascript
    assert "show_label_participation: state.showLabelParticipation" in javascript
    assert "filter((value) => value.length).join(' · ')" in javascript
    assert "event.key.toLowerCase() === 't'" in javascript
    assert "event.key.toLowerCase() === 'i'" in javascript
    assert "event.key.toLowerCase() === 'm'" in javascript
    assert "event.key.toLowerCase() === 'p'" in javascript

    assert "const SETTINGS_SCHEMA = 'hf-vocab-sphere/settings'" in javascript
    assert "function buildSettingsSnapshot()" in javascript
    assert "function validateSettingsSnapshot(snapshot)" in javascript
    assert "function loadSettingsFile(file)" in javascript
    assert "settingsModelMatchesActive" in javascript
    assert "selected_tokens: selectedRows().map(compactSelectedToken)" in javascript
    assert 'accept=".json,application/json"' in html
