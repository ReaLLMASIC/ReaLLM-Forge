from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_search_buttons_exist_for_each_picker() -> None:
    html = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text()
    for prefix in ["tokenA", "tokenB", "anchor"]:
        assert f'id="{prefix}Search"' in html
        assert f'id="{prefix}Query"' in html
        assert f'id="{prefix}Results"' in html



def test_search_is_not_triggered_by_typing_or_initialization() -> None:
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()
    setup = js[js.index("function setupPicker"):js.index("async function computeAngle")]
    dom_ready = js[js.index("window.addEventListener('DOMContentLoaded'"):]

    assert "debounce" not in setup
    assert "addEventListener('input', debounce" not in setup
    assert "queryInput.addEventListener('input', () => markSearchStale(prefix, key))" in setup
    assert "searchButton.addEventListener('click', () => search(prefix, key))" in setup
    assert "event.key === 'Enter'" in setup
    assert "limit: '200'" not in js

    # The old implementation called search(prefix, key) during setupPicker and
    # then setupPicker during DOMContentLoaded, which caused automatic blank
    # searches. Now setup only wires events and the page load only calls setup.
    setup_lines = [line.strip() for line in setup.splitlines() if line.strip()]
    assert setup_lines[-1] == "}"
    assert setup_lines[-2] == "byId(`${prefix}UseId`).addEventListener('click', () => selectFromId(prefix, key));"
    assert "search(prefix" not in dom_ready


def test_model_loader_controls_exist() -> None:
    html = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text()
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()

    assert 'id="modelNameInput"' in html
    assert 'id="loadModelButton"' in html
    assert 'id="localModelSelect"' in html
    assert 'id="refreshLocalModelsButton"' in html
    assert 'id="allowDownloadInput"' in html
    assert 'placeholder="Qwen/Qwen3.5-0.8B-Base"' in html
    assert "fetchJson('/api/model/load'" in js
    assert "fetchJson('/api/status')" in js
    assert "fetchJson('/api/models/available')" in js
    assert "allow_download: allowDownload" in js
    assert "Download if missing" in html
    assert "No model loaded yet" in js
    assert 'loadRequestedModel' in js
    assert "loadAvailableModels" in js
    assert "resetAllSelectionsAfterModelChange" in js
    assert "loadModelFromInput" not in js


def test_pairwise_angle_distribution_controls_exist() -> None:
    html = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text()
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()

    assert 'id="pairwiseBinsButton"' in html
    assert 'id="pairwiseBlockSize"' in html
    assert 'id="pairwiseComputeDevice"' in html
    assert 'id="pairwiseIncludeSelf"' in html
    assert 'id="pairwiseLogScale"' in html
    assert 'id="pairwiseAnglePlot"' in html
    assert 'id="pairwiseBinsTable"' in html
    assert 'Compute pairwise bins' in html
    assert 'fetchJson(`/api/pairwise-angle-bins?' in js
    assert 'drawPairwiseAngleBinPlot' in js
    assert 'buildLogTicks' in js
    assert 'buildLinearTicks' in js
    assert 'log10' in js
    assert 'resetPairwiseBinsOutput' in js


def test_common_close_tokens_controls_exist() -> None:
    html = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text()
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()
    css = (PROJECT_ROOT / "app" / "static" / "styles.css").read_text()

    assert 'id="commonCloseThreshold"' in html
    assert 'value="35"' in html
    assert 'id="commonCloseButton"' in html
    assert 'id="commonCloseOutput"' in html
    assert 'id="commonCloseTable"' in html
    assert 'Angle to A °' in html
    assert 'Angle to B °' in html
    assert 'fetchJson(`/api/common-close-tokens?' in js
    assert 'computeCommonCloseTokens' in js
    assert 'renderCommonCloseTable' in js
    assert 'resetCommonCloseOutput' in js
    assert 'common-close-scroll' in css


def test_pairwise_bin_token_list_ui_exists_and_rows_are_clickable() -> None:
    html = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text()
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()
    css = (PROJECT_ROOT / "app" / "static" / "styles.css").read_text()

    assert 'id="pairwiseBinTokensPanel"' in html
    assert 'id="pairwiseBinTokensTable"' in html
    assert 'Unique tokens' in html
    assert 'Search label' in html
    assert 'loadPairwiseBinTokens' in js
    assert 'fetchJson(`/api/pairwise-angle-bins/${bin.bin_index}/tokens`)' in js
    assert 'clickable-row' in js
    assert 'token-list-scroll' in css
    assert 'overflow: auto' in css


def test_homepage_avoids_template_response_signature_path() -> None:
    main_py = (PROJECT_ROOT / "app" / "main.py").read_text()
    assert "TemplateResponse" not in main_py
    assert "Jinja2Templates" not in main_py
    assert "INDEX_TEMPLATE_PATH.read_text" in main_py


def test_pairwise_actions_resolve_typed_ids_before_running() -> None:
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()
    assert "async function ensurePickerSelection" in js
    assert "async function ensurePairwiseSelections" in js
    assert "await ensurePairwiseSelections()" in js
    assert "Search/select a token or enter an ID and click Use ID" in js


def test_export_buttons_exist_for_tables_and_graphs() -> None:
    html = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text()
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()

    for export_id in [
        "exportAngleSummaryCsv",
        "exportCommonCloseCsv",
        "exportNeighborhoodCsv",
        "exportPairwisePlotPng",
        "exportPairwiseBinsCsv",
        "exportPairwiseBinTokensCsv",
        "exportMinDistancePlotPng",
        "exportMinDistancesCsv",
    ]:
        assert f'id="{export_id}"' in html
        assert export_id in js

    assert "exportHtmlTableAsCsv" in js
    assert "exportCanvasAsPng" in js
    assert "setupExportButtons" in js


def test_minimum_angular_distance_ui_exists() -> None:
    html = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text()
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()
    css = (PROJECT_ROOT / "app" / "static" / "styles.css").read_text()

    assert 'id="minDistancesButton"' in html
    assert 'id="minDistanceBlockSize"' in html
    assert 'id="minDistanceComputeDevice"' in html
    assert 'id="minDistanceSort"' in html
    assert 'id="minDistancePlot"' in html
    assert 'id="minDistancesTable"' in html
    assert "Closest non-self token for every token" in html
    assert "fetchJson(`/api/min-angular-distances?" in js
    assert "computeMinAngularDistances" in js
    assert "drawMinDistancePlot" in js
    assert "renderMinDistanceTable" in js
    assert "resetMinDistancesOutput" in js
    assert "min-distance-scroll" in css
