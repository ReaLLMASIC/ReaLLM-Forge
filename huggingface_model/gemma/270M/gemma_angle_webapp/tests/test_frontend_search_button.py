from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_search_buttons_are_rendered_for_all_token_pickers() -> None:
    html = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text()

    assert 'id="tokenASearch"' in html
    assert 'id="tokenBSearch"' in html
    assert 'id="anchorSearch"' in html
    assert html.count('type="button">Search</button>') == 3


def test_search_is_explicit_button_or_enter_not_keypress_debounce() -> None:
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()

    assert "debounce" not in js
    assert "addEventListener('input', debounce" not in js
    assert "queryInput.addEventListener('input', () => markSearchStale(prefix, key));" in js
    assert "searchButton.addEventListener('click', () => search(prefix, key));" in js
    assert "new URLSearchParams({ q: query })" in js
    assert "limit: '200'" not in js
    assert "if (event.key === 'Enter')" in js


def test_frontend_no_longer_searches_on_picker_setup() -> None:
    js = (PROJECT_ROOT / "app" / "static" / "app.js").read_text()
    setup_start = js.index("function setupPicker(prefix, key)")
    setup_end = js.index("\n}\n\nasync function computeAngle", setup_start)
    setup_body = js[setup_start:setup_end]

    assert "setSelectMessage(results, 'Click Search to load matches');" in setup_body
    assert "search(prefix, key);\n}" not in setup_body
