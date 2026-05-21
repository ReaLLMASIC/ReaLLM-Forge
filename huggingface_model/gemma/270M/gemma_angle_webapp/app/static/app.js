const state = {
  tokenA: null,
  tokenB: null,
  anchor: null,
  angleSummary: null,
  commonCloseRows: [],
  neighborhoodRows: [],
  pairwiseBins: null,
  pairwiseSelectedBinIndex: null,
  pairwiseBinTokens: [],
  minDistanceRows: [],
};

const pickerPrefixes = ['tokenA', 'tokenB', 'anchor'];

const cache = {
  tokenA: [],
  tokenB: [],
  anchor: [],
};

const searchRequestIds = {
  tokenA: 0,
  tokenB: 0,
  anchor: 0,
};

const lastSearchedQuery = {
  tokenA: null,
  tokenB: null,
  anchor: null,
};


function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatErrorDetail(detail) {
  if (!detail) return 'Request failed';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') return item;
        const location = Array.isArray(item.loc) ? item.loc.join('.') : item.loc;
        const message = item.msg || JSON.stringify(item);
        return location ? `${location}: ${message}` : message;
      })
      .join('; ');
  }
  try {
    return JSON.stringify(detail);
  } catch (_) {
    return String(detail);
  }
}

function tokenLabel(token) {
  if (!token) return 'No token selected.';
  const raw = JSON.stringify(token.raw);
  const display = token.display && token.display !== token.raw ? ` — ${JSON.stringify(token.display)}` : '';
  return `${String(token.token_id).padStart(6, ' ')} | ${raw}${display}`;
}

function setOutput(elementId, html, muted = false) {
  const el = byId(elementId);
  el.classList.toggle('muted', muted);
  el.innerHTML = html;
}

function setExportVisible(elementId, visible) {
  const element = byId(elementId);
  if (!element) return;
  element.classList.toggle('hidden', !visible);
}

function filenamePart(value, fallback = 'export') {
  const text = String(value || fallback).trim() || fallback;
  return text.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 120) || fallback;
}

function downloadBlob(blobOrText, filename, mime = 'text/plain;charset=utf-8') {
  const blob = blobOrText instanceof Blob ? blobOrText : new Blob([blobOrText], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function csvEscape(value) {
  const text = String(value ?? '');
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportHtmlTableAsCsv(tableId, filename) {
  const table = byId(tableId);
  if (!table || table.classList.contains('hidden')) return;

  const headers = Array.from(table.querySelectorAll('thead th')).map((cell) => cell.textContent.trim());
  const rows = Array.from(table.querySelectorAll('tbody tr')).map((row) => (
    Array.from(row.querySelectorAll('td')).map((cell) => cell.textContent.trim())
  ));
  if (headers.length === 0 || rows.length === 0) return;

  const csv = [headers, ...rows].map((row) => row.map(csvEscape).join(',')).join('\n') + '\n';
  downloadBlob(csv, filename, 'text/csv;charset=utf-8');
}

function exportCanvasAsPng(canvasId, filename) {
  const canvas = byId(canvasId);
  if (!canvas) return;
  canvas.toBlob((blob) => {
    if (!blob) return;
    downloadBlob(blob, filename, 'image/png');
  }, 'image/png');
}

function setSelectMessage(select, message) {
  select.innerHTML = '';
  const option = document.createElement('option');
  option.textContent = message;
  option.value = '';
  select.appendChild(option);
}

function setBadge(text, className = null) {
  const badge = byId('statusBadge');
  badge.classList.remove('ready', 'error');
  if (className) badge.classList.add(className);
  badge.textContent = text;
}

function setPickerSelection(prefix, key, token, message = null) {
  state[key] = token;
  const idInput = byId(`${prefix}Id`);
  const selected = byId(`${prefix}Selected`);

  if (token) {
    idInput.value = String(token.token_id);
    selected.textContent = `Selected: ${tokenLabel(token)}`;
  } else {
    idInput.value = '';
    selected.textContent = message || 'No token selected.';
  }

  if (key === 'tokenA' || key === 'tokenB') {
    resetAngleOutput();
    resetCommonCloseOutput('Token selection changed. Click Find shared close tokens to refresh this list.');
  } else if (key === 'anchor') {
    resetNeighborhoodOutput();
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = formatErrorDetail(data.detail || data);
    } catch (_) {
      // Keep response.statusText.
    }
    throw new Error(detail);
  }
  return response.json();
}

function updateStatusCard(status) {
  const loaded = Boolean(status.loaded);
  setBadge(loaded ? 'Model ready' : 'No model loaded', loaded ? 'ready' : null);
  byId('statusModel').textContent = status.model_name;
  byId('statusDevice').textContent = `${status.effective_device} (requested ${status.requested_device})`;
  byId('statusVocab').textContent = loaded ? status.vocab_size.toLocaleString() : '—';
  byId('statusHidden').textContent = loaded ? status.hidden_dim.toLocaleString() : '—';
  byId('modelNameInput').value = status.model_name;
  byId('modelDeviceInput').value = status.requested_device;
}


function formatBytes(bytes) {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const precision = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}

async function loadAvailableModels() {
  const select = byId('localModelSelect');
  const refreshButton = byId('refreshLocalModelsButton');
  const currentInput = byId('modelNameInput').value.trim();

  select.disabled = true;
  refreshButton.disabled = true;
  setSelectMessage(select, 'Scanning local Hugging Face cache…');

  try {
    const data = await fetchJson('/api/models/available');
    select.innerHTML = '';

    if (!data.models || data.models.length === 0) {
      setSelectMessage(select, 'No locally cached models found');
      return;
    }

    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = `Select a cached model… (${data.models.length})`;
    select.appendChild(placeholder);

    for (const model of data.models) {
      const option = document.createElement('option');
      option.value = model.model_name;
      const size = formatBytes(model.size_bytes);
      option.textContent = size ? `${model.model_name} — ${size}` : model.model_name;
      if (model.model_name === currentInput) option.selected = true;
      select.appendChild(option);
    }
  } catch (error) {
    setSelectMessage(select, `Cache scan failed: ${error.message}`);
  } finally {
    select.disabled = false;
    refreshButton.disabled = false;
  }
}

async function loadStatus() {
  setBadge('Checking status…');
  try {
    const status = await fetchJson('/api/status');
    updateStatusCard(status);
    byId('modelLoadMessage').textContent = status.loaded
      ? 'Active model loaded.'
      : 'No model loaded yet. Enter a Hugging Face model ID, choose a cached model, then click Load model.';
  } catch (error) {
    setBadge('Status failed', 'error');
    byId('modelLoadMessage').textContent = `Status failed: ${error.message}`;
  }
}

function resetPicker(prefix, key) {
  state[key] = null;
  cache[prefix] = [];
  searchRequestIds[prefix] += 1;
  lastSearchedQuery[prefix] = null;

  const select = byId(`${prefix}Results`);
  setSelectMessage(select, 'Click Search to load matches');
  select.disabled = false;
  select.classList.add('stale');

  const searchButton = byId(`${prefix}Search`);
  searchButton.disabled = false;
  searchButton.textContent = 'Search';
  searchButton.classList.add('needs-search');

  byId(`${prefix}Id`).value = '';
  byId(`${prefix}Selected`).textContent = prefix === 'anchor' ? 'No anchor selected.' : 'No token selected.';
}

function resetCommonCloseOutput(message = 'Set a max angle, then click Find shared close tokens.') {
  const output = byId('commonCloseOutput');
  if (!output) return;
  state.commonCloseRows = [];
  setOutput('commonCloseOutput', message, true);
  byId('commonCloseCount').textContent = 'No results yet';
  const table = byId('commonCloseTable');
  table.classList.add('hidden');
  table.querySelector('tbody').innerHTML = '';
  setExportVisible('exportCommonCloseCsv', false);
}

function resetAngleOutput() {
  state.angleSummary = null;
  setOutput('angleOutput', 'Choose two tokens, then compute their angle.', true);
  setExportVisible('exportAngleSummaryCsv', false);
}

function resetNeighborhoodOutput() {
  state.neighborhoodRows = [];
  setOutput('neighborhoodOutput', 'Choose an anchor token, then compute its closest tokens.', true);
  byId('neighborhoodTable').classList.add('hidden');
  byId('neighborhoodTable').querySelector('tbody').innerHTML = '';
  byId('downloadCsv').classList.add('hidden');
  setExportVisible('exportNeighborhoodCsv', false);
}

function resetAllSelectionsAfterModelChange() {
  resetPicker('tokenA', 'tokenA');
  resetPicker('tokenB', 'tokenB');
  resetPicker('anchor', 'anchor');

  resetAngleOutput();
  resetCommonCloseOutput();
  resetNeighborhoodOutput();
  resetPairwiseBinsOutput();
  resetMinDistancesOutput();
}

async function loadRequestedModel() {
  const modelName = byId('modelNameInput').value.trim();
  const device = byId('modelDeviceInput').value.trim() || 'cpu';
  const allowDownload = byId('allowDownloadInput').checked;
  const button = byId('loadModelButton');
  const message = byId('modelLoadMessage');

  if (!modelName) {
    message.textContent = 'Enter a Hugging Face model ID before loading.';
    setBadge('Model ID required', 'error');
    return;
  }

  button.disabled = true;
  button.textContent = 'Loading…';
  setBadge('Loading model…');
  message.textContent = allowDownload
    ? `Loading ${modelName} on ${device}; downloading missing files if needed…`
    : `Loading ${modelName} on ${device} from the local cache only…`;

  try {
    const status = await fetchJson('/api/model/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName, device, force_reload: false, allow_download: allowDownload }),
    });
    updateStatusCard(status);
    resetAllSelectionsAfterModelChange();
    message.textContent = `Loaded ${status.model_name}.`;
    loadAvailableModels();
    setOutput('angleOutput', 'Choose two tokens, then compute their angle.', true);
    setOutput('neighborhoodOutput', 'Choose an anchor token, then compute its closest tokens.', true);
  } catch (error) {
    setBadge('Model load failed', 'error');
    message.textContent = `Model load failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = 'Load model';
  }
}

async function search(prefix, key) {
  const requestId = ++searchRequestIds[prefix];
  const query = byId(`${prefix}Query`).value;
  const select = byId(`${prefix}Results`);
  const searchButton = byId(`${prefix}Search`);
  select.disabled = true;
  searchButton.disabled = true;
  searchButton.textContent = 'Searching…';
  setSelectMessage(select, 'Searching…');

  try {
    const params = new URLSearchParams({ q: query });
    const data = await fetchJson(`/api/tokens/search?${params.toString()}`);

    // Ignore stale responses if another explicit search/model load started while this request was in flight.
    if (requestId !== searchRequestIds[prefix]) return;

    lastSearchedQuery[prefix] = query;
    searchButton.classList.remove('needs-search');
    searchButton.textContent = 'Search';
    select.classList.remove('stale');

    cache[prefix] = data.results;
    select.innerHTML = '';

    if (data.results.length === 0) {
      setSelectMessage(select, 'No matches');
      setPickerSelection(prefix, key, null, 'No matching tokens found.');
      markSearchStale(prefix, key);
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const token of data.results) {
      const option = document.createElement('option');
      option.value = String(token.token_id);
      option.textContent = tokenLabel(token);
      fragment.appendChild(option);
    }
    select.appendChild(fragment);

    // Match Streamlit's selectbox behavior: when matches exist, one token is
    // selected immediately. Preserve the previous selection if it is still in
    // the visible result set; otherwise choose the first match.
    const previousId = state[key]?.token_id;
    const selected = data.results.find((token) => token.token_id === previousId) || data.results[0];
    select.value = String(selected.token_id);
    setPickerSelection(prefix, key, selected);
    markSearchStale(prefix, key);
  } catch (error) {
    if (requestId !== searchRequestIds[prefix]) return;
    cache[prefix] = [];
    setSelectMessage(select, `Error: ${error.message}`);
    setPickerSelection(prefix, key, null, `Search failed: ${error.message}`);
  } finally {
    if (requestId === searchRequestIds[prefix]) {
      select.disabled = false;
      searchButton.disabled = false;
      searchButton.textContent = searchButton.classList.contains('needs-search') ? 'Search updated query' : 'Search';
    }
  }
}

async function selectFromId(prefix, key) {
  const rawValue = byId(`${prefix}Id`).value;
  const tokenId = Number(rawValue);
  if (!Number.isInteger(tokenId) || tokenId < 0) {
    setPickerSelection(prefix, key, null, 'Enter a non-negative integer token ID.');
    return;
  }

  try {
    const token = await fetchJson(`/api/tokens/id/${tokenId}`);
    setPickerSelection(prefix, key, token);
  } catch (error) {
    setPickerSelection(prefix, key, null, `Token lookup failed: ${error.message}`);
  }
}

function selectFromDropdown(prefix, key) {
  const select = byId(`${prefix}Results`);
  const selectedId = Number(select.value);
  const token = cache[prefix].find((candidate) => candidate.token_id === selectedId);
  if (!token) return;
  setPickerSelection(prefix, key, token);
}

async function ensurePickerSelection(prefix, key) {
  // Make action buttons forgiving: if a user typed an ID but did not click
  // "Use ID", or if the browser still has a selected dropdown value while
  // state was reset, resolve the token before running the action.
  if (state[key]) return state[key];

  const idInput = byId(`${prefix}Id`);
  const rawId = idInput.value.trim();
  if (rawId !== '') {
    const tokenId = Number(rawId);
    if (Number.isInteger(tokenId) && tokenId >= 0) {
      const token = await fetchJson(`/api/tokens/id/${tokenId}`);
      setPickerSelection(prefix, key, token);
      return token;
    }
  }

  const select = byId(`${prefix}Results`);
  const selectedId = Number(select.value);
  if (Number.isInteger(selectedId) && selectedId >= 0) {
    let token = cache[prefix].find((candidate) => candidate.token_id === selectedId);
    if (!token) token = await fetchJson(`/api/tokens/id/${selectedId}`);
    setPickerSelection(prefix, key, token);
    return token;
  }

  return null;
}

async function ensurePairwiseSelections() {
  const tokenA = await ensurePickerSelection('tokenA', 'tokenA');
  const tokenB = await ensurePickerSelection('tokenB', 'tokenB');
  return { tokenA, tokenB };
}

function markSearchStale(prefix, key) {
  // Query edits should not issue network requests. They only mark the picker
  // as needing an explicit Search click.
  const searchButton = byId(`${prefix}Search`);
  const select = byId(`${prefix}Results`);
  const query = byId(`${prefix}Query`).value;
  searchButton.disabled = false;

  if (lastSearchedQuery[prefix] === null) {
    searchButton.classList.add('needs-search');
    searchButton.textContent = 'Search';
    setSelectMessage(select, 'Click Search to load matches');
    select.classList.add('stale');
    return;
  }

  if (query !== lastSearchedQuery[prefix]) {
    searchButton.classList.add('needs-search');
    searchButton.textContent = 'Search updated query';
    select.classList.add('stale');
  } else {
    searchButton.classList.remove('needs-search');
    searchButton.textContent = 'Search';
    select.classList.remove('stale');
  }
}

function setupPicker(prefix, key) {
  const queryInput = byId(`${prefix}Query`);
  const searchButton = byId(`${prefix}Search`);
  const results = byId(`${prefix}Results`);

  setSelectMessage(results, 'Click Search to load matches');
  results.classList.add('stale');
  searchButton.classList.add('needs-search');

  queryInput.addEventListener('input', () => markSearchStale(prefix, key));
  queryInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      search(prefix, key);
    }
  });
  searchButton.addEventListener('click', () => search(prefix, key));
  byId(`${prefix}Results`).addEventListener('change', () => selectFromDropdown(prefix, key));
  byId(`${prefix}UseId`).addEventListener('click', () => selectFromId(prefix, key));
}

async function computeAngle() {
  let tokenA;
  let tokenB;
  try {
    ({ tokenA, tokenB } = await ensurePairwiseSelections());
  } catch (error) {
    setOutput('angleOutput', `<strong>Token selection failed:</strong> ${escapeHtml(error.message)}`);
    return;
  }

  if (!tokenA || !tokenB) {
    setOutput('angleOutput', 'Select Token A and Token B first. Search/select a token or enter an ID and click Use ID.', true);
    setExportVisible('exportAngleSummaryCsv', false);
    return;
  }

  state.angleSummary = null;
  setExportVisible('exportAngleSummaryCsv', false);
  setOutput('angleOutput', 'Computing angle…', true);
  try {
    const data = await fetchJson(`/api/angle?token_a=${tokenA.token_id}&token_b=${tokenB.token_id}`);
    state.angleSummary = data;
    setOutput('angleOutput', `
      <div class="metric"><span>Angle</span><strong>${data.angle_deg.toFixed(6)}°</strong></div>
      <table id="angleSummaryTable" class="mini-table">
        <tbody>
          <tr><th>Token A</th><td>${escapeHtml(tokenLabel({ token_id: data.token_a_id, raw: data.token_a_raw, display: data.token_a_display }))}</td></tr>
          <tr><th>Token A vector length</th><td>${data.token_a_magnitude.toFixed(6)}</td></tr>
          <tr><th>Token B</th><td>${escapeHtml(tokenLabel({ token_id: data.token_b_id, raw: data.token_b_raw, display: data.token_b_display }))}</td></tr>
          <tr><th>Token B vector length</th><td>${data.token_b_magnitude.toFixed(6)}</td></tr>
          <tr><th>Angle °</th><td>${data.angle_deg.toFixed(6)}</td></tr>
        </tbody>
      </table>
    `);
    setExportVisible('exportAngleSummaryCsv', true);
  } catch (error) {
    state.angleSummary = null;
    setExportVisible('exportAngleSummaryCsv', false);
    setOutput('angleOutput', `<strong>Angle failed:</strong> ${escapeHtml(error.message)}`);
  }
}

function renderCommonCloseTable(rows) {
  const table = byId('commonCloseTable');
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';

  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.token_id}</td>
      <td><code>${escapeHtml(row.token_raw)}</code></td>
      <td><code>${escapeHtml(row.token_display)}</code></td>
      <td>${Number(row.angle_to_token_a_deg).toFixed(6)}</td>
      <td>${Number(row.angle_to_token_b_deg).toFixed(6)}</td>
      <td>${Number(row.magnitude).toFixed(6)}</td>
    `;
    fragment.appendChild(tr);
  }

  tbody.appendChild(fragment);
  table.classList.toggle('hidden', rows.length === 0);
}

async function computeCommonCloseTokens() {
  let tokenA;
  let tokenB;
  try {
    ({ tokenA, tokenB } = await ensurePairwiseSelections());
  } catch (error) {
    byId('commonCloseCount').textContent = 'Selection failed';
    setOutput('commonCloseOutput', `<strong>Token selection failed:</strong> ${escapeHtml(error.message)}`);
    return;
  }

  if (!tokenA || !tokenB) {
    setOutput('commonCloseOutput', 'Select Token A and Token B first. Search/select a token or enter an ID and click Use ID.', true);
    setExportVisible('exportCommonCloseCsv', false);
    return;
  }

  const threshold = Number(byId('commonCloseThreshold').value || 35);
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 180) {
    setOutput('commonCloseOutput', 'Enter a max angle from 0 to 180 degrees.', true);
    return;
  }

  const button = byId('commonCloseButton');
  state.commonCloseRows = [];
  byId('commonCloseTable').classList.add('hidden');
  byId('commonCloseTable').querySelector('tbody').innerHTML = '';
  setExportVisible('exportCommonCloseCsv', false);
  byId('commonCloseCount').textContent = 'Computing…';
  setOutput('commonCloseOutput', `Finding tokens within ${threshold}° of both selected tokens…`, true);
  button.disabled = true;
  button.textContent = 'Finding…';

  try {
    const params = new URLSearchParams({
      token_a: String(tokenA.token_id),
      token_b: String(tokenB.token_id),
      max_angle_deg: String(threshold),
    });
    const data = await fetchJson(`/api/common-close-tokens?${params.toString()}`);
    state.commonCloseRows = data.rows || [];
    byId('commonCloseCount').textContent = `${formatCount(data.match_count)} matches`;
    setOutput('commonCloseOutput', `
      <div class="metric"><span>Tokens within ${Number(data.threshold_deg).toFixed(3)}° of both</span><strong>${formatCount(data.match_count)}</strong></div>
      <table class="mini-table">
        <tbody>
          <tr><th>Token A</th><td>${escapeHtml(tokenLabel({ token_id: data.token_a_id, raw: data.token_a_raw, display: data.token_a_display }))}</td></tr>
          <tr><th>Token A vector length</th><td>${Number(data.token_a_magnitude).toFixed(6)}</td></tr>
          <tr><th>Token B</th><td>${escapeHtml(tokenLabel({ token_id: data.token_b_id, raw: data.token_b_raw, display: data.token_b_display }))}</td></tr>
          <tr><th>Token B vector length</th><td>${Number(data.token_b_magnitude).toFixed(6)}</td></tr>
          <tr><th>Sort order</th><td>Worst of the two angles, then angle sum, then token ID</td></tr>
        </tbody>
      </table>
    `);
    renderCommonCloseTable(state.commonCloseRows);
    setExportVisible('exportCommonCloseCsv', state.commonCloseRows.length > 0);
  } catch (error) {
    state.commonCloseRows = [];
    setExportVisible('exportCommonCloseCsv', false);
    byId('commonCloseCount').textContent = 'Failed';
    setOutput('commonCloseOutput', `<strong>Shared-close search failed:</strong> ${escapeHtml(error.message)}`);
  } finally {
    button.disabled = false;
    button.textContent = 'Find shared close tokens';
  }
}

async function computeNeighborhood() {
  if (!state.anchor) {
    setOutput('neighborhoodOutput', 'Select an anchor token first.', true);
    setExportVisible('exportNeighborhoodCsv', false);
    return;
  }

  const limit = Number(byId('neighborhoodLimit').value || 500);
  state.neighborhoodRows = [];
  setOutput('neighborhoodOutput', 'Computing neighborhood…', true);
  byId('neighborhoodTable').classList.add('hidden');
  byId('downloadCsv').classList.add('hidden');
  setExportVisible('exportNeighborhoodCsv', false);

  try {
    const data = await fetchJson(`/api/neighborhood?anchor_id=${state.anchor.token_id}&limit=${limit}`);
    state.neighborhoodRows = data.rows || [];
    setOutput('neighborhoodOutput', `
      <div class="metric"><span>Anchor vector length</span><strong>${data.anchor_magnitude.toFixed(6)}</strong></div>
      <p>Anchor: ${escapeHtml(tokenLabel({ token_id: data.anchor_id, raw: data.anchor_raw, display: data.anchor_display }))}</p>
    `);

    const tbody = byId('neighborhoodTable').querySelector('tbody');
    tbody.innerHTML = '';
    for (const row of state.neighborhoodRows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${row.rank}</td>
        <td>${row.token_id}</td>
        <td><code>${escapeHtml(row.token_raw)}</code></td>
        <td><code>${escapeHtml(row.token_display)}</code></td>
        <td>${row.angle_deg.toFixed(6)}</td>
        <td>${row.magnitude.toFixed(6)}</td>
      `;
      tbody.appendChild(tr);
    }

    const table = byId('neighborhoodTable');
    table.classList.remove('hidden');

    const download = byId('downloadCsv');
    download.href = `/api/neighborhood.csv?anchor_id=${state.anchor.token_id}`;
    download.classList.remove('hidden');
    setExportVisible('exportNeighborhoodCsv', state.neighborhoodRows.length > 0);
  } catch (error) {
    state.neighborhoodRows = [];
    setExportVisible('exportNeighborhoodCsv', false);
    setOutput('neighborhoodOutput', `<strong>Neighborhood failed:</strong> ${escapeHtml(error.message)}`);
  }
}


function resetPairwiseBinTokensOutput() {
  state.pairwiseSelectedBinIndex = null;
  state.pairwiseBinTokens = [];
  const panel = byId('pairwiseBinTokensPanel');
  if (!panel) return;
  panel.classList.add('hidden');
  byId('pairwiseBinTokensCount').textContent = 'No bin selected';
  setOutput('pairwiseBinTokensOutput', 'Click an angle-bin table row to show the unique tokens that participate in that bin.', true);
  const table = byId('pairwiseBinTokensTable');
  table.classList.add('hidden');
  table.querySelector('tbody').innerHTML = '';
  setExportVisible('exportPairwiseBinTokensCsv', false);
  for (const row of byId('pairwiseBinsTable').querySelectorAll('tbody tr')) {
    row.classList.remove('selected-row');
  }
}

function resetPairwiseBinsOutput() {
  const output = byId('pairwiseBinsOutput');
  if (!output) return;
  setOutput('pairwiseBinsOutput', 'Load a model, then compute the global pairwise angle-bin distribution. A CUDA device is used automatically when available.', true);
  state.pairwiseBins = null;
  state.pairwiseSelectedBinIndex = null;
  byId('pairwisePlotCard').classList.add('hidden');
  byId('pairwiseBinsTable').classList.add('hidden');
  byId('pairwiseBinsTable').querySelector('tbody').innerHTML = '';
  setExportVisible('exportPairwiseBinsCsv', false);
  resetPairwiseBinTokensOutput();
}

function formatCount(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toLocaleString() : String(value);
}

function niceCeil(value) {
  const safeValue = Math.max(1, Number(value || 0));
  const exponent = Math.floor(Math.log10(safeValue));
  const magnitude = 10 ** exponent;
  const normalized = safeValue / magnitude;
  let niceNormalized;
  if (normalized <= 1) niceNormalized = 1;
  else if (normalized <= 2) niceNormalized = 2;
  else if (normalized <= 5) niceNormalized = 5;
  else niceNormalized = 10;
  return niceNormalized * magnitude;
}

function buildLinearTicks(maxCount) {
  const maxNice = niceCeil(maxCount);
  const tickCount = 6;
  const ticks = [];
  for (let i = 0; i < tickCount; i += 1) {
    ticks.push({
      value: (maxNice * i) / (tickCount - 1),
      label: Math.round((maxNice * i) / (tickCount - 1)).toLocaleString(),
    });
  }
  return { ticks, yMax: maxNice };
}

function buildLogTicks(maxCount) {
  const maxPower = Math.max(0, Math.ceil(Math.log10(Math.max(1, maxCount))));
  const step = maxPower > 10 ? 2 : 1;
  const ticks = [];
  for (let power = 0; power <= maxPower; power += step) {
    ticks.push({
      value: power,
      label: Math.round(10 ** power).toLocaleString(),
    });
  }
  if (ticks[ticks.length - 1].value !== maxPower) {
    ticks.push({
      value: maxPower,
      label: Math.round(10 ** maxPower).toLocaleString(),
    });
  }
  return { ticks, yMax: Math.max(1, maxPower) };
}

function drawPairwiseAngleBinPlot(bins, logScale = true) {
  const canvas = byId('pairwiseAnglePlot');
  const card = byId('pairwisePlotCard');
  const ctx = canvas.getContext('2d');
  const style = getComputedStyle(document.documentElement);
  const textColor = style.getPropertyValue('--text').trim() || '#e5e7eb';
  const mutedColor = style.getPropertyValue('--muted').trim() || '#9ca3af';
  const borderColor = style.getPropertyValue('--border').trim() || '#374151';
  const accentColor = style.getPropertyValue('--accent-strong').trim() || '#60a5fa';

  const angleBins = [...bins].sort((a, b) => Number(a.angle_min_deg) - Number(b.angle_min_deg));
  const cssWidth = Math.max(980, card.clientWidth || 1100);
  const cssHeight = 460;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const counts = angleBins.map((bin) => Number(bin.count || 0));
  const maxCount = Math.max(1, ...counts);
  const yScale = logScale ? buildLogTicks(maxCount) : buildLinearTicks(maxCount);

  ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
  const maxTickLabelWidth = Math.max(0, ...yScale.ticks.map((tick) => ctx.measureText(tick.label).width));
  const margin = {
    top: 28,
    right: 32,
    bottom: 94,
    left: Math.ceil(Math.max(150, maxTickLabelWidth + 64)),
  };
  const width = cssWidth - margin.left - margin.right;
  const height = cssHeight - margin.top - margin.bottom;
  const binCount = Math.max(1, angleBins.length);

  function xForIndex(index) {
    if (binCount === 1) return margin.left + width / 2;
    return margin.left + (index / (binCount - 1)) * width;
  }

  function yValueForCount(count) {
    return logScale ? Math.log10(Math.max(1, count)) : count;
  }

  function yForValue(value) {
    return margin.top + ((yScale.yMax - value) / yScale.yMax) * height;
  }

  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + height);
  ctx.lineTo(margin.left + width, margin.top + height);
  ctx.stroke();

  ctx.fillStyle = mutedColor;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const tick of yScale.ticks) {
    const y = yForValue(tick.value);
    ctx.strokeStyle = borderColor;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + width, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(tick.label, margin.left - 12, y);
  }

  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  for (const [index, bin] of angleBins.entries()) {
    const x = xForIndex(index);
    ctx.save();
    ctx.translate(x, margin.top + height + 14);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(String(bin.label), 0, 0);
    ctx.restore();
  }

  ctx.save();
  ctx.translate(24, margin.top + height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(logScale ? 'Pair count, log scale' : 'Pair count', 0, 0);
  ctx.restore();

  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('Angle bin (degrees)', margin.left + width / 2, cssHeight - 18);

  ctx.fillStyle = accentColor;
  for (const [index, bin] of angleBins.entries()) {
    const x = xForIndex(index);
    const y = yForValue(yValueForCount(Number(bin.count || 0)));
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  const caption = byId('pairwisePlotCaption');
  if (caption) {
    caption.textContent = logScale
      ? 'Y axis uses log10(count). X axis is the 5° angle-bin range ordered from 0° to 90°.'
      : 'Y axis uses the raw pair count. X axis is the 5° angle-bin range ordered from 0° to 90°.';
  }
  const summary = byId('pairwiseYScaleSummary');
  if (summary) {
    summary.textContent = logScale ? 'log10(count)' : 'raw count';
  }
}

function markSelectedPairwiseBinRow(binIndex) {
  for (const row of byId('pairwiseBinsTable').querySelectorAll('tbody tr')) {
    row.classList.toggle('selected-row', Number(row.dataset.binIndex) === Number(binIndex));
  }
}

function renderPairwiseBinTokens(data) {
  const table = byId('pairwiseBinTokensTable');
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';

  const fragment = document.createDocumentFragment();
  const tokens = [...(data.tokens || [])].sort((a, b) => Number(a.token_id) - Number(b.token_id));
  state.pairwiseBinTokens = tokens;
  for (const token of tokens) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${token.token_id}</td>
      <td><code>${escapeHtml(tokenLabel(token))}</code></td>
      <td><code>${escapeHtml(token.raw)}</code></td>
      <td><code>${escapeHtml(token.display)}</code></td>
    `;
    fragment.appendChild(tr);
  }
  tbody.appendChild(fragment);
  table.classList.toggle('hidden', tokens.length === 0);
  setExportVisible('exportPairwiseBinTokensCsv', tokens.length > 0);
}

async function loadPairwiseBinTokens(bin) {
  const panel = byId('pairwiseBinTokensPanel');
  panel.classList.remove('hidden');
  state.pairwiseSelectedBinIndex = Number(bin.bin_index);
  markSelectedPairwiseBinRow(bin.bin_index);

  byId('pairwiseBinTokensTable').classList.add('hidden');
  byId('pairwiseBinTokensTable').querySelector('tbody').innerHTML = '';
  state.pairwiseBinTokens = [];
  setExportVisible('exportPairwiseBinTokensCsv', false);
  byId('pairwiseBinTokensCount').textContent = `${escapeHtml(bin.label)} selected`;
  setOutput('pairwiseBinTokensOutput', `Loading tokens for ${escapeHtml(bin.label)}…`, true);

  try {
    const data = await fetchJson(`/api/pairwise-angle-bins/${bin.bin_index}/tokens`);
    byId('pairwiseBinTokensCount').textContent = `${formatCount(data.token_count)} unique tokens`;
    setOutput('pairwiseBinTokensOutput', `
      Showing unique tokens that participate in at least one pair in the <strong>${escapeHtml(data.label)}</strong> bin.
      Tokens are sorted by token ID and use the same label format as token search.
    `);
    renderPairwiseBinTokens(data);
  } catch (error) {
    byId('pairwiseBinTokensCount').textContent = 'Load failed';
    setOutput('pairwiseBinTokensOutput', `<strong>Token list failed:</strong> ${escapeHtml(error.message)}`);
  }
}

function renderPairwiseBinsTable(bins) {
  const table = byId('pairwiseBinsTable');
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';
  for (const bin of bins) {
    const tr = document.createElement('tr');
    tr.classList.add('clickable-row');
    tr.tabIndex = 0;
    tr.dataset.binIndex = String(bin.bin_index);
    tr.title = `Show tokens in ${bin.label}`;
    tr.innerHTML = `
      <td>${bin.rank}</td>
      <td>${escapeHtml(bin.label)}</td>
      <td>${bin.bin_index}</td>
      <td>${formatCount(bin.count)}</td>
      <td>${formatCount(bin.token_count || 0)}</td>
    `;
    tr.addEventListener('click', () => loadPairwiseBinTokens(bin));
    tr.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        loadPairwiseBinTokens(bin);
      }
    });
    tbody.appendChild(tr);
  }
  table.classList.remove('hidden');
  setExportVisible('exportPairwiseBinsCsv', bins.length > 0);
}

async function computePairwiseAngleBins() {
  const button = byId('pairwiseBinsButton');
  const blockSize = Number(byId('pairwiseBlockSize').value || 2048);
  const computeDevice = byId('pairwiseComputeDevice').value.trim() || 'auto';
  const includeSelf = byId('pairwiseIncludeSelf').checked;
  const logScale = byId('pairwiseLogScale').checked;

  byId('pairwisePlotCard').classList.add('hidden');
  byId('pairwiseBinsTable').classList.add('hidden');
  setExportVisible('exportPairwiseBinsCsv', false);
  resetPairwiseBinTokensOutput();
  setOutput('pairwiseBinsOutput', 'Computing all-pairs angle bins block by block… The full pairwise matrix is not cached or written to disk.', true);
  button.disabled = true;
  button.textContent = 'Computing…';

  try {
    const params = new URLSearchParams({
      block_size: String(blockSize),
      compute_device: computeDevice,
      include_self: String(includeSelf),
    });
    const data = await fetchJson(`/api/pairwise-angle-bins?${params.toString()}`);
    setOutput('pairwiseBinsOutput', `
      <div class="metric"><span>Total pairs binned</span><strong>${formatCount(data.total_pairs)}</strong></div>
      <table class="mini-table">
        <tbody>
          <tr><th>Model</th><td>${escapeHtml(data.model_name)}</td></tr>
          <tr><th>Vocab × hidden</th><td>${formatCount(data.vocab_size)} × ${formatCount(data.hidden_dim)}</td></tr>
          <tr><th>Angle definition</th><td>Acute angle, abs(cosine), 0°–90°</td></tr>
          <tr><th>Bin width</th><td>${data.bin_degrees}°</td></tr>
          <tr><th>Compute device</th><td>${escapeHtml(data.compute_device)}</td></tr>
          <tr><th>Block size</th><td>${formatCount(data.block_size)}</td></tr>
          <tr><th>Self-pairs</th><td>${data.include_self ? 'included' : 'excluded'}</td></tr>
          <tr><th>Y scale</th><td id="pairwiseYScaleSummary">${logScale ? 'log10(count)' : 'raw count'}</td></tr>
          <tr><th>Elapsed</th><td>${Number(data.elapsed_seconds).toFixed(3)} seconds</td></tr>
        </tbody>
      </table>
    `);
    byId('pairwisePlotCard').classList.remove('hidden');
    state.pairwiseBins = data.bins;
    drawPairwiseAngleBinPlot(data.bins, logScale);
    renderPairwiseBinsTable(data.bins);
  } catch (error) {
    state.pairwiseBins = null;
    setExportVisible('exportPairwiseBinsCsv', false);
    setOutput('pairwiseBinsOutput', `<strong>Pairwise binning failed:</strong> ${escapeHtml(error.message)}`);
  } finally {
    button.disabled = false;
    button.textContent = 'Compute pairwise bins';
  }
}


function buildAngleTicks(maxAngle) {
  const yMax = Math.min(180, Math.max(1, Math.ceil(Number(maxAngle || 0) / 5) * 5));
  const tickCount = 6;
  const ticks = [];
  for (let i = 0; i < tickCount; i += 1) {
    const value = (yMax * i) / (tickCount - 1);
    ticks.push({ value, label: `${Number(value.toFixed(3)).toLocaleString()}°` });
  }
  return { ticks, yMax };
}

function drawMinDistancePlot(rows) {
  const canvas = byId('minDistancePlot');
  const card = byId('minDistancePlotCard');
  const ctx = canvas.getContext('2d');
  const style = getComputedStyle(document.documentElement);
  const textColor = style.getPropertyValue('--text').trim() || '#e5e7eb';
  const mutedColor = style.getPropertyValue('--muted').trim() || '#9ca3af';
  const borderColor = style.getPropertyValue('--border').trim() || '#374151';
  const accentColor = style.getPropertyValue('--accent-strong').trim() || '#60a5fa';

  const sortedRows = [...(rows || [])].sort((a, b) => (
    Number(a.min_angle_deg) - Number(b.min_angle_deg)
    || Number(a.token_id) - Number(b.token_id)
  ));
  const rowCount = sortedRows.length;
  if (rowCount === 0) {
    card.classList.add('hidden');
    return;
  }

  const ratio = window.devicePixelRatio || 1;
  const cssWidth = Math.max(1100, Math.floor(card.clientWidth || 1100) - 36);
  const cssHeight = 460;
  canvas.width = Math.floor(cssWidth * ratio);
  canvas.height = Math.floor(cssHeight * ratio);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const maxAngle = Math.max(1, ...sortedRows.map((row) => Number(row.min_angle_deg || 0)));
  const yScale = buildAngleTicks(maxAngle);

  ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
  const maxTickLabelWidth = Math.max(0, ...yScale.ticks.map((tick) => ctx.measureText(tick.label).width));
  const margin = {
    top: 28,
    right: 32,
    bottom: 62,
    left: Math.ceil(Math.max(132, maxTickLabelWidth + 72)),
  };
  const width = cssWidth - margin.left - margin.right;
  const height = cssHeight - margin.top - margin.bottom;

  function xForRank(rankIndex) {
    if (rowCount === 1) return margin.left + width / 2;
    return margin.left + (rankIndex / (rowCount - 1)) * width;
  }

  function yForAngle(angle) {
    return margin.top + ((yScale.yMax - angle) / yScale.yMax) * height;
  }

  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + height);
  ctx.lineTo(margin.left + width, margin.top + height);
  ctx.stroke();

  ctx.fillStyle = mutedColor;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const tick of yScale.ticks) {
    const y = yForAngle(tick.value);
    ctx.strokeStyle = borderColor;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + width, y);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(tick.label, margin.left - 12, y);
  }

  const xTicks = Array.from(new Set([
    1,
    Math.max(1, Math.round(rowCount / 4)),
    Math.max(1, Math.round(rowCount / 2)),
    Math.max(1, Math.round((rowCount * 3) / 4)),
    rowCount,
  ])).sort((a, b) => a - b);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const rank of xTicks) {
    const x = xForRank(rank - 1);
    ctx.fillText(rank.toLocaleString(), x, margin.top + height + 12);
  }

  ctx.save();
  ctx.translate(26, margin.top + height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('Minimum angle (degrees)', 0, 0);
  ctx.restore();

  ctx.fillStyle = textColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('Min-angle rank, lowest to highest', margin.left + width / 2, cssHeight - 18);

  ctx.fillStyle = accentColor;
  if (rowCount > 20000) {
    for (let index = 0; index < rowCount; index += 1) {
      const row = sortedRows[index];
      ctx.fillRect(xForRank(index), yForAngle(Number(row.min_angle_deg || 0)), 1, 1);
    }
  } else {
    const radius = rowCount > 5000 ? 1.75 : 3;
    for (let index = 0; index < rowCount; index += 1) {
      const row = sortedRows[index];
      ctx.beginPath();
      ctx.arc(xForRank(index), yForAngle(Number(row.min_angle_deg || 0)), radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  byId('minDistancePlotCaption').textContent = 'Y axis is the minimum non-self angle. X axis is rank after sorting tokens from lowest to highest minimum angle.';
}

function sortedMinDistanceRowsForTable() {
  const sortMode = byId('minDistanceSort').value;
  const rows = [...state.minDistanceRows];
  if (sortMode === 'min_angle') {
    rows.sort((a, b) => (
      Number(a.min_angle_deg) - Number(b.min_angle_deg)
      || Number(a.token_id) - Number(b.token_id)
    ));
  } else {
    rows.sort((a, b) => Number(a.token_id) - Number(b.token_id));
  }
  return rows;
}

function renderMinDistanceTable() {
  const table = byId('minDistancesTable');
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';

  const rows = sortedMinDistanceRowsForTable();
  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.min_angle_rank}</td>
      <td>${row.token_id}</td>
      <td><code>${escapeHtml(row.token_raw)}</code></td>
      <td><code>${escapeHtml(row.token_display)}</code></td>
      <td>${Number(row.magnitude).toFixed(6)}</td>
      <td>${Number(row.min_angle_deg).toFixed(6)}</td>
      <td>${row.other_token_id}</td>
      <td><code>${escapeHtml(row.other_token_raw)}</code></td>
      <td><code>${escapeHtml(row.other_token_display)}</code></td>
      <td>${Number(row.other_magnitude).toFixed(6)}</td>
    `;
    fragment.appendChild(tr);
  }
  tbody.appendChild(fragment);
  table.classList.toggle('hidden', rows.length === 0);
  setExportVisible('exportMinDistancesCsv', rows.length > 0);
}

function resetMinDistancesOutput() {
  state.minDistanceRows = [];
  const output = byId('minDistancesOutput');
  if (!output) return;
  setOutput('minDistancesOutput', 'Load a model, then compute each token’s closest non-self angular neighbor. CUDA is used automatically when available.', true);
  byId('minDistancePlotCard').classList.add('hidden');
  const table = byId('minDistancesTable');
  table.classList.add('hidden');
  table.querySelector('tbody').innerHTML = '';
  setExportVisible('exportMinDistancesCsv', false);
}

async function computeMinAngularDistances() {
  const button = byId('minDistancesButton');
  const blockSize = Number(byId('minDistanceBlockSize').value || 2048);
  const computeDevice = byId('minDistanceComputeDevice').value.trim() || 'auto';

  state.minDistanceRows = [];
  byId('minDistancePlotCard').classList.add('hidden');
  byId('minDistancesTable').classList.add('hidden');
  byId('minDistancesTable').querySelector('tbody').innerHTML = '';
  setExportVisible('exportMinDistancesCsv', false);
  setOutput('minDistancesOutput', 'Computing closest non-self angular neighbor for every token… The full pairwise matrix is not cached or written to disk.', true);
  button.disabled = true;
  button.textContent = 'Computing…';

  try {
    const params = new URLSearchParams({
      block_size: String(blockSize),
      compute_device: computeDevice,
    });
    const data = await fetchJson(`/api/min-angular-distances?${params.toString()}`);
    state.minDistanceRows = data.rows || [];
    setOutput('minDistancesOutput', `
      <div class="metric"><span>Tokens analyzed</span><strong>${formatCount(data.vocab_size)}</strong></div>
      <table class="mini-table">
        <tbody>
          <tr><th>Model</th><td>${escapeHtml(data.model_name)}</td></tr>
          <tr><th>Vocab × hidden</th><td>${formatCount(data.vocab_size)} × ${formatCount(data.hidden_dim)}</td></tr>
          <tr><th>Pairs considered</th><td>${formatCount(data.total_pairs)}</td></tr>
          <tr><th>Angle definition</th><td>Signed vector angle, self-distance excluded</td></tr>
          <tr><th>Compute device</th><td>${escapeHtml(data.compute_device)}</td></tr>
          <tr><th>Block size</th><td>${formatCount(data.block_size)}</td></tr>
          <tr><th>Elapsed</th><td>${Number(data.elapsed_seconds).toFixed(3)} seconds</td></tr>
        </tbody>
      </table>
    `);
    byId('minDistancePlotCard').classList.remove('hidden');
    drawMinDistancePlot(state.minDistanceRows);
    renderMinDistanceTable();
  } catch (error) {
    state.minDistanceRows = [];
    setExportVisible('exportMinDistancesCsv', false);
    setOutput('minDistancesOutput', `<strong>Minimum-distance computation failed:</strong> ${escapeHtml(error.message)}`);
  } finally {
    button.disabled = false;
    button.textContent = 'Compute minimum distances';
  }
}

function exportAngleSummaryCsv() {
  const data = state.angleSummary;
  if (!data) return;
  const rows = [
    ['field', 'value'],
    ['token_a_id', data.token_a_id],
    ['token_a_raw', data.token_a_raw],
    ['token_a_display', data.token_a_display],
    ['token_a_magnitude', data.token_a_magnitude],
    ['token_b_id', data.token_b_id],
    ['token_b_raw', data.token_b_raw],
    ['token_b_display', data.token_b_display],
    ['token_b_magnitude', data.token_b_magnitude],
    ['angle_deg', data.angle_deg],
  ];
  const csv = rows.map((row) => row.map(csvEscape).join(',')).join('\n') + '\n';
  downloadBlob(csv, `angle_summary_${data.token_a_id}_${data.token_b_id}.csv`, 'text/csv;charset=utf-8');
}

function setupExportButtons() {
  byId('exportAngleSummaryCsv').addEventListener('click', exportAngleSummaryCsv);
  byId('exportCommonCloseCsv').addEventListener('click', () => exportHtmlTableAsCsv('commonCloseTable', 'shared_close_tokens.csv'));
  byId('exportNeighborhoodCsv').addEventListener('click', () => exportHtmlTableAsCsv('neighborhoodTable', 'visible_neighborhood.csv'));
  byId('exportPairwisePlotPng').addEventListener('click', () => exportCanvasAsPng('pairwiseAnglePlot', 'pairwise_angle_bins.png'));
  byId('exportPairwiseBinsCsv').addEventListener('click', () => exportHtmlTableAsCsv('pairwiseBinsTable', 'pairwise_angle_bins.csv'));
  byId('exportPairwiseBinTokensCsv').addEventListener('click', () => exportHtmlTableAsCsv('pairwiseBinTokensTable', `pairwise_bin_${filenamePart(state.pairwiseSelectedBinIndex, 'selected')}_tokens.csv`));
  byId('exportMinDistancePlotPng').addEventListener('click', () => exportCanvasAsPng('minDistancePlot', 'minimum_angular_distances.png'));
  byId('exportMinDistancesCsv').addEventListener('click', () => exportHtmlTableAsCsv('minDistancesTable', 'minimum_angular_distances.csv'));
}

function setupModelControls() {
  const loadButton = byId('loadModelButton');
  const modelInput = byId('modelNameInput');
  const deviceInput = byId('modelDeviceInput');
  const localModelSelect = byId('localModelSelect');
  const refreshLocalModelsButton = byId('refreshLocalModelsButton');

  loadButton.addEventListener('click', loadRequestedModel);
  localModelSelect.addEventListener('change', () => {
    if (localModelSelect.value) modelInput.value = localModelSelect.value;
  });
  refreshLocalModelsButton.addEventListener('click', loadAvailableModels);
  for (const input of [modelInput, deviceInput]) {
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        loadRequestedModel();
      }
    });
  }
}

window.addEventListener('DOMContentLoaded', () => {
  setupModelControls();
  setupExportButtons();
  loadStatus();
  loadAvailableModels();
  setupPicker('tokenA', 'tokenA');
  setupPicker('tokenB', 'tokenB');
  setupPicker('anchor', 'anchor');
  byId('angleButton').addEventListener('click', computeAngle);
  byId('commonCloseButton').addEventListener('click', computeCommonCloseTokens);
  byId('commonCloseThreshold').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      computeCommonCloseTokens();
    }
  });
  byId('commonCloseThreshold').addEventListener('input', () => {
    resetCommonCloseOutput('Threshold changed. Click Find shared close tokens to refresh this list.');
  });
  byId('neighborhoodButton').addEventListener('click', computeNeighborhood);
  byId('pairwiseBinsButton').addEventListener('click', computePairwiseAngleBins);
  byId('minDistancesButton').addEventListener('click', computeMinAngularDistances);
  byId('minDistanceSort').addEventListener('change', () => {
    if (state.minDistanceRows.length > 0) renderMinDistanceTable();
  });
  byId('pairwiseLogScale').addEventListener('change', () => {
    if (state.pairwiseBins && !byId('pairwisePlotCard').classList.contains('hidden')) {
      drawPairwiseAngleBinPlot(state.pairwiseBins, byId('pairwiseLogScale').checked);
    }
  });
});
