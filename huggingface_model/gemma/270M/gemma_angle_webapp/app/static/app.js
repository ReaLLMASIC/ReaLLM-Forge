const state = {
  tokenA: null,
  tokenB: null,
  anchor: null,
  groupSeed: null,
  angleSummary: null,
  commonCloseRows: [],
  neighborhoodRows: [],
  pairwiseBins: null,
  pairwiseSelectedBinIndex: null,
  pairwiseBinTokens: [],
  minDistanceRows: [],
  minDistanceHighlightedRows: [],
  recursiveGroup: null,
  recursiveGroupNetwork: null,
  recursiveGroupNetworkData: null,
  recursiveGroupHighlightMinEdges: false,
  linearTransformPairs: [],
  linearTransformRows: [],
  linearTransformSummary: null,
};

const pickerPrefixes = ['tokenA', 'tokenB', 'anchor', 'groupSeed', 'transformSource', 'transformTarget', 'transformInput'];

const cache = {
  tokenA: [],
  tokenB: [],
  anchor: [],
  groupSeed: [],
  transformSource: [],
  transformTarget: [],
  transformInput: [],
};

const searchRequestIds = {
  tokenA: 0,
  tokenB: 0,
  anchor: 0,
  groupSeed: 0,
  transformSource: 0,
  transformTarget: 0,
  transformInput: 0,
};

const lastSearchedQuery = {
  tokenA: null,
  tokenB: null,
  anchor: null,
  groupSeed: null,
  transformSource: null,
  transformTarget: null,
  transformInput: null,
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

function minDistanceRowsToCsv(rows) {
  const headers = [
    'Min-angle rank',
    'Token ID',
    'Raw token',
    'Display',
    'Vector length',
    'Min angle °',
    'Other token ID',
    'Other raw token',
    'Other display',
    'Other vector length',
  ];
  const body = rows.map((row) => [
    row.min_angle_rank,
    row.token_id,
    row.token_raw,
    row.token_display,
    row.magnitude,
    row.min_angle_deg,
    row.other_token_id,
    row.other_token_raw,
    row.other_token_display,
    row.other_magnitude,
  ]);
  return [headers, ...body].map((csvRow) => csvRow.map(csvEscape).join(',')).join('\n') + '\n';
}

function exportHighlightedMinDistanceRowsCsv() {
  const rows = highlightedMinDistanceRows();
  if (rows.length === 0) return;
  downloadBlob(minDistanceRowsToCsv(rows), 'highlighted_minimum_angular_distances.csv', 'text/csv;charset=utf-8');
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

function pickerEmptyMessage(prefix) {
  if (prefix === 'anchor') return 'No anchor selected.';
  if (prefix === 'groupSeed') return 'No seed selected.';
  if (prefix === 'transformSource') return 'No source selected.';
  if (prefix === 'transformTarget') return 'No target selected.';
  if (prefix === 'transformInput') return 'No input selected.';
  return 'No token selected.';
}

function parseNonNegativeInteger(rawValue) {
  const raw = String(rawValue ?? '').trim();
  if (raw === '') return null;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error('Enter a non-negative integer token ID.');
  }
  return value;
}

function markPickerDependentOutputsStale(key) {
  if (key === 'tokenA' || key === 'tokenB') {
    resetAngleOutput();
    resetCommonCloseOutput('Token selection changed. Click Find shared close tokens to refresh this list.');
  } else if (key === 'anchor') {
    resetNeighborhoodOutput();
  } else if (key === 'groupSeed') {
    resetRecursiveGroupOutput('Seed token changed. Click Build recursive group to refresh the graph.');
  } else if (key === 'transformSource' || key === 'transformTarget' || key === 'transformInput') {
    resetLinearTransformOutput('Transform token selection changed. Click Run transform neighbors to refresh this list.');
  }
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
    selected.textContent = message || pickerEmptyMessage(prefix);
  }

  markPickerDependentOutputsStale(key);
}

function setPickerSelectionFromKnownToken(prefix, key, token) {
  // Used after a compute endpoint has already validated the token ID and
  // returned token text.  This avoids an extra /api/tokens/id lookup and does
  // not clear the freshly rendered result panels.
  state[key] = token;
  byId(`${prefix}Id`).value = String(token.token_id);
  byId(`${prefix}Selected`).textContent = `Selected: ${tokenLabel(token)}`;
}

function markManualIdPending(prefix, key) {
  const rawId = byId(`${prefix}Id`).value.trim();
  const selected = byId(`${prefix}Selected`);
  const current = state[key];

  if (rawId === '') {
    state[key] = null;
    selected.textContent = pickerEmptyMessage(prefix);
  } else if (!current || String(current.token_id) !== rawId) {
    state[key] = null;
    selected.textContent = `Token ID ${rawId} entered. Click Use ID or run the action button.`;
  }

  markPickerDependentOutputsStale(key);
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
  byId(`${prefix}Selected`).textContent = pickerEmptyMessage(prefix);
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
  resetPicker('groupSeed', 'groupSeed');
  resetPicker('transformSource', 'transformSource');
  resetPicker('transformTarget', 'transformTarget');
  resetPicker('transformInput', 'transformInput');
  clearLinearTransformPairs(false);

  resetAngleOutput();
  resetCommonCloseOutput();
  resetNeighborhoodOutput();
  resetPairwiseBinsOutput();
  resetMinDistancesOutput();
  resetRecursiveGroupOutput();
  clearLinearTransformPairs(false);
  resetLinearTransformOutput();
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
  const selected = byId(`${prefix}Selected`);
  let tokenId;
  try {
    tokenId = parseNonNegativeInteger(byId(`${prefix}Id`).value);
  } catch (error) {
    state[key] = null;
    selected.textContent = error.message;
    markPickerDependentOutputsStale(key);
    return;
  }
  if (tokenId === null) {
    state[key] = null;
    selected.textContent = pickerEmptyMessage(prefix);
    markPickerDependentOutputsStale(key);
    return;
  }

  try {
    const token = await fetchJson(`/api/tokens/id/${tokenId}`);
    setPickerSelection(prefix, key, token);
  } catch (error) {
    state[key] = null;
    selected.textContent = `Token lookup failed: ${error.message}`;
    markPickerDependentOutputsStale(key);
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
  // "Use ID", resolve that ID first.  Manual ID input intentionally takes
  // precedence over stale state from a previous dropdown/search selection.
  const rawId = byId(`${prefix}Id`).value.trim();
  if (rawId !== '') {
    const tokenId = parseNonNegativeInteger(rawId);
    if (state[key] && state[key].token_id === tokenId) return state[key];
    const token = await fetchJson(`/api/tokens/id/${tokenId}`);
    setPickerSelection(prefix, key, token);
    return token;
  }

  if (state[key]) return state[key];

  const rawSelectedId = byId(`${prefix}Results`).value.trim();
  if (rawSelectedId !== '') {
    const selectedId = parseNonNegativeInteger(rawSelectedId);
    let token = cache[prefix].find((candidate) => candidate.token_id === selectedId);
    if (!token) token = await fetchJson(`/api/tokens/id/${selectedId}`);
    setPickerSelection(prefix, key, token);
    return token;
  }

  return null;
}

function getPickerTokenId(prefix, key) {
  // Lightweight path for compute endpoints that only need the token ID.  This
  // lets the recursive group feature work even when the user simply types an
  // ID and presses Build, without needing a separate Use ID click first.
  const rawId = byId(`${prefix}Id`).value.trim();
  if (rawId !== '') return parseNonNegativeInteger(rawId);

  if (state[key]) return state[key].token_id;

  const rawSelectedId = byId(`${prefix}Results`).value.trim();
  if (rawSelectedId !== '') return parseNonNegativeInteger(rawSelectedId);

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
  const idInput = byId(`${prefix}Id`);
  idInput.addEventListener('input', () => markManualIdPending(prefix, key));
  idInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      selectFromId(prefix, key);
    }
  });
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

function getMinDistanceHighlightSpec() {
  const startInput = byId('minDistanceHighlightStart');
  const endInput = byId('minDistanceHighlightEnd');
  const idsInput = byId('minDistanceHighlightIds');
  if (!startInput || !endInput || !idsInput) return null;
  const startText = startInput.value.trim();
  const endText = endInput.value.trim();
  const idsText = idsInput.value.trim();
  if (startText === '' && endText === '' && idsText === '') return null;

  let range = null;
  const startValue = startText === '' ? Number(endText) : Number(startText);
  const endValue = endText === '' ? Number(startText) : Number(endText);
  if (startText !== '' || endText !== '') {
    if (Number.isFinite(startValue) && Number.isFinite(endValue)) {
      range = {
        low: Math.max(0, Math.floor(Math.min(startValue, endValue))),
        high: Math.max(0, Math.floor(Math.max(startValue, endValue))),
      };
    }
  }

  const exactIds = new Set();
  for (const part of idsText.split(/[,\s]+/)) {
    if (!part) continue;
    const value = Number(part);
    if (Number.isFinite(value) && value >= 0) exactIds.add(Math.floor(value));
  }

  if (!range && exactIds.size === 0) return null;
  return { range, exactIds };
}

function minDistanceRowMatchesHighlight(row, highlightSpec) {
  if (!highlightSpec) return false;
  const tokenId = Number(row.token_id);
  if (!Number.isFinite(tokenId)) return false;
  if (highlightSpec.exactIds.has(tokenId)) return true;
  return Boolean(highlightSpec.range && tokenId >= highlightSpec.range.low && tokenId <= highlightSpec.range.high);
}

function minDistanceHighlightDescription(highlightSpec) {
  if (!highlightSpec) return '';
  const parts = [];
  if (highlightSpec.range) {
    parts.push(`IDs ${highlightSpec.range.low.toLocaleString()}–${highlightSpec.range.high.toLocaleString()}`);
  }
  if (highlightSpec.exactIds.size > 0) {
    const ids = Array.from(highlightSpec.exactIds).sort((a, b) => a - b);
    const preview = ids.slice(0, 12).map((id) => id.toLocaleString()).join(', ');
    const suffix = ids.length > 12 ? `, … +${(ids.length - 12).toLocaleString()} more` : '';
    parts.push(`exact IDs ${preview}${suffix}`);
  }
  return parts.join(' and ');
}

function highlightedMinDistanceRows() {
  const highlightSpec = getMinDistanceHighlightSpec();
  if (!highlightSpec) return [];
  return sortedMinDistanceRowsForTable().filter((row) => minDistanceRowMatchesHighlight(row, highlightSpec));
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
  const highlightColor = '#f97316';

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

  const highlightSpec = getMinDistanceHighlightSpec();
  state.minDistanceHighlightedRows = [];
  if (highlightSpec) {
    ctx.fillStyle = highlightColor;
    const highlightRadius = rowCount > 20000 ? 2.5 : 4.5;
    for (let index = 0; index < rowCount; index += 1) {
      const row = sortedRows[index];
      if (!minDistanceRowMatchesHighlight(row, highlightSpec)) continue;
      state.minDistanceHighlightedRows.push(row);
      const x = xForRank(index);
      const y = yForAngle(Number(row.min_angle_deg || 0));
      if (rowCount > 20000) {
        ctx.fillRect(x - highlightRadius / 2, y - highlightRadius / 2, highlightRadius, highlightRadius);
      } else {
        ctx.beginPath();
        ctx.arc(x, y, highlightRadius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  let caption = 'Y axis is the minimum non-self angle. X axis is rank after sorting tokens from lowest to highest minimum angle.';
  if (highlightSpec) {
    const highlightCount = state.minDistanceHighlightedRows.length;
    const description = minDistanceHighlightDescription(highlightSpec);
    caption += ` Orange dots highlight ${highlightCount.toLocaleString()} token${highlightCount === 1 ? '' : 's'} matching ${description}.`;
  } else {
    caption += ' Enter a token ID range or comma-separated token IDs above to overlay matching tokens as orange dots.';
  }
  byId('minDistancePlotCaption').textContent = caption;
  setExportVisible('exportMinDistanceHighlightCsv', state.minDistanceHighlightedRows.length > 0);
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
  state.minDistanceHighlightedRows = [];
  const output = byId('minDistancesOutput');
  if (!output) return;
  setOutput('minDistancesOutput', 'Load a model, then compute each token’s closest non-self angular neighbor. CUDA is used automatically when available.', true);
  byId('minDistancePlotCard').classList.add('hidden');
  const table = byId('minDistancesTable');
  table.classList.add('hidden');
  table.querySelector('tbody').innerHTML = '';
  setExportVisible('exportMinDistancesCsv', false);
  setExportVisible('exportMinDistanceHighlightCsv', false);
}

function refreshMinDistanceHighlight() {
  if (state.minDistanceRows.length > 0) {
    drawMinDistancePlot(state.minDistanceRows);
  }
}

async function computeMinAngularDistances() {
  const button = byId('minDistancesButton');
  const blockSize = Number(byId('minDistanceBlockSize').value || 2048);
  const computeDevice = byId('minDistanceComputeDevice').value.trim() || 'auto';

  state.minDistanceRows = [];
  state.minDistanceHighlightedRows = [];
  byId('minDistancePlotCard').classList.add('hidden');
  byId('minDistancesTable').classList.add('hidden');
  byId('minDistancesTable').querySelector('tbody').innerHTML = '';
  setExportVisible('exportMinDistancesCsv', false);
  setExportVisible('exportMinDistanceHighlightCsv', false);
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
    state.minDistanceHighlightedRows = [];
    setExportVisible('exportMinDistancesCsv', false);
    setExportVisible('exportMinDistanceHighlightCsv', false);
    setOutput('minDistancesOutput', `<strong>Minimum-distance computation failed:</strong> ${escapeHtml(error.message)}`);
  } finally {
    button.disabled = false;
    button.textContent = 'Compute minimum distances';
  }
}


function destroyRecursiveGroupNetwork() {
  if (state.recursiveGroupNetwork) {
    state.recursiveGroupNetwork.destroy();
    state.recursiveGroupNetwork = null;
  }
  state.recursiveGroupNetworkData = null;
}

function resetRecursiveGroupOutput(message = 'Choose a seed token, set the group controls, then build the recursive graph.') {
  state.recursiveGroup = null;
  destroyRecursiveGroupNetwork();
  const output = byId('recursiveGroupOutput');
  if (!output) return;
  setOutput('recursiveGroupOutput', message, true);
  byId('recursiveGroupGraphCard').classList.add('hidden');
  byId('recursiveGroupGraph').innerHTML = '';
  byId('recursiveGroupTable').classList.add('hidden');
  byId('recursiveGroupTable').querySelector('tbody').innerHTML = '';
  setExportVisible('exportRecursiveGroupGraphSvg', false);
  setExportVisible('exportRecursiveGroupGraphPng', false);
  setExportVisible('exportRecursiveGroupAdjacencyCsv', false);
  setExportVisible('exportRecursiveGroupDictionaryJson', false);
  setExportVisible('exportRecursiveGroupListCsv', false);
}

function createSvgElement(name) {
  return document.createElementNS('http://www.w3.org/2000/svg', name);
}

function truncateTokenForNode(value, maxLength = 18) {
  const text = String(value ?? '');
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(1, maxLength - 1))}…`;
}

function recursiveGraphNodeLabel(node) {
  const token = JSON.stringify(String(node.token_raw ?? ''));
  return `${node.token_id}\n${truncateTokenForNode(token, 24)}`;
}

function recursiveGraphNodeTitle(node) {
  return [
    `<strong>${escapeHtml(node.token_id)}</strong>`,
    `raw: <code>${escapeHtml(node.token_raw)}</code>`,
    `display: <code>${escapeHtml(node.token_display)}</code>`,
    `vector length: ${Number(node.magnitude).toFixed(6)}`,
    `connected nodes: ${formatCount(node.connected_count)}`,
  ].join('<br>');
}

function recursiveGraphEdgeTitle(edge) {
  return `${escapeHtml(edge.source_token_id)} ↔ ${escapeHtml(edge.target_token_id)}<br>${Number(edge.angle_deg).toFixed(6)}°`;
}

function getRecursiveMinEdgeKeys(data) {
  const edges = data?.edges || [];
  const minAngleByNode = new Map();

  for (const edge of edges) {
    const sourceId = Number(edge.source_token_id);
    const targetId = Number(edge.target_token_id);
    const angle = Number(edge.angle_deg);
    if (!Number.isFinite(sourceId) || !Number.isFinite(targetId) || !Number.isFinite(angle)) continue;

    for (const nodeId of [sourceId, targetId]) {
      const existing = minAngleByNode.get(nodeId);
      if (existing === undefined || angle < existing) minAngleByNode.set(nodeId, angle);
    }
  }

  const highlighted = new Set();
  const tolerance = 1e-9;
  for (const edge of edges) {
    const sourceId = Number(edge.source_token_id);
    const targetId = Number(edge.target_token_id);
    const angle = Number(edge.angle_deg);
    if (!Number.isFinite(sourceId) || !Number.isFinite(targetId) || !Number.isFinite(angle)) continue;

    const sourceMin = minAngleByNode.get(sourceId);
    const targetMin = minAngleByNode.get(targetId);
    if ((sourceMin !== undefined && angle <= sourceMin + tolerance) ||
        (targetMin !== undefined && angle <= targetMin + tolerance)) {
      highlighted.add(recursiveEdgeKey(sourceId, targetId));
    }
  }

  return highlighted;
}

function isRecursiveMinEdgeHighlightEnabled() {
  const input = byId('recursiveGroupHighlightMinEdges');
  return Boolean(input && input.checked);
}

function applyRecursiveGraphMinEdgeHighlight(svg, data, enabled = isRecursiveMinEdgeHighlightEnabled()) {
  if (!svg || !data) return new Set();
  const highlighted = enabled ? getRecursiveMinEdgeKeys(data) : new Set();
  svg.querySelectorAll('.graph-edge, .graph-edge-label').forEach((element) => {
    const sourceId = Number(element.dataset.sourceTokenId);
    const targetId = Number(element.dataset.targetTokenId);
    const key = element.dataset.edgeKey || recursiveEdgeKey(sourceId, targetId);
    const shouldHighlight = highlighted.has(key);
    element.classList.toggle('graph-edge-min', enabled && shouldHighlight && element.classList.contains('graph-edge'));
    element.classList.toggle('graph-edge-label-min', enabled && shouldHighlight && element.classList.contains('graph-edge-label'));
  });
  return highlighted;
}

function recursiveGroupCaptionText(data, interactive = true) {
  const nodeCount = (data?.nodes || []).length;
  const edgeCount = (data?.edges || []).length;
  const dragText = interactive
    ? 'Drag nodes to reposition them; drag empty space to pan; mouse-wheel zooms; double-click resets the view. SVG/PNG exports use the current node positions.'
    : 'SVG support was limited, so this fallback is static.';
  let text = `Showing ${formatCount(nodeCount)} nodes and ${formatCount(edgeCount)} labelled edges. ${dragText}`;
  if (isRecursiveMinEdgeHighlightEnabled()) {
    const highlightedCount = getRecursiveMinEdgeKeys(data).size;
    text += ` Highlighting ${formatCount(highlightedCount)} unique edge${highlightedCount === 1 ? '' : 's'} that are the lowest-angle incident edge for at least one node.`;
  }
  return text;
}

function refreshRecursiveGroupMinEdgeHighlight() {
  const input = byId('recursiveGroupHighlightMinEdges');
  state.recursiveGroupHighlightMinEdges = Boolean(input && input.checked);
  const svg = byId('recursiveGroupGraph')?.querySelector('svg');
  if (svg && state.recursiveGroup) {
    applyRecursiveGraphMinEdgeHighlight(svg, state.recursiveGroup, state.recursiveGroupHighlightMinEdges);
    byId('recursiveGroupGraphCaption').textContent = recursiveGroupCaptionText(state.recursiveGroup, !byId('recursiveGroupGraph')?.classList.contains('static-svg-fallback'));
  }
}

function renderRecursiveGroupTable(nodes) {
  const table = byId('recursiveGroupTable');
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';

  const rows = [...(nodes || [])].sort((a, b) => Number(a.token_id) - Number(b.token_id));
  const fragment = document.createDocumentFragment();
  for (const node of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${node.token_id}</td>
      <td><code>${escapeHtml(node.token_raw)}</code></td>
      <td><code>${escapeHtml(node.token_display)}</code></td>
      <td>${formatCount(node.connected_count)}</td>
      <td>${Number(node.magnitude).toFixed(6)}</td>
    `;
    fragment.appendChild(tr);
  }
  tbody.appendChild(fragment);
  table.classList.toggle('hidden', rows.length === 0);
  setExportVisible('exportRecursiveGroupListCsv', rows.length > 0);
}

const RECURSIVE_GRAPH_NODE_WIDTH = 150;
const RECURSIVE_GRAPH_NODE_HEIGHT = 54;

function circularRecursiveGraphPositions(nodes, width, height) {
  const nodeCount = nodes.length;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = nodeCount === 1 ? 0 : Math.max(180, Math.min(width, height) / 2 - 115);
  const positions = new Map();
  for (const [index, node] of nodes.entries()) {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(1, nodeCount);
    positions.set(Number(node.token_id), {
      x: nodeCount === 1 ? centerX : centerX + radius * Math.cos(angle),
      y: nodeCount === 1 ? centerY : centerY + radius * Math.sin(angle),
    });
  }
  return positions;
}

function forceRecursiveGraphPositions(nodes, edges, width, height) {
  const base = circularRecursiveGraphPositions(nodes, width, height);
  const simNodes = nodes.map((node) => {
    const id = Number(node.token_id);
    const point = base.get(id) || { x: width / 2, y: height / 2 };
    return { id, x: point.x, y: point.y, vx: 0, vy: 0 };
  });
  const byId = new Map(simNodes.map((node) => [node.id, node]));
  const simEdges = (edges || [])
    .map((edge) => ({
      source: byId.get(Number(edge.source_token_id)),
      target: byId.get(Number(edge.target_token_id)),
    }))
    .filter((edge) => edge.source && edge.target);

  const count = simNodes.length;
  if (count <= 1) return base;

  const centerX = width / 2;
  const centerY = height / 2;
  const idealLink = Math.max(120, Math.min(260, 110 + count * 2.0));
  const repelStrength = Math.max(2600, Math.min(11000, 5200 + count * 36));
  const iterations = Math.max(80, Math.min(240, 80 + count * 2));

  for (let step = 0; step < iterations; step += 1) {
    const alpha = 1 - step / iterations;

    for (let i = 0; i < count; i += 1) {
      const a = simNodes[i];
      for (let j = i + 1; j < count; j += 1) {
        const b = simNodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let distance2 = dx * dx + dy * dy;
        if (distance2 < 0.01) {
          dx = (i % 2 === 0 ? 1 : -1) * 0.1;
          dy = (j % 2 === 0 ? 1 : -1) * 0.1;
          distance2 = dx * dx + dy * dy;
        }
        const distance = Math.sqrt(distance2);
        const force = (repelStrength * alpha) / Math.max(80, distance2);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }

    for (const edge of simEdges) {
      const a = edge.source;
      const b = edge.target;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const stretch = distance - idealLink;
      const force = stretch * 0.012 * alpha;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }

    for (const node of simNodes) {
      node.vx += (centerX - node.x) * 0.0025 * alpha;
      node.vy += (centerY - node.y) * 0.0025 * alpha;
      node.vx *= 0.78;
      node.vy *= 0.78;
      node.x += node.vx;
      node.y += node.vy;
      node.x = Math.max(RECURSIVE_GRAPH_NODE_WIDTH, Math.min(width - RECURSIVE_GRAPH_NODE_WIDTH, node.x));
      node.y = Math.max(RECURSIVE_GRAPH_NODE_HEIGHT, Math.min(height - RECURSIVE_GRAPH_NODE_HEIGHT, node.y));
    }
  }

  const positions = new Map();
  for (const node of simNodes) {
    positions.set(node.id, { x: node.x, y: node.y });
  }
  return positions;
}

function recursiveGraphExportCss() {
  return `
    .recursive-graph-svg { background: #020617; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .graph-edge { stroke: rgba(148, 163, 184, 0.52); stroke-width: 1.25px; }
    .graph-edge-min { stroke: #fbbf24; stroke-width: 3.2px; opacity: 0.98; }
    .graph-edge-label { fill: #94a3b8; font-size: 10px; text-anchor: middle; dominant-baseline: central; paint-order: stroke; stroke: #020617; stroke-width: 3px; stroke-linejoin: round; }
    .graph-edge-label-min { fill: #fde68a; font-size: 11px; font-weight: 800; stroke-width: 3.5px; }
    .graph-node rect { fill: #111827; stroke: #38bdf8; stroke-width: 1.1px; }
    .graph-node text { text-anchor: middle; pointer-events: none; }
    .graph-node-id { fill: #f8fafc; font-weight: 700; font-size: 13px; }
    .graph-node-token { fill: #cbd5e1; font-size: 11px; }
  `;
}

function embedRecursiveGraphStyles(svg) {
  const existing = svg.querySelector('style[data-recursive-graph-export]');
  if (existing) existing.remove();
  const style = createSvgElement('style');
  style.setAttribute('data-recursive-graph-export', 'true');
  style.textContent = recursiveGraphExportCss();
  svg.insertBefore(style, svg.firstChild);
  return svg;
}

function createRecursiveGroupSvg(data, positions, options = {}) {
  const nodes = [...(data.nodes || [])].sort((a, b) => Number(a.token_id) - Number(b.token_id));
  const edges = data.edges || [];
  const nodeWidth = RECURSIVE_GRAPH_NODE_WIDTH;
  const nodeHeight = RECURSIVE_GRAPH_NODE_HEIGHT;
  const margin = 80;

  let minX = 0;
  let minY = 0;
  let maxX = 980;
  let maxY = 720;
  if (positions && positions.size > 0) {
    const values = [...positions.values()];
    minX = Math.min(...values.map((point) => point.x)) - nodeWidth / 2 - margin;
    maxX = Math.max(...values.map((point) => point.x)) + nodeWidth / 2 + margin;
    minY = Math.min(...values.map((point) => point.y)) - nodeHeight / 2 - margin;
    maxY = Math.max(...values.map((point) => point.y)) + nodeHeight / 2 + margin;
  }
  const width = Number(options.width || Math.max(980, Math.ceil(maxX - minX)));
  const height = Number(options.height || Math.max(720, Math.ceil(maxY - minY)));
  const viewBox = options.viewBox || `${minX} ${minY} ${width} ${height}`;

  const svg = createSvgElement('svg');
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  svg.setAttribute('viewBox', viewBox);
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Recursive token angle graph');
  svg.classList.add('recursive-graph-svg');
  if (options.interactive) {
    svg.classList.add('recursive-graph-svg-local');
    svg.setAttribute('tabindex', '0');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
  }
  if (options.embedStyles) embedRecursiveGraphStyles(svg);

  const edgeLayer = createSvgElement('g');
  edgeLayer.classList.add('graph-edge-layer');
  const labelLayer = createSvgElement('g');
  labelLayer.classList.add('graph-edge-label-layer');
  const nodeLayer = createSvgElement('g');
  nodeLayer.classList.add('graph-node-layer');
  const highlightedEdgeKeys = options.highlightMinEdges ? getRecursiveMinEdgeKeys(data) : new Set();

  for (const edge of edges) {
    const sourceId = Number(edge.source_token_id);
    const targetId = Number(edge.target_token_id);
    const source = positions.get(sourceId);
    const target = positions.get(targetId);
    if (!source || !target) continue;

    const edgeKey = recursiveEdgeKey(sourceId, targetId);
    const line = createSvgElement('line');
    line.classList.add('graph-edge');
    line.dataset.sourceTokenId = String(sourceId);
    line.dataset.targetTokenId = String(targetId);
    line.dataset.edgeKey = edgeKey;
    line.dataset.angleDeg = String(edge.angle_deg);
    line.setAttribute('x1', String(source.x));
    line.setAttribute('y1', String(source.y));
    line.setAttribute('x2', String(target.x));
    line.setAttribute('y2', String(target.y));
    if (highlightedEdgeKeys.has(edgeKey)) line.classList.add('graph-edge-min');
    edgeLayer.appendChild(line);

    const label = createSvgElement('text');
    label.classList.add('graph-edge-label');
    label.dataset.sourceTokenId = String(sourceId);
    label.dataset.targetTokenId = String(targetId);
    label.dataset.edgeKey = edgeKey;
    label.dataset.angleDeg = String(edge.angle_deg);
    label.setAttribute('x', String((source.x + target.x) / 2));
    label.setAttribute('y', String((source.y + target.y) / 2));
    label.textContent = `${Number(edge.angle_deg).toFixed(2)}°`;
    if (highlightedEdgeKeys.has(edgeKey)) label.classList.add('graph-edge-label-min');
    labelLayer.appendChild(label);
  }

  for (const node of nodes) {
    const id = Number(node.token_id);
    const position = positions.get(id);
    if (!position) continue;

    const group = createSvgElement('g');
    group.classList.add('graph-node');
    group.dataset.tokenId = String(id);
    group.setAttribute('transform', `translate(${position.x - nodeWidth / 2}, ${position.y - nodeHeight / 2})`);

    const title = createSvgElement('title');
    title.textContent = `${node.token_id} | ${node.token_raw}`;
    group.appendChild(title);

    const rect = createSvgElement('rect');
    rect.setAttribute('width', String(nodeWidth));
    rect.setAttribute('height', String(nodeHeight));
    rect.setAttribute('rx', '12');
    group.appendChild(rect);

    const idText = createSvgElement('text');
    idText.classList.add('graph-node-id');
    idText.setAttribute('x', String(nodeWidth / 2));
    idText.setAttribute('y', '20');
    idText.textContent = String(node.token_id);
    group.appendChild(idText);

    const tokenText = createSvgElement('text');
    tokenText.classList.add('graph-node-token');
    tokenText.setAttribute('x', String(nodeWidth / 2));
    tokenText.setAttribute('y', '38');
    tokenText.textContent = truncateTokenForNode(JSON.stringify(String(node.token_raw ?? '')), 22);
    group.appendChild(tokenText);

    nodeLayer.appendChild(group);
  }

  svg.appendChild(edgeLayer);
  svg.appendChild(labelLayer);
  svg.appendChild(nodeLayer);
  return svg;
}

function renderRecursiveGroupStaticSvgGraph(data) {
  const container = byId('recursiveGroupGraph');
  container.classList.add('static-svg-fallback');
  const nodes = [...(data.nodes || [])].sort((a, b) => Number(a.token_id) - Number(b.token_id));
  const nodeCount = nodes.length;
  const width = Math.max(980, Math.min(2600, 820 + nodeCount * 16));
  const height = Math.max(720, Math.min(2000, 680 + nodeCount * 8));
  const positions = circularRecursiveGraphPositions(nodes, width, height);
  const svg = createRecursiveGroupSvg(data, positions, {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    embedStyles: false,
    highlightMinEdges: isRecursiveMinEdgeHighlightEnabled(),
  });
  container.appendChild(svg);
  byId('recursiveGroupGraphCaption').textContent = recursiveGroupCaptionText(data, false);
}

function svgPointFromClient(svg, event) {
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const ctm = svg.getScreenCTM();
  if (!ctm) return { x: point.x, y: point.y };
  return point.matrixTransform(ctm.inverse());
}

function setRecursiveSvgViewBox(svg, viewBox) {
  svg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
}

function updateRecursiveGraphElements(svg, positions, tokenId = null) {
  const nodeWidth = RECURSIVE_GRAPH_NODE_WIDTH;
  const nodeHeight = RECURSIVE_GRAPH_NODE_HEIGHT;
  const updateNode = (id) => {
    const position = positions.get(Number(id));
    const group = svg.querySelector(`.graph-node[data-token-id="${String(id)}"]`);
    if (position && group) {
      group.setAttribute('transform', `translate(${position.x - nodeWidth / 2}, ${position.y - nodeHeight / 2})`);
    }
  };
  const updateEdge = (element) => {
    const sourceId = Number(element.dataset.sourceTokenId);
    const targetId = Number(element.dataset.targetTokenId);
    if (tokenId !== null && sourceId !== tokenId && targetId !== tokenId) return;
    const source = positions.get(sourceId);
    const target = positions.get(targetId);
    if (!source || !target) return;
    if (element.tagName.toLowerCase() === 'line') {
      element.setAttribute('x1', String(source.x));
      element.setAttribute('y1', String(source.y));
      element.setAttribute('x2', String(target.x));
      element.setAttribute('y2', String(target.y));
    } else {
      element.setAttribute('x', String((source.x + target.x) / 2));
      element.setAttribute('y', String((source.y + target.y) / 2));
    }
  };

  if (tokenId === null) {
    for (const id of positions.keys()) updateNode(id);
  } else {
    updateNode(tokenId);
  }
  svg.querySelectorAll('.graph-edge, .graph-edge-label').forEach(updateEdge);
}

function attachRecursiveGraphInteractions(svg, positions, width, height) {
  let activeNodeId = null;
  let activeNodeElement = null;
  let nodeOffset = { x: 0, y: 0 };
  let panStart = null;
  const viewBox = { x: 0, y: 0, width, height };
  const eventOptions = { passive: false };

  const handlePointerDown = (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const nodeElement = target ? target.closest('.graph-node') : null;
    if (nodeElement && svg.contains(nodeElement)) {
      event.preventDefault();
      activeNodeId = Number(nodeElement.dataset.tokenId);
      activeNodeElement = nodeElement;
      activeNodeElement.classList.add('dragging');
      const current = positions.get(activeNodeId) || { x: 0, y: 0 };
      const point = svgPointFromClient(svg, event);
      nodeOffset = { x: current.x - point.x, y: current.y - point.y };
      svg.setPointerCapture?.(event.pointerId);
      return;
    }

    event.preventDefault();
    panStart = {
      clientX: event.clientX,
      clientY: event.clientY,
      viewBox: { ...viewBox },
    };
    svg.classList.add('panning');
    svg.setPointerCapture?.(event.pointerId);
  };

  const handlePointerMove = (event) => {
    if (activeNodeId !== null) {
      event.preventDefault();
      const point = svgPointFromClient(svg, event);
      const next = { x: point.x + nodeOffset.x, y: point.y + nodeOffset.y };
      positions.set(activeNodeId, next);
      updateRecursiveGraphElements(svg, positions, activeNodeId);
      return;
    }
    if (panStart) {
      event.preventDefault();
      const clientWidth = Math.max(1, svg.clientWidth || width);
      const clientHeight = Math.max(1, svg.clientHeight || height);
      const dx = ((event.clientX - panStart.clientX) / clientWidth) * panStart.viewBox.width;
      const dy = ((event.clientY - panStart.clientY) / clientHeight) * panStart.viewBox.height;
      viewBox.x = panStart.viewBox.x - dx;
      viewBox.y = panStart.viewBox.y - dy;
      setRecursiveSvgViewBox(svg, viewBox);
    }
  };

  const endPointerInteraction = (event) => {
    if (activeNodeElement) activeNodeElement.classList.remove('dragging');
    activeNodeId = null;
    activeNodeElement = null;
    panStart = null;
    svg.classList.remove('panning');
    if (event && event.pointerId !== undefined) svg.releasePointerCapture?.(event.pointerId);
  };

  const handleWheel = (event) => {
    event.preventDefault();
    const point = svgPointFromClient(svg, event);
    const zoomFactor = event.deltaY > 0 ? 1.14 : 0.88;
    const nextWidth = Math.max(180, Math.min(12000, viewBox.width * zoomFactor));
    const nextHeight = Math.max(140, Math.min(9000, viewBox.height * zoomFactor));
    const widthRatio = nextWidth / viewBox.width;
    const heightRatio = nextHeight / viewBox.height;
    viewBox.x = point.x - (point.x - viewBox.x) * widthRatio;
    viewBox.y = point.y - (point.y - viewBox.y) * heightRatio;
    viewBox.width = nextWidth;
    viewBox.height = nextHeight;
    setRecursiveSvgViewBox(svg, viewBox);
  };

  const handleDblClick = (event) => {
    event.preventDefault();
    viewBox.x = 0;
    viewBox.y = 0;
    viewBox.width = width;
    viewBox.height = height;
    setRecursiveSvgViewBox(svg, viewBox);
  };

  svg.addEventListener('pointerdown', handlePointerDown);
  svg.addEventListener('pointermove', handlePointerMove);
  svg.addEventListener('pointerup', endPointerInteraction);
  svg.addEventListener('pointercancel', endPointerInteraction);
  svg.addEventListener('lostpointercapture', endPointerInteraction);
  svg.addEventListener('wheel', handleWheel, eventOptions);
  svg.addEventListener('dblclick', handleDblClick);

  return {
    destroy() {
      svg.removeEventListener('pointerdown', handlePointerDown);
      svg.removeEventListener('pointermove', handlePointerMove);
      svg.removeEventListener('pointerup', endPointerInteraction);
      svg.removeEventListener('pointercancel', endPointerInteraction);
      svg.removeEventListener('lostpointercapture', endPointerInteraction);
      svg.removeEventListener('wheel', handleWheel, eventOptions);
      svg.removeEventListener('dblclick', handleDblClick);
    },
    getPositions(ids) {
      const result = {};
      for (const id of ids) {
        const point = positions.get(Number(id));
        if (point) result[id] = { x: point.x, y: point.y };
      }
      return result;
    },
    svg,
  };
}

function hasLocalSvgGraphSupport() {
  return typeof document !== 'undefined' && typeof document.createElementNS === 'function';
}

function renderRecursiveGroupInteractiveGraph(data) {
  const container = byId('recursiveGroupGraph');
  container.classList.remove('static-svg-fallback');
  container.classList.add('local-draggable-graph');
  const nodes = [...(data.nodes || [])].sort((a, b) => Number(a.token_id) - Number(b.token_id));
  const edges = data.edges || [];
  const width = Math.max(1040, Math.min(3200, 900 + nodes.length * 18));
  const height = Math.max(720, Math.min(2400, 680 + nodes.length * 9));
  const positions = forceRecursiveGraphPositions(nodes, edges, width, height);
  const svg = createRecursiveGroupSvg(data, positions, {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    interactive: true,
    embedStyles: false,
    highlightMinEdges: isRecursiveMinEdgeHighlightEnabled(),
  });
  container.appendChild(svg);
  state.recursiveGroupNetwork = attachRecursiveGraphInteractions(svg, positions, width, height);
  state.recursiveGroupNetworkData = { renderer: 'local-svg-drag' };

  byId('recursiveGroupGraphCaption').textContent = recursiveGroupCaptionText(data, true);
}

function renderRecursiveGroupGraph(data) {
  const container = byId('recursiveGroupGraph');
  destroyRecursiveGroupNetwork();
  container.innerHTML = '';
  container.classList.remove('local-draggable-graph', 'static-svg-fallback');
  const nodes = [...(data.nodes || [])].sort((a, b) => Number(a.token_id) - Number(b.token_id));
  if (nodes.length === 0) {
    byId('recursiveGroupGraphCard').classList.add('hidden');
    return;
  }

  if (hasLocalSvgGraphSupport()) {
    renderRecursiveGroupInteractiveGraph(data);
  } else {
    renderRecursiveGroupStaticSvgGraph(data);
  }

  byId('recursiveGroupGraphCard').classList.remove('hidden');
  setExportVisible('exportRecursiveGroupGraphSvg', true);
  setExportVisible('exportRecursiveGroupGraphPng', true);
}

async function computeRecursiveAngleGroup() {
  let seedId;
  try {
    seedId = getPickerTokenId('groupSeed', 'groupSeed');
  } catch (error) {
    setOutput('recursiveGroupOutput', `<strong>Seed token ID failed:</strong> ${escapeHtml(error.message)}`);
    return;
  }

  if (seedId === null) {
    setOutput('recursiveGroupOutput', 'Enter a seed token ID, or search/select a seed token first.', true);
    return;
  }

  const groupSizeLimit = Number(byId('recursiveGroupLimit').value || 100);
  const maxAngle = Number(byId('recursiveGroupMaxAngle').value || 35);
  const blockSize = Number(byId('recursiveGroupBlockSize').value || 2048);
  const computeDevice = byId('recursiveGroupComputeDevice').value.trim() || 'auto';

  if (!Number.isInteger(groupSizeLimit) || groupSizeLimit < 1 || groupSizeLimit > 1000) {
    setOutput('recursiveGroupOutput', 'Enter a group size limit from 1 to 1000.', true);
    return;
  }
  if (!Number.isFinite(maxAngle) || maxAngle < 0 || maxAngle > 180) {
    setOutput('recursiveGroupOutput', 'Enter a maximum angle from 0 to 180 degrees.', true);
    return;
  }
  if (!Number.isInteger(blockSize) || blockSize < 1 || blockSize > 16384) {
    setOutput('recursiveGroupOutput', 'Enter a block size from 1 to 16384.', true);
    return;
  }

  const button = byId('recursiveGroupButton');
  state.recursiveGroup = null;
  destroyRecursiveGroupNetwork();
  byId('recursiveGroupGraphCard').classList.add('hidden');
  byId('recursiveGroupGraph').innerHTML = '';
  byId('recursiveGroupTable').classList.add('hidden');
  byId('recursiveGroupTable').querySelector('tbody').innerHTML = '';
  setExportVisible('exportRecursiveGroupGraphSvg', false);
  setExportVisible('exportRecursiveGroupGraphPng', false);
  setExportVisible('exportRecursiveGroupAdjacencyCsv', false);
  setExportVisible('exportRecursiveGroupDictionaryJson', false);
  setExportVisible('exportRecursiveGroupListCsv', false);
  setOutput('recursiveGroupOutput', `Building recursive group from seed ${seedId} with max angle ${maxAngle}°…`, true);
  button.disabled = true;
  button.textContent = 'Building…';

  try {
    const params = new URLSearchParams({
      seed_id: String(seedId),
      max_angle_deg: String(maxAngle),
      group_size_limit: String(groupSizeLimit),
      block_size: String(blockSize),
      compute_device: computeDevice,
    });
    const data = await fetchJson(`/api/recursive-angle-group?${params.toString()}`);
    state.recursiveGroup = data;
    setPickerSelectionFromKnownToken('groupSeed', 'groupSeed', {
      token_id: data.seed_token_id,
      raw: data.seed_token_raw,
      display: data.seed_token_display,
    });
    setOutput('recursiveGroupOutput', `
      <div class="metric"><span>Collected tokens</span><strong>${formatCount(data.node_count)}</strong></div>
      <div class="metric"><span>Edges</span><strong>${formatCount(data.edge_count)}</strong></div>
      <table class="mini-table">
        <tbody>
          <tr><th>Seed token</th><td>${escapeHtml(tokenLabel({ token_id: data.seed_token_id, raw: data.seed_token_raw, display: data.seed_token_display }))}</td></tr>
          <tr><th>Max angle</th><td>${Number(data.max_angle_deg).toFixed(3)}°</td></tr>
          <tr><th>Size limit</th><td>${formatCount(data.group_size_limit)}</td></tr>
          <tr><th>Expansion status</th><td>${data.truncated ? 'Stopped at size limit' : 'Frontier exhausted'}</td></tr>
          <tr><th>Tokens scanned</th><td>${formatCount(data.scanned_count)}</td></tr>
          <tr><th>Compute device</th><td>${escapeHtml(data.compute_device)}</td></tr>
          <tr><th>Block size</th><td>${formatCount(data.block_size)}</td></tr>
          <tr><th>Elapsed</th><td>${Number(data.elapsed_seconds).toFixed(3)} seconds</td></tr>
        </tbody>
      </table>
    `);
    renderRecursiveGroupGraph(data);
    renderRecursiveGroupTable(data.nodes || []);
    setExportVisible('exportRecursiveGroupAdjacencyCsv', data.node_count > 0);
    setExportVisible('exportRecursiveGroupDictionaryJson', data.node_count > 0);
  } catch (error) {
    state.recursiveGroup = null;
    setOutput('recursiveGroupOutput', `<strong>Recursive group failed:</strong> ${escapeHtml(error.message)}`);
  } finally {
    button.disabled = false;
    button.textContent = 'Build recursive group';
  }
}


function resetLinearTransformOutput(message = 'Choose source, target, and input tokens, then run transform neighbors.') {
  state.linearTransformRows = [];
  state.linearTransformSummary = null;
  const output = byId('linearTransformOutput');
  if (!output) return;
  setOutput('linearTransformOutput', message, true);
  const table = byId('linearTransformTable');
  table.classList.add('hidden');
  table.querySelector('tbody').innerHTML = '';
  setExportVisible('exportLinearTransformCsv', false);
}

function updateLinearTransformPairControls() {
  const count = state.linearTransformPairs.length;
  const countElement = byId('linearTransformExampleCount');
  if (countElement) {
    countElement.textContent = count === 0
      ? 'No saved examples; current source/target picker will be used.'
      : `${count} saved source→target example${count === 1 ? '' : 's'}.`;
  }

  const select = byId('linearTransformType');
  if (!select) return;
  const effectiveExampleCount = count > 0 ? count : 1;
  for (const option of Array.from(select.options)) {
    const minExamples = Number(option.dataset.minExamples || 1);
    option.disabled = effectiveExampleCount < minExamples;
  }
  if (select.selectedOptions[0]?.disabled) {
    select.value = 'closest_identity';
  }
}

function renderLinearTransformPairsTable() {
  const table = byId('linearTransformExamplesTable');
  if (!table) return;
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';

  const fragment = document.createDocumentFragment();
  state.linearTransformPairs.forEach((pair, index) => {
    const tr = document.createElement('tr');
    const angleText = Number.isFinite(Number(pair.angle_deg)) ? Number(pair.angle_deg).toFixed(6) : '—';
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${pair.source.token_id}</td>
      <td><code>${escapeHtml(tokenLabel(pair.source))}</code></td>
      <td>${pair.target.token_id}</td>
      <td><code>${escapeHtml(tokenLabel(pair.target))}</code></td>
      <td>${angleText}</td>
      <td><button type="button" class="remove-transform-pair remove-example-button" data-index="${index}">Remove</button></td>
    `;
    fragment.appendChild(tr);
  });
  tbody.appendChild(fragment);

  for (const button of Array.from(tbody.querySelectorAll('.remove-transform-pair'))) {
    button.addEventListener('click', () => removeLinearTransformPair(Number(button.dataset.index)));
  }

  table.classList.toggle('hidden', state.linearTransformPairs.length === 0);
  updateLinearTransformPairControls();
}

async function addLinearTransformPair() {
  let source;
  let target;
  try {
    source = await ensurePickerSelection('transformSource', 'transformSource');
    target = await ensurePickerSelection('transformTarget', 'transformTarget');
  } catch (error) {
    setOutput('linearTransformOutput', `<strong>Example pair selection failed:</strong> ${escapeHtml(error.message)}`);
    return;
  }
  if (!source || !target) {
    setOutput('linearTransformOutput', 'Select an example source and target first. Search/select tokens or type token IDs.', true);
    return;
  }

  const duplicate = state.linearTransformPairs.some((pair) => (
    pair.source.token_id === source.token_id && pair.target.token_id === target.token_id
  ));
  if (duplicate) {
    setOutput('linearTransformOutput', 'That exact source→target example pair is already saved.', true);
    return;
  }

  let angleDeg = null;
  try {
    const angleData = await fetchJson(`/api/angle?token_a=${encodeURIComponent(source.token_id)}&token_b=${encodeURIComponent(target.token_id)}`);
    angleDeg = Number(angleData.angle_deg);
  } catch (_) {
    angleDeg = null;
  }

  state.linearTransformPairs.push({ source: { ...source }, target: { ...target }, angle_deg: angleDeg });
  renderLinearTransformPairsTable();
  resetLinearTransformOutput('Example pair added. Click Run transform neighbors to use the updated example set.');
}

function removeLinearTransformPair(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.linearTransformPairs.length) return;
  state.linearTransformPairs.splice(index, 1);
  renderLinearTransformPairsTable();
  resetLinearTransformOutput('Example pair removed. Click Run transform neighbors to use the updated example set.');
}

function clearLinearTransformPairs(resetOutput = true) {
  state.linearTransformPairs = [];
  const table = byId('linearTransformExamplesTable');
  if (table) {
    table.classList.add('hidden');
    table.querySelector('tbody').innerHTML = '';
  }
  updateLinearTransformPairControls();
  if (resetOutput) {
    resetLinearTransformOutput('Saved example pairs cleared. The current source/target picker will be used as one implicit example.');
  }
}

function linearTransformExamplePayload(source, target) {
  if (state.linearTransformPairs.length > 0) {
    return state.linearTransformPairs.map((pair) => ({
      source_token_id: Number(pair.source.token_id),
      target_token_id: Number(pair.target.token_id),
    }));
  }
  if (!source || !target) return [];
  return [{
    source_token_id: Number(source.token_id),
    target_token_id: Number(target.token_id),
  }];
}

function renderLinearTransformTable(rows) {
  const table = byId('linearTransformTable');
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';

  const fragment = document.createDocumentFragment();
  for (const row of rows || []) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.token_id}</td>
      <td><code>${escapeHtml(row.token_raw)}</code></td>
      <td><code>${escapeHtml(row.token_display)}</code></td>
      <td>${Number(row.angle_deg).toFixed(6)}</td>
      <td>${Number(row.angle_to_original_input_deg).toFixed(6)}</td>
      <td>${Number(row.cosine_similarity).toFixed(8)}</td>
      <td>${Number(row.magnitude).toFixed(6)}</td>
    `;
    fragment.appendChild(tr);
  }

  tbody.appendChild(fragment);
  table.classList.toggle('hidden', !rows || rows.length === 0);
  setExportVisible('exportLinearTransformCsv', Boolean(rows && rows.length > 0));
}

function renderLinearTransformExamplesSummary(examples) {
  if (!examples || examples.length === 0) return '<p>No examples returned.</p>';
  const rows = examples.map((example) => `
    <tr>
      <td>${example.example_index}</td>
      <td>${escapeHtml(tokenLabel({ token_id: example.source_token_id, raw: example.source_token_raw, display: example.source_token_display }))}</td>
      <td>${escapeHtml(tokenLabel({ token_id: example.target_token_id, raw: example.target_token_raw, display: example.target_token_display }))}</td>
      <td>${Number(example.source_to_target_angle_deg).toFixed(6)}°</td>
    </tr>
  `).join('');
  return `
    <details class="transform-example-summary" ${examples.length <= 4 ? 'open' : ''}>
      <summary>${examples.length} source→target example${examples.length === 1 ? '' : 's'}</summary>
      <div class="table-wrap mini-table-wrap">
        <table class="mini-table transform-examples-mini">
          <thead><tr><th>#</th><th>Source</th><th>Target</th><th>Pair angle</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </details>
  `;
}

// Backward-compatible single-pair GET route remains available for scripts/tests: fetchJson(`/api/linear-transform-neighbors?source_token_id=0&target_token_id=1&input_token_id=2`);
// Backward-compatible GET clients may still use fetchJson(`/api/linear-transform-neighbors?${params.toString()}`); the UI uses POST for multiple examples.
async function computeLinearTransformNeighbors() {
  let source = null;
  let target = null;
  let input;
  try {
    if (state.linearTransformPairs.length === 0) {
      source = await ensurePickerSelection('transformSource', 'transformSource');
      target = await ensurePickerSelection('transformTarget', 'transformTarget');
    }
    input = await ensurePickerSelection('transformInput', 'transformInput');
  } catch (error) {
    setOutput('linearTransformOutput', `<strong>Token selection failed:</strong> ${escapeHtml(error.message)}`);
    return;
  }

  const examples = linearTransformExamplePayload(source, target);
  if (examples.length === 0 || !input) {
    setOutput('linearTransformOutput', 'Select at least one source→target example and an input token. You can save examples with Add source→target example pair or use the current source/target picker as one implicit example.', true);
    return;
  }

  const limit = Number(byId('linearTransformLimit').value || 200);
  if (!Number.isInteger(limit) || limit < 1 || limit > 5000) {
    setOutput('linearTransformOutput', 'Enter a nearest-neighbor row limit from 1 to 5000.', true);
    return;
  }
  const ridgeLambda = Number(byId('linearTransformRidgeLambda').value || 0);
  if (!Number.isFinite(ridgeLambda) || ridgeLambda <= 0) {
    setOutput('linearTransformOutput', 'Enter a positive ridge λ value.', true);
    return;
  }
  const rawTransformScale = byId('linearTransformScale').value.trim();
  const transformScale = rawTransformScale === '' ? 1 : Number(rawTransformScale);
  if (!Number.isFinite(transformScale) || Math.abs(transformScale) > 1000) {
    setOutput('linearTransformOutput', 'Enter a finite transform scale between -1000 and 1000.', true);
    return;
  }
  const transformType = byId('linearTransformType').value || 'closest_identity';

  const button = byId('linearTransformButton');
  state.linearTransformRows = [];
  state.linearTransformSummary = null;
  byId('linearTransformTable').classList.add('hidden');
  byId('linearTransformTable').querySelector('tbody').innerHTML = '';
  setExportVisible('exportLinearTransformCsv', false);
  setOutput('linearTransformOutput', 'Applying transform and ranking nearest token vectors…', true);
  button.disabled = true;
  button.textContent = 'Running…';

  try {
    const data = await fetchJson('/api/linear-transform-neighbors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        examples,
        input_token_id: Number(input.token_id),
        limit,
        transform_type: transformType,
        ridge_lambda: ridgeLambda,
        transform_scale: transformScale,
      }),
    });
    state.linearTransformSummary = data;
    state.linearTransformRows = data.rows || [];
    if (data.examples && data.examples.length > 0) {
      const first = data.examples[0];
      setPickerSelectionFromKnownToken('transformSource', 'transformSource', {
        token_id: first.source_token_id,
        raw: first.source_token_raw,
        display: first.source_token_display,
      });
      setPickerSelectionFromKnownToken('transformTarget', 'transformTarget', {
        token_id: first.target_token_id,
        raw: first.target_token_raw,
        display: first.target_token_display,
      });
    }
    setPickerSelectionFromKnownToken('transformInput', 'transformInput', {
      token_id: data.input_token_id,
      raw: data.input_token_raw,
      display: data.input_token_display,
    });
    setOutput('linearTransformOutput', `
      <div class="metric"><span>Nearest rows</span><strong>${formatCount(state.linearTransformRows.length)}</strong></div>
      <table class="mini-table">
        <tbody>
          <tr><th>Example count</th><td>${formatCount(data.pair_count)}</td></tr>
          <tr><th>Effective source rank</th><td>${formatCount(data.effective_source_rank)}</td></tr>
          <tr><th>Input</th><td>${escapeHtml(tokenLabel({ token_id: data.input_token_id, raw: data.input_token_raw, display: data.input_token_display }))}</td></tr>
          <tr><th>Input vector length</th><td>${Number(data.input_token_magnitude).toFixed(6)}</td></tr>
          <tr><th>Transform</th><td>${escapeHtml(data.transform_description)}</td></tr>
          <tr><th>Transform scale</th><td>${Number(data.transform_scale ?? transformScale).toFixed(6)}×</td></tr>
          <tr><th>First source→target angle</th><td>${Number(data.source_to_target_angle_deg).toFixed(6)}°</td></tr>
          <tr><th>Example fit RMS angle</th><td>${Number(data.example_fit_rmse ?? 0).toFixed(6)}°</td></tr>
          <tr><th>Example fit max angle</th><td>${Number(data.example_fit_max_angle_deg ?? 0).toFixed(6)}°</td></tr>
          <tr><th>Input→transformed angle</th><td>${Number(data.input_to_transformed_angle_deg).toFixed(6)}°</td></tr>
          <tr><th>${escapeHtml(data.transform_parameter_label || 'Transform parameter')}</th><td>${Number(data.coefficient).toFixed(8)}</td></tr>
          <tr><th>Ridge λ</th><td>${Number(data.ridge_lambda).toPrecision(6)}</td></tr>
          <tr><th>Transformed vector length</th><td>${Number(data.transformed_vector_magnitude).toFixed(6)}</td></tr>
        </tbody>
      </table>
      ${renderLinearTransformExamplesSummary(data.examples || [])}
    `);
    renderLinearTransformTable(state.linearTransformRows);
  } catch (error) {
    state.linearTransformRows = [];
    state.linearTransformSummary = null;
    setExportVisible('exportLinearTransformCsv', false);
    setOutput('linearTransformOutput', `<strong>Transform-neighbor search failed:</strong> ${escapeHtml(error.message)}`);
  } finally {
    button.disabled = false;
    button.textContent = 'Run transform neighbors';
  }
}

function recursiveEdgeKey(a, b) {
  const source = Math.min(Number(a), Number(b));
  const target = Math.max(Number(a), Number(b));
  return `${source}:${target}`;
}

function exportRecursiveGroupAdjacencyCsv() {
  const data = state.recursiveGroup;
  if (!data || !data.nodes) return;
  const nodes = [...data.nodes].sort((a, b) => Number(a.token_id) - Number(b.token_id));
  const edgeAngles = new Map();
  for (const edge of data.edges || []) {
    edgeAngles.set(recursiveEdgeKey(edge.source_token_id, edge.target_token_id), Number(edge.angle_deg));
  }

  const header = ['token_id', ...nodes.map((node) => node.token_id)];
  const rows = [header];
  for (const rowNode of nodes) {
    const row = [rowNode.token_id];
    for (const colNode of nodes) {
      if (Number(rowNode.token_id) === Number(colNode.token_id)) {
        row.push('0');
      } else {
        const angle = edgeAngles.get(recursiveEdgeKey(rowNode.token_id, colNode.token_id));
        row.push(angle === undefined ? '' : angle.toFixed(6));
      }
    }
    rows.push(row);
  }
  const csv = rows.map((row) => row.map(csvEscape).join(',')).join('\n') + '\n';
  downloadBlob(csv, `recursive_group_${filenamePart(data.seed_token_id, 'seed')}_adjacency.csv`, 'text/csv;charset=utf-8');
}

function exportRecursiveGroupDictionaryJson() {
  const data = state.recursiveGroup;
  if (!data) return;
  const dictionary = data.dictionary || Object.fromEntries((data.nodes || []).map((node) => [String(node.token_id), node]));
  downloadBlob(
    JSON.stringify(dictionary, null, 2),
    `recursive_group_${filenamePart(data.seed_token_id, 'seed')}_dictionary.json`,
    'application/json;charset=utf-8'
  );
}

function getRecursiveGroupCurrentPositions(data) {
  const nodes = [...(data.nodes || [])].sort((a, b) => Number(a.token_id) - Number(b.token_id));
  if (state.recursiveGroupNetwork) {
    const ids = nodes.map((node) => Number(node.token_id));
    const rawPositions = state.recursiveGroupNetwork.getPositions(ids);
    const positions = new Map();
    for (const node of nodes) {
      const key = Number(node.token_id);
      const position = rawPositions[key] || rawPositions[String(key)];
      if (position && Number.isFinite(position.x) && Number.isFinite(position.y)) {
        positions.set(key, { x: Number(position.x), y: Number(position.y) });
      }
    }
    if (positions.size === nodes.length) return positions;
  }

  const width = Math.max(980, Math.min(2400, 820 + nodes.length * 14));
  const height = Math.max(720, Math.min(1800, 680 + nodes.length * 6));
  return circularRecursiveGraphPositions(nodes, width, height);
}

function exportRecursiveGroupGraphSvg() {
  const data = state.recursiveGroup;
  if (!data || !(data.nodes || []).length) return;
  const positions = getRecursiveGroupCurrentPositions(data);
  const svg = createRecursiveGroupSvg(data, positions, {
    embedStyles: true,
    highlightMinEdges: isRecursiveMinEdgeHighlightEnabled(),
  });
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const text = new XMLSerializer().serializeToString(svg);
  downloadBlob(text, `recursive_group_${filenamePart(data.seed_token_id, 'seed')}_graph.svg`, 'image/svg+xml;charset=utf-8');
}

function parseSvgViewBox(svg) {
  const raw = svg.getAttribute('viewBox') || '';
  const parts = raw.trim().split(/\s+/).map(Number);
  if (parts.length === 4 && parts.every(Number.isFinite)) {
    return { x: parts[0], y: parts[1], width: parts[2], height: parts[3] };
  }
  return {
    x: 0,
    y: 0,
    width: Number(svg.getAttribute('width')) || 1200,
    height: Number(svg.getAttribute('height')) || 800,
  };
}

function exportRecursiveGroupGraphPng() {
  const data = state.recursiveGroup;
  const currentSvg = byId('recursiveGroupGraph').querySelector('svg');
  if (!data || !currentSvg) return;

  const svg = currentSvg.cloneNode(true);
  embedRecursiveGraphStyles(svg);
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const viewBox = parseSvgViewBox(svg);
  const width = Math.max(1, Math.ceil(viewBox.width));
  const height = Math.max(1, Math.ceil(viewBox.height));
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));

  const text = new XMLSerializer().serializeToString(svg);
  const blob = new Blob([text], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      URL.revokeObjectURL(url);
      return;
    }
    ctx.drawImage(image, 0, 0, width, height);
    URL.revokeObjectURL(url);
    canvas.toBlob((pngBlob) => {
      if (!pngBlob) return;
      downloadBlob(pngBlob, `recursive_group_${filenamePart(data.seed_token_id, 'seed')}_graph.png`, 'image/png');
    }, 'image/png');
  };
  image.onerror = () => URL.revokeObjectURL(url);
  image.src = url;
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
  byId('exportMinDistanceHighlightCsv').addEventListener('click', exportHighlightedMinDistanceRowsCsv);
  byId('exportRecursiveGroupGraphSvg').addEventListener('click', exportRecursiveGroupGraphSvg);
  byId('exportRecursiveGroupGraphPng').addEventListener('click', exportRecursiveGroupGraphPng);
  byId('exportRecursiveGroupAdjacencyCsv').addEventListener('click', exportRecursiveGroupAdjacencyCsv);
  byId('exportRecursiveGroupDictionaryJson').addEventListener('click', exportRecursiveGroupDictionaryJson);
  byId('exportRecursiveGroupListCsv').addEventListener('click', () => exportHtmlTableAsCsv('recursiveGroupTable', 'recursive_group_token_list.csv'));
  byId('exportLinearTransformCsv').addEventListener('click', () => exportHtmlTableAsCsv('linearTransformTable', 'linear_transform_neighbors.csv'));
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
  setupPicker('groupSeed', 'groupSeed');
  setupPicker('transformSource', 'transformSource');
  setupPicker('transformTarget', 'transformTarget');
  setupPicker('transformInput', 'transformInput');
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
  byId('recursiveGroupButton').addEventListener('click', computeRecursiveAngleGroup);
  byId('linearTransformButton').addEventListener('click', computeLinearTransformNeighbors);
  byId('addLinearTransformExample').addEventListener('click', addLinearTransformPair);
  byId('clearLinearTransformExamples').addEventListener('click', () => clearLinearTransformPairs(true));
  updateLinearTransformPairControls();
  byId('recursiveGroupHighlightMinEdges').addEventListener('change', refreshRecursiveGroupMinEdgeHighlight);
  byId('minDistanceSort').addEventListener('change', () => {
    if (state.minDistanceRows.length > 0) renderMinDistanceTable();
  });
  for (const controlId of ['minDistanceHighlightStart', 'minDistanceHighlightEnd', 'minDistanceHighlightIds']) {
    byId(controlId).addEventListener('input', refreshMinDistanceHighlight);
    byId(controlId).addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        refreshMinDistanceHighlight();
      }
    });
  }
  byId('clearMinDistanceHighlight').addEventListener('click', () => {
    byId('minDistanceHighlightStart').value = '';
    byId('minDistanceHighlightEnd').value = '';
    byId('minDistanceHighlightIds').value = '';
    refreshMinDistanceHighlight();
  });

  for (const controlId of ['recursiveGroupLimit', 'recursiveGroupMaxAngle', 'recursiveGroupBlockSize', 'recursiveGroupComputeDevice']) {
    byId(controlId).addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        computeRecursiveAngleGroup();
      }
    });
    byId(controlId).addEventListener('input', () => {
      resetRecursiveGroupOutput('Controls changed. Click Build recursive group to refresh the graph.');
    });
  }
  for (const controlId of ['linearTransformType', 'linearTransformLimit', 'linearTransformRidgeLambda', 'linearTransformScale']) {
    byId(controlId).addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        computeLinearTransformNeighbors();
      }
    });
    byId(controlId).addEventListener('input', () => {
      resetLinearTransformOutput('Transform settings changed. Click Run transform neighbors to refresh this list.');
    });
  }

  byId('pairwiseLogScale').addEventListener('change', () => {
    if (state.pairwiseBins && !byId('pairwisePlotCard').classList.contains('hidden')) {
      drawPairwiseAngleBinPlot(state.pairwiseBins, byId('pairwiseLogScale').checked);
    }
  });
});
