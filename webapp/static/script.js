const $ = (id) => document.getElementById(id);
let currentFile = null;

const CLASS_COLOR = {
  Agriculture: '#d99633',
  Forest: '#1e8d68',
  Urban: '#8d9aa5',
  Water: '#3b6ea8'
};

const DEFAULT_KERNELS = [3, 5];
const DEFAULT_SHARPEN_OPTIONS = [
  { value: 'true', label: 'Unsharp 1.5×' },
  { value: 'false', label: 'Off' }
];

async function loadConfig() {
  const cfg = await (await fetch('/api/config')).json();
  $('resolution').innerHTML = cfg.resolutions.map((r) => `<option value="${r}">${r}</option>`).join('');
  $('kernel').innerHTML = cfg.kernels.map((k) => `<option value="${k}">${k === 3 ? 'Bilateral 3×3' : 'Bilateral 5×5'}</option>`).join('');
  $('contrast').innerHTML = cfg.contrasts.map((c) => `<option value="${c}">${c}</option>`).join('');
  $('sharpen').innerHTML = DEFAULT_SHARPEN_OPTIONS.map((item) => `<option value="${item.value}">${item.label}</option>`).join('');

  $('classList').innerHTML = cfg.classes.map((c) => {
    const color = CLASS_COLOR[c.name] || c.color || '#7a838d';
    return `<li data-name="${c.name}"><span class="class-swatch" style="background:${color}"></span>${c.icon} ${c.name}</li>`;
  }).join('');
}

function setActiveClass(name) {
  document.querySelectorAll('#classList li').forEach((li) => {
    li.classList.toggle('active', li.dataset.name === name);
  });
}

function mean(arr) {
  const total = arr.reduce((sum, v) => sum + v, 0) || 1;
  const meanValue = arr.reduce((sum, v, idx) => sum + (idx * v), 0) / total;
  const variance = arr.reduce((sum, v, idx) => sum + (v * (idx - meanValue) ** 2), 0) / total;
  return { mean: meanValue, std: Math.sqrt(variance) };
}

function drawHistogram(canvas, before, after, color) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth * 2;
  const h = canvas.height = canvas.clientHeight * 2;
  ctx.clearRect(0, 0, w, h);

  const plot = (values, stroke, dashed) => {
    const max = Math.max(...values, 1);
    ctx.beginPath();
    ctx.setLineDash(dashed ? [6, 6] : []);
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - (v / max) * (h - 8) - 4;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();
  };

  plot(before, '#b9c2ca', true);
  plot(after, color, false);
}

function renderStages(stages) {
  $('stagesGrid').innerHTML = stages.map((s) => `
    <div class="card stage-card">
      <img src="${s.image}" alt="${s.name}">
      <div class="stage-body">
        <div class="stage-id">Stage ${s.id.replace(/^0+/, '')}</div>
        <div class="stage-name">${s.name}</div>
        <div class="stage-text">${s.text}</div>
      </div>
    </div>
  `).join('');
}

function renderHistograms(before, after) {
  const series = [
    { name: 'Red', key: 'r', color: '#d86262' },
    { name: 'Green', key: 'g', color: '#1e8d68' },
    { name: 'Blue', key: 'b', color: '#3b6ea8' }
  ];

  $('histGrid').innerHTML = series.map((s) => {
    const b = before.rgb[s.key];
    const a = after.rgb[s.key];
    const stats = mean(a);
    const scale = 256 / a.length;
    return `
      <div class="card hist-card">
        <div class="hist-card-head">
          <div class="hist-label"><span class="hist-dot" style="background:${s.color}"></span>${s.name}</div>
          <div class="hist-stats">μ ${((stats.mean * scale)).toFixed(1)} · σ ${(stats.std * scale).toFixed(1)}</div>
        </div>
        <canvas></canvas>
        <div class="hist-axis"><span>0</span><span>128</span><span>255</span></div>
      </div>
    `;
  }).join('');

  const canvases = document.querySelectorAll('#histGrid canvas');
  series.forEach((s, index) => {
    const canvas = canvases[index];
    if (!canvas) return;
    const b = before.rgb[s.key];
    const a = after.rgb[s.key];
    requestAnimationFrame(() => drawHistogram(canvas, b, a, s.color));
  });
}

function renderFeatures(featureGroups) {
  const orderedGroups = featureGroups.filter((group) => group.group === 'COLOR' || group.group === 'HISTOGRAM');

  $('featureGrid').innerHTML = orderedGroups.map((group) => {
    const title = group.group === 'COLOR' ? 'Color' : 'Histogram';
    return `
      <div class="feature-panel">
        <h3>${title}</h3>
        ${group.items.map((item) => `
          <div class="feature-row">
            <div class="label">${item.label}</div>
            <div class="value">${item.value}</div>
          </div>
        `).join('')}
      </div>
    `;
  }).join('');
}

function inferMetaChip(value, fallback = 'Not available') {
  if (value == null || value === '' || value === 'undefined') return fallback;
  return value;
}

function renderInput(input) {
  $('inputImg').src = input.image;
  $('m-filename').textContent = input.filename || 'Not available';
  $('m-res').textContent = input.width && input.height ? `${input.width} × ${input.height} px` : 'Not available';
  $('m-format').textContent = input.format || 'Not available';
  $('m-channels').textContent = input.channels ? `${input.channels} Channels` : 'Not available';

  $('miniFileTag').textContent = (input.format || 'TIFF').toUpperCase();
  $('imgMetaText').textContent = input.width && input.height
    ? `Zoom Inspection · ${input.width}×${input.height}`
    : 'Zoom Inspection';

}

function renderResult(result) {
  const scores = Object.entries(result.scores || {}).sort((a, b) => b[1] - a[1]);
  const topClass = scores[0]?.[0] || result.predicted_class;
  const topScore = scores[0]?.[1] ?? Number(result.confidence || 0) * 100;
  const secondScore = scores[1]?.[1] ?? 0;
  const confidencePct = Math.round((result.confidence || 0) * 100);

  $('resultClass').textContent = String(topClass).toUpperCase();
  $('resultBadge').textContent = result.icon || '•';
  $('resultCode').textContent = `${topClass} rule winner`;
  $('confVal').textContent = `${confidencePct}%`;
  $('confBar').style.width = `${Math.min(100, Math.max(0, confidencePct))}%`;
  $('confMeta').textContent = confidencePct >= 75 ? 'Consistent' : 'Borderline';
  $('confState').textContent = confidencePct >= 75 ? 'Consistent' : 'Review';

  const uncertainty = Math.max(0, Math.round((1 - (result.confidence || 0)) * 100));
  $('uncVal').textContent = `${uncertainty} percentage points`;
  $('uncMeta').textContent = `Based on score margin`;

  const tbody = $('evidenceTable');
  tbody.innerHTML = scores.map(([cls, score], idx) => {
    const color = CLASS_COLOR[cls] || '#7a838d';
    const marginValue = idx === 0 ? (score - secondScore) : (score - topScore);
    const margin = idx === 0 ? `+${marginValue.toFixed(0)} (Class Lead)` : `${marginValue.toFixed(0)}`;
    const className = cls === 'Forest' ? 'Forest' : cls;
    return `
      <tr>
        <td><div class="class-name"><span class="class-dot" style="background:${color}"></span>${className}</div></td>
        <td><span class="score-text score-number">${score.toFixed(0)}</span></td>
        <td>
          <div class="evidence-score-wrap">
            <div class="evidence-track"><span style="width:${Math.min(100, Math.max(0, score))}%; background:${color};"></span></div>
            <span class="score-text">${score.toFixed(0)} / 100</span>
          </div>
        </td>
        <td class="${marginValue >= 0 ? 'margin-pos' : 'margin-neg'}">${margin}</td>
      </tr>
    `;
  }).join('');
}

function renderReasons(details, fallbackReasons) {
  const iconMap = { support: '✓', against: '×', info: 'i' };
  const rows = details.length ? details : fallbackReasons.map(([kind, text]) => ({
    kind,
    description: text,
    actual: null,
    rule: 'Classifier evidence'
  }));

  $('reasons').innerHTML = rows.map((item) => {
    const actual = item.actual_label || (item.actual == null ? 'Measured' : `Actual term: ${Number(item.actual).toFixed(2)}`);
    const rule = item.rule || `Weight: ${Number(item.weight).toFixed(2)}`;
    const title = String(item.name || item.description).replaceAll('_', ' ');
    const sentence = String(item.description || '').replace(/^\w[\w -]*:\s*/, '');
    return `
      <div class="reason-row">
        <div class="reason-main">
          <div class="reason-icon">${iconMap[item.kind] || '✓'}</div>
          <div class="reason-text"><strong>${title}</strong><span>${sentence}</span></div>
        </div>
        <div class="reason-meta"><span>${actual}</span><span>${rule}</span></div>
      </div>
    `;
  }).join('');
}

async function analyze() {
  if (!currentFile) return;
  $('statusPill').textContent = '● Analyzing…';
  $('errorBox').classList.add('hidden');

  const fd = new FormData();
  fd.append('image', currentFile);
  fd.append('resolution', $('resolution').value);
  fd.append('kernel', $('kernel').value);
  fd.append('contrast', $('contrast').value);
  fd.append('sharpen', $('sharpen').value);

  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed');

    $('empty-state').classList.add('hidden');
    $('results').classList.remove('hidden');
    renderInput(data.input);
    renderStages(data.stages);
    renderHistograms(data.hist_before, data.hist_after);
    renderFeatures(data.feature_groups || []);
    renderResult(data.result || {});
    renderReasons(data.result?.reason_details || [], data.result?.reasons || []);
    $('statusPill').textContent = '● Ready';
  } catch (error) {
    $('errorBox').textContent = error.message;
    $('errorBox').classList.remove('hidden');
    $('statusPill').textContent = '● Error';
  }
}

function handleFile(file) {
  currentFile = file;
  analyze();
}

window.addEventListener('DOMContentLoaded', async () => {
  await loadConfig();
  $('browseBtn').addEventListener('click', () => $('fileInput').click());
  $('fileInput').addEventListener('change', (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) handleFile(file);
  });
  $('rerunBtn').addEventListener('click', () => {
    if (currentFile) analyze();
  });

  ['dragover', 'drop'].forEach((evt) => {
    $('dropzone').addEventListener(evt, (event) => event.preventDefault());
  });

  $('dropzone').addEventListener('drop', (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) handleFile(file);
  });
});
