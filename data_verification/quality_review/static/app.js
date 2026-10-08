'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const NAMES = {0: 'Bad', 1: 'OK', 2: 'Good'};
const KEY_TO_VALUE = {b: 0, o: 1, g: 2, 0: 0, 1: 1, 2: 2};
const ICON_LOCK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>';
const ICON_CHECK = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
// Sheets are 4 x 2 panels of 256 px under an 18 px label strip (render_aligned_heatmaps._sheet).
const ROW_Y = ['6.1644%', '100%'];

const app = {config: null, axes: [], user: null, view: null, s: null, lightbox: null, historyOffset: 0};

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {'Content-Type': 'application/json', ...(options.headers || {})}});
  let body = null;
  try { body = await response.json(); } catch (_) { /* empty body */ }
  if (!response.ok) {
    const detail = body && body.detail;
    throw new Error((Array.isArray(detail) ? detail.map(d => d.msg).join('; ') : detail) || `${response.status} ${response.statusText}`);
  }
  return body;
}
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const pill = value => value == null ? '<span class="muted">—</span>' : `<span class="pill v${value}">${NAMES[value]}</span>`;
const pct = (value, digits = 1) => value == null ? '—' : `${value.toFixed(digits)}%`;
const fixed = (value, digits = 2) => value == null ? '—' : value.toFixed(digits);
const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`);
const locked = text => `<div class="locked-bar">${ICON_LOCK}<span>${esc(text)}</span></div>`;
const tagLabel = (axisId, tag) => (app.axes.find(a => a.id === axisId).tags.find(t => t.id === tag) || {label: tag}).label;

function toast(message, isError = false) {
  const node = $('#toast');
  node.textContent = message;
  node.classList.toggle('error', isError);
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, isError ? 4500 : 1600);
}

// ------------------------------------------------------------------ session

async function boot() {
  app.config = await api('/api/config');
  app.axes = app.config.rubric.axes;
  buildGuide();
  try {
    enterShell(await api('/api/me'));
  } catch (_) {
    $('#code-field').hidden = !app.config.needs_code;
    $('#code').required = app.config.needs_code;
    $('#signin').hidden = false;
    $('#username').focus();
  }
}

function enterShell(me) {
  app.user = me.username;
  $('#signin').hidden = true;
  $('#shell').hidden = false;
  $('#who').textContent = me.username;
  $('#avatar').textContent = me.username.slice(0, 1);
  renderProgress(me.progress);
  show('review');
}

$('#signin-form').addEventListener('submit', async event => {
  event.preventDefault();
  $('#signin-error').textContent = '';
  try {
    await api('/api/login', {method: 'POST', body: JSON.stringify({username: $('#username').value, code: $('#code').value})});
    enterShell(await api('/api/me'));
  } catch (error) { $('#signin-error').textContent = error.message; }
});
$('#signout').addEventListener('click', async () => { await api('/api/logout', {method: 'POST'}); location.reload(); });

function renderProgress(p) {
  if (!p) return;
  $('#meter-mine').textContent = p.mine;
  $('#meter-total').textContent = p.samples;
  $('#meter-fill').style.width = `${p.samples ? 100 * p.mine / p.samples : 0}%`;
}

// ------------------------------------------------------------------ navigation

$$('.tab').forEach(tab => tab.addEventListener('click', () => show(tab.dataset.view)));

async function show(view) {
  app.view = view;
  const tab = view === 'revision' ? 'history' : view;
  $$('.tab').forEach(node => node.classList.toggle('active', node.dataset.view === tab));
  $('#view-workspace').hidden = !['review', 'revision'].includes(view);
  $('#view-progress').hidden = view !== 'progress';
  $('#view-history').hidden = view !== 'history';
  try {
    if (view === 'review') await loadBlind();
    if (view === 'progress') await loadProgress();
    if (view === 'history') await loadHistory(true);
  } catch (error) { toast(error.message, true); }
}

function showEmpty(title, text) {
  app.s = null;
  $('#workspace').hidden = true;
  $('#workspace-empty').hidden = false;
  $('#empty-title').textContent = title;
  $('#empty-text').textContent = text;
}

// ------------------------------------------------------------------ loading items

function startItem(mode, sample) {
  const labels = {}, tags = {};
  for (const axis of app.axes) { labels[axis.id] = null; tags[axis.id] = new Set(); }
  // A heatmap carried over from the pilot starts pre-filled with this rater's own pilot rating.
  const carried = sample.carried || {};
  for (const [axis, plan] of Object.entries(carried)) { labels[axis] = plan.label; tags[axis] = new Set(plan.tags); }
  app.s = {mode, sample, steps: app.axes, carried, labels, tags, revealed: mode === 'blind' ? 1 : app.axes.length,
           open: null, shown: 0, started: performance.now(), key: uuid(), busy: false};
  $('#note').value = '';
  $('#form-error').textContent = '';
  $('#workspace').hidden = false;
  $('#workspace-empty').hidden = true;
  for (const kind of ['A', 'B']) { const image = new Image(); image.src = sample.images[kind]; }
  window.scrollTo({top: 0});
}

async function loadBlind(skip = null) {
  const data = await api(`/api/queue/next${skip ? `?skip=${encodeURIComponent(skip)}` : ''}`, {method: 'POST'});
  renderProgress(data.progress);
  if (!data.sample) return showEmpty('All done', 'You have rated every pair. Thank you!');
  startItem('blind', data.sample);
  app.s.open = app.s.steps[0].id;
  render();
}

async function loadRevision(sampleId) {
  const data = await api(`/api/mine?sample_id=${encodeURIComponent(sampleId)}`);
  await show('revision');
  startItem('revision', data.sample);
  for (const axis of app.s.steps) {
    app.s.labels[axis.id] = data.rating.labels[axis.id];
    app.s.tags[axis.id] = new Set(data.rating.tags[axis.id] || []);
  }
  $('#note').value = data.rating.comment || '';
  render();
}

// ------------------------------------------------------------------ rating state

const stepIndex = axisId => app.s.steps.findIndex(axis => axis.id === axisId);
// Evidence for an axis is visible once its step is revealed.
const shown = axisId => stepIndex(axisId) < app.s.revealed;
const sameTags = (a, b) => a.length === b.size && a.every(tag => b.has(tag));
// A carried pilot rating is only sent back when the rater actually changed it.
const changed = axisId => {
  const plan = app.s.carried[axisId];
  return !plan || plan.label !== app.s.labels[axisId] || !sameTags(plan.tags, app.s.tags[axisId]);
};

function complete(axisId) {
  const value = app.s.labels[axisId];
  return value === 2 || (value != null && app.s.tags[axisId].size > 0);
}
const allComplete = () => app.s.steps.every(axis => complete(axis.id));

function nextOpen(after = -1) {
  const ids = app.s.steps.map(axis => axis.id);
  const candidates = [...ids.slice(after + 1), ...ids.slice(0, after + 1)];
  return candidates.find(id => stepIndex(id) < app.s.revealed && !complete(id)) || null;
}

function choose(axisId, value) {
  const s = app.s;
  const index = stepIndex(axisId);
  if (!s || index < 0 || index >= s.revealed) return;
  s.labels[axisId] = value;
  if (value === 2) s.tags[axisId].clear();
  s.revealed = Math.max(s.revealed, Math.min(index + 2, s.steps.length));
  s.open = value === 2 ? nextOpen(index) : axisId;
  $('#form-error').textContent = '';
  render();
}

function toggleTag(axisId, tag) {
  const chosen = app.s.tags[axisId];
  chosen.has(tag) ? chosen.delete(tag) : chosen.add(tag);
  $('#form-error').textContent = '';
  render();
}

function advance() {
  const s = app.s;
  if (allComplete()) return submit();
  if (s.open && complete(s.open)) {
    s.open = nextOpen(stepIndex(s.open));
    return render();
  }
  const target = s.open || nextOpen();
  if (target) {
    const axis = app.s.steps[stepIndex(target)];
    toast(s.labels[target] == null ? `Rate ${axis.title.toLowerCase()} first.`
      : `Pick at least one reason for ${axis.title.toLowerCase()}.`, true);
  }
}

async function submit() {
  const s = app.s;
  if (!s || s.busy || !allComplete()) return;
  s.busy = true;
  renderActions();
  const axes = s.steps.map(axis => axis.id).filter(changed);
  const payload = {
    sample_id: s.sample.sample_id,
    labels: Object.fromEntries(axes.map(id => [id, s.labels[id]])),
    tags: Object.fromEntries(axes.map(id => [id, [...s.tags[id]]])),
    comment: $('#note').value,
    elapsed_ms: Math.round(performance.now() - s.started),
    idempotency_key: s.key,
  };
  try {
    await api(s.mode === 'revision' ? '/api/revisions' : '/api/annotations', {method: 'POST', body: JSON.stringify(payload)});
    if (s.mode === 'revision') {
      toast('Rating updated');
      await show('history');
    } else {
      toast('Saved');
      await loadBlind();
    }
  } catch (error) {
    s.busy = false;
    $('#form-error').textContent = error.message;
    renderActions();
  }
}

// ------------------------------------------------------------------ rendering

function render() {
  if (app.s.shown !== app.s.revealed) {
    renderEvidence();
    app.s.shown = app.s.revealed;
  }
  renderSteps();
  renderActions();
}

function renderEvidence() {
  const {sample, mode} = app.s;
  $('#object-name').textContent = sample.object_name;
  $('#sample-id').textContent = sample.sample_id;
  $('#sample-id').title = sample.sample_id;
  $('#task-text').textContent = sample.task;
  $('#mode-chip').hidden = mode === 'blind';
  $('#mode-chip').textContent = 'Editing';
  $('#roles').innerHTML = shown('role') ? sample.roles.map(roleCard).join('')
    : locked('Hand roles appear after step 1. Judge the task from the views first.');
  renderViews();
}

function roleCard(role) {
  const hand = role.id === 'A' ? 'a' : 'b';
  return `<div class="role ${hand}">
    <div class="role-head"><span class="dot-${hand}"></span><span class="hand">Hand ${esc(role.id)}</span>
      <span class="verb">${esc(role.role)}</span><span class="target">${esc(role.target)}</span></div>
    <dl><dt>at</dt><dd>${esc(role.contact_region)}</dd><dt>to</dt><dd>${esc(role.function)}</dd></dl>
  </div>`;
}

function tile(kind, i, label = '') {
  const url = app.s.sample.images[kind];
  const style = `--src:url('${url}');--x:${(i % 4) * 100 / 3}%;--y:${ROW_Y[Math.floor(i / 4)]}`;
  return `<button type="button" class="tile${kind === 'rgb' ? '' : ' heat'}" data-i="${i}" style="${esc(style)}"
    aria-label="${esc(app.s.sample.view_labels[i] || 'view')}">${label ? `<span class="tile-label">${esc(label)}</span>` : ''}</button>`;
}

function renderViews() {
  const s = app.s;
  const labels = s.sample.view_labels;
  const root = $('#views');
  if (!shown('heatmap')) {
    root.className = 'views grid';
    root.innerHTML = labels.map((label, i) => tile('rgb', i, label)).join('') +
      locked(`Heatmaps appear after step ${stepIndex('heatmap')}.`);
    return;
  }
  const roles = Object.fromEntries(s.sample.roles.map(role => [role.id, role]));
  const row = (kind, title, dot, sub) => `<div class="row-head"><span>${dot}${title}</span><small>${esc(sub)}</small></div>` +
    labels.map((_, i) => tile(kind, i)).join('');
  root.className = 'views matrix';
  root.style.setProperty('--n', labels.length);
  root.innerHTML = '<div></div>' + labels.map(label => `<div class="col-head">${esc(label)}</div>`).join('') +
    row('rgb', 'RGB', '', 'views') +
    row('A', 'Hand A', '<i class="dot-a"></i>', roles.A ? roles.A.role : '') +
    row('B', 'Hand B', '<i class="dot-b"></i>', roles.B ? roles.B.role : '');
}

function renderSteps() {
  $('#steps').innerHTML = app.s.steps.map(stepHtml).join('');
}

function stepHtml(axis, index) {
  const s = app.s, id = axis.id, value = s.labels[id];
  const pilot = s.carried[id];
  const lockedStep = index >= s.revealed;
  const state = lockedStep ? 'locked' : s.open === id ? 'open' : complete(id) ? 'done' : 'pending';
  const reasonCount = s.tags[id].size;
  let result = '';
  if (lockedStep) result = `${ICON_LOCK}<span>after step ${index}</span>`;
  else if (value != null && s.open !== id) result = pill(value) + (pilot && !changed(id) ? '<span>pilot</span>'
    : reasonCount ? `<span>${reasonCount} reason${reasonCount > 1 ? 's' : ''}</span>` : '');
  const options = [2, 1, 0].map(v => `<button type="button" class="option v${v}${value === v ? ' selected' : ''}" data-axis="${id}" data-v="${v}" role="radio" aria-checked="${value === v}">
      <span class="o-dot"></span><span><b>${NAMES[v]}</b><small>${esc(axis.verdicts[v])}</small></span><kbd>${esc(app.config.rubric.scale[v].key)}</kbd></button>`).join('');
  const reasons = value != null && value < 2 ? `<div class="reasons">
      <p class="reasons-label">Why ${NAMES[value]}? <small>choose at least one</small></p>
      <div class="chips">${axis.tags.map(tag => `<button type="button" class="chip${s.tags[id].has(tag.id) ? ' on' : ''}" data-axis="${id}" data-tag="${tag.id}">${esc(tag.label)}</button>`).join('')}</div>
    </div>` : '';
  const pilotNote = pilot ? '<p class="pilot-note">Pre-filled with your pilot rating. Change it only if you now see it differently.</p>' : '';
  return `<section class="step" data-axis="${id}" data-state="${state}">
    <button type="button" class="step-head" data-open="${id}"${lockedStep ? ' disabled' : ''}>
      <span class="step-num">${state === 'done' ? ICON_CHECK : index + 1}</span>
      <span><span class="step-title">${esc(axis.title)}</span><span class="step-q">${esc(axis.question)}</span></span>
      <span class="step-result">${result}</span>
    </button>
    <div class="step-body${value != null ? ' decided' : ''}">${pilotNote}${axis.guidance ? `<p class="step-guidance">${esc(axis.guidance)}</p>` : ''}
      <div class="options" role="radiogroup" aria-label="${esc(axis.title)}">${options}</div>${reasons}</div>
  </section>`;
}

function renderActions() {
  const s = app.s;
  $('#submit').innerHTML = `${s.mode === 'revision' ? 'Save changes' : 'Submit'} <kbd>↵</kbd>`;
  $('#submit').disabled = s.busy || !allComplete();
  $('#skip').textContent = s.mode === 'revision' ? 'Cancel' : 'Skip';
}

$('#steps').addEventListener('click', event => {
  if (!app.s) return;
  const option = event.target.closest('.option');
  if (option) return choose(option.dataset.axis, Number(option.dataset.v));
  const chip = event.target.closest('.chip');
  if (chip) return toggleTag(chip.dataset.axis, chip.dataset.tag);
  const head = event.target.closest('.step-head');
  if (head && !head.disabled) {
    app.s.open = app.s.open === head.dataset.open ? null : head.dataset.open;
    render();
  }
});
$('#submit').addEventListener('click', submit);
$('#skip').addEventListener('click', async () => {
  const s = app.s;
  if (!s) return;
  try {
    if (s.mode === 'blind') await loadBlind(s.sample.sample_id);
    else await show('history');
  } catch (error) { toast(error.message, true); }
});

// ------------------------------------------------------------------ views: hover + lightbox

$('#views').addEventListener('mouseover', event => {
  const target = event.target.closest('.tile');
  const matrix = $('#views').classList.contains('matrix');
  $$('#views .tile').forEach(node => node.classList.toggle('hl', matrix && !!target && node !== target && node.dataset.i === target.dataset.i));
});
$('#views').addEventListener('mouseleave', () => $$('#views .tile').forEach(node => node.classList.remove('hl')));
$('#views').addEventListener('click', event => {
  const target = event.target.closest('.tile');
  if (target) openLightbox(Number(target.dataset.i));
});

function openLightbox(i) {
  app.lightbox = i;
  $('#lightbox').hidden = false;
  renderLightbox();
}
function closeLightbox() {
  app.lightbox = null;
  $('#lightbox').hidden = true;
}
function moveLightbox(delta) {
  const n = app.s.sample.view_labels.length;
  app.lightbox = (app.lightbox + delta + n) % n;
  renderLightbox();
}
function renderLightbox() {
  const s = app.s, i = app.lightbox;
  const kinds = shown('heatmap')
    ? [['rgb', 'RGB', ''], ['A', 'Hand A', '<i class="dot-a"></i>'], ['B', 'Hand B', '<i class="dot-b"></i>']]
    : [['rgb', 'RGB', '']];
  const size = Math.floor(Math.min((window.innerWidth - 170 - 16 * (kinds.length - 1)) / kinds.length,
                                   window.innerHeight - 140, 560));
  $('#lightbox-title').textContent = `${s.sample.view_labels[i]} · ${i + 1} / ${s.sample.view_labels.length}`;
  $('#lightbox-panels').style.setProperty('--big', `${size}px`);
  $('#lightbox-panels').innerHTML = kinds.map(([kind, title, dot]) =>
    `<figure><figcaption>${dot}${title}</figcaption>${tile(kind, i)}</figure>`).join('');
}
$('#lightbox-close').addEventListener('click', closeLightbox);
$('#lightbox-prev').addEventListener('click', () => moveLightbox(-1));
$('#lightbox-next').addEventListener('click', () => moveLightbox(1));
window.addEventListener('resize', () => { if (app.lightbox != null) renderLightbox(); });

// ------------------------------------------------------------------ keyboard

document.addEventListener('keydown', event => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (!$('#lightbox').hidden) {
    if (event.key === 'Escape') closeLightbox();
    if (event.key === 'ArrowRight') moveLightbox(1);
    if (event.key === 'ArrowLeft') moveLightbox(-1);
    return;
  }
  if (event.key === 'Escape' && !$('#guide').hidden) { $('#guide').hidden = true; return; }
  if (event.target.matches && event.target.matches('input, textarea')) {
    if (event.key === 'Escape') event.target.blur();
    return;
  }
  if (event.key === '?') { $('#guide').hidden = !$('#guide').hidden; return; }
  const s = app.s;
  if (!s || $('#view-workspace').hidden || $('#workspace').hidden) return;
  const key = event.key.toLowerCase();
  if (key in KEY_TO_VALUE) {
    const target = s.open || nextOpen();
    if (target) { event.preventDefault(); choose(target, KEY_TO_VALUE[key]); }
  } else if (event.key === 'Enter') {
    event.preventDefault();
    advance();
  }
});

// ------------------------------------------------------------------ guide

function buildGuide() {
  const rubric = app.config.rubric;
  $('#guide-body').innerHTML = `
    <h3>Principles</h3>
    <ol>${rubric.principles.map(text => `<li>${esc(text)}</li>`).join('')}</ol>
    <h3>Criteria, revealed in order</h3>
    ${rubric.axes.map((axis, i) => `<div class="guide-axis">
      <h4><span class="step-num">${i + 1}</span>${esc(axis.title)}</h4>
      <p class="q">${esc(axis.question)}</p>
      <p class="ev">Evidence: ${esc(axis.evidence)}</p>
      ${[2, 1, 0].map(v => `<div class="g-verdict">${pill(v)}<span>${esc(axis.verdicts[v])}</span></div>`).join('')}
      ${axis.guidance ? `<p class="ev" style="margin-top:10px">${esc(axis.guidance)}</p>` : ''}
      <div class="g-tags">${axis.tags.map(tag => `<span class="tag">${esc(tag.label)}</span>`).join('')}</div>
    </div>`).join('')}
    <h3>Keyboard</h3>
    <div class="shortcuts">
      <span><kbd>G</kbd> <kbd>O</kbd> <kbd>B</kbd></span><span>Good, OK or Bad for the open step (<kbd>2</kbd> <kbd>1</kbd> <kbd>0</kbd> also work)</span>
      <span><kbd>↵</kbd></span><span>Next step, or submit once every step is rated</span>
      <span><kbd>Esc</kbd></span><span>Close a panel</span>
      <span><kbd>?</kbd></span><span>Open or close this guide</span>
    </div>
    <p class="muted" style="margin-top:20px;font-size:12px">Rubric ${esc(rubric.version)}. Every rater scores every pair on
      their own; each criterion's result is the average of the raters' scores. If you rated heatmaps in the pilot,
      your pilot rating is pre-filled wherever the released heatmap is unchanged.</p>`;
}
$('#guide-open').addEventListener('click', () => { $('#guide').hidden = !$('#guide').hidden; });
$('#guide-close').addEventListener('click', () => { $('#guide').hidden = true; });

// ------------------------------------------------------------------ progress + history

async function loadProgress() {
  const data = await api('/api/dashboard');
  const p = data.progress, results = data.results, agreement = data.agreement;
  renderProgress(p);
  const team = data.raters.reduce((sum, r) => sum + r.rated, 0);
  $('#kpis').innerHTML = [
    ['Your progress', `${p.mine} / ${p.samples}`, `${pct(p.samples ? 100 * p.mine / p.samples : 0, 0)} of the pairs`],
    ['Raters', data.raters.filter(r => r.rated).length, 'who have started'],
    ['Team ratings', team, 'pairs scored, all raters together'],
  ].map(([label, value, sub]) => `<div class="card kpi"><span>${label}</span><b>${value}</b><small>${sub}</small></div>`).join('');

  const distribution = summary => {
    const [bad, ok, good] = summary.distribution, n = bad + ok + good || 1;
    return `<div class="dist"><i class="d0" style="width:${100 * bad / n}%"></i><i class="d1" style="width:${100 * ok / n}%"></i><i class="d2" style="width:${100 * good / n}%"></i></div>
      <div class="dist-legend"><span>Bad ${bad}</span><span>OK ${ok}</span><span>Good ${good}</span></div>`;
  };
  const row = (title, summary, reliability) => `<tr>
      <td>${title}</td><td class="num big">${summary.score == null ? '—' : summary.score.toFixed(1)}${
        summary.score_sd == null ? '' : `<small class="muted"> ± ${summary.score_sd.toFixed(1)}</small>`}</td>
      <td class="num">${pct(summary.good_pct)}</td><td>${distribution(summary)}</td>
      <td class="num">${fixed(reliability.alpha_ordinal)}</td>
      <td class="num">${pct(reliability.exact_agreement == null ? null : 100 * reliability.exact_agreement, 0)}</td></tr>`;
  $('#results').innerHTML = `<table class="data"><thead><tr><th>Criterion</th><th class="num">Score</th><th class="num">Good</th>
      <th>Distribution</th><th class="num">Krippendorff α</th><th class="num">Exact agreement</th></tr></thead><tbody>
      ${app.axes.map(axis => row(esc(axis.title), results.axes[axis.id], agreement[axis.id])).join('')}
      </tbody></table>
    <p class="muted" style="margin:12px 0 0;font-size:12px">Score = each rater's mean rating (Bad 0, OK 1, Good 2) ÷ 2 × 100,
      averaged over the raters (± is the spread across raters); Good is averaged the same way. Until everyone has finished,
      raters cover different pairs, so read the final numbers at the end.</p>`;

  $('#carried-note').hidden = !data.carried;
  if (data.carried) {
    $('#carried-note').innerHTML = `The two pilot raters' heatmap ratings are carried over for <b>${data.carried.carried}</b>
      pairs (pre-filled and editable for them). <b>${data.carried.rerated}</b> pairs whose released heatmap changed since
      the pilot are rated fresh by everyone.`;
  }
  $('#raters').innerHTML = data.raters.length ? `<table class="data"><thead><tr><th>Rater</th><th class="num">Progress</th>
      <th class="num">Median time</th>${app.axes.map(axis => `<th class="num">${esc(axis.short)}</th>`).join('')}</tr></thead><tbody>
      ${data.raters.map(r => `<tr><td>${esc(r.username)}</td><td class="num">${r.rated} / ${p.samples}</td>
        <td class="num">${r.median_seconds == null ? '—' : `${r.median_seconds.toFixed(0)} s`}</td>
        ${app.axes.map(axis => `<td class="num">${r.score[axis.id] == null ? '—' : r.score[axis.id].toFixed(1)}</td>`).join('')}</tr>`).join('')}
      </tbody></table>`
    : '<p class="muted">No ratings yet.</p>';

  $('#reasons').innerHTML = app.axes.map(axis => {
    const rows = (data.tags[axis.id] || []).slice(0, 5);
    return `<p class="reason-axis">${esc(axis.title)}</p>` + (rows.length
      ? rows.map(([tag, count]) => `<div class="reason-row"><span>${esc(tagLabel(axis.id, tag))}</span><b>${count}</b></div>`).join('')
      : '<div class="reason-row"><span class="muted">None yet</span><span></span></div>');
  }).join('');
}

async function loadHistory(reset) {
  if (reset) {
    app.historyOffset = 0;
    $('#history').innerHTML = `<div class="history-row head"><span class="eyebrow">Pair</span>
      <div class="pills">${app.axes.map(axis => `<span class="eyebrow">${esc(axis.short)}</span>`).join('')}</div><span></span></div>`;
  }
  const data = await api(`/api/history?offset=${app.historyOffset}&limit=50`);
  app.historyOffset += data.items.length;
  if (!data.total) $('#history').innerHTML = '<div class="history-empty">No ratings yet. Start in the Review tab.</div>';
  $('#history').insertAdjacentHTML('beforeend', data.items.map(item => `<div class="history-row">
      <div><div class="t">${esc(item.task)}</div>
        <div class="s">${esc(item.object_name)} · ${new Date(item.created_at).toLocaleString()}${item.kind === 'revision' ? ' · edited' : ''}</div></div>
      <div class="pills">${app.axes.map(axis => item.pilot.includes(axis.id)
        ? `<span class="from-pilot" title="${esc(axis.title)}: your pilot rating">${pill(item.labels[axis.id])}</span>`
        : `<span title="${esc(axis.title)}">${pill(item.labels[axis.id])}</span>`).join('')}</div>
      <button class="btn ghost sm" data-edit="${esc(item.sample_id)}">Edit</button>
    </div>`).join(''));
  $('#history-more').hidden = app.historyOffset >= data.total;
}
$('#history').addEventListener('click', event => {
  const button = event.target.closest('[data-edit]');
  if (button) loadRevision(button.dataset.edit).catch(error => toast(error.message, true));
});
$('#history-more').addEventListener('click', () => loadHistory(false).catch(error => toast(error.message, true)));

boot().catch(error => toast(error.message, true));
