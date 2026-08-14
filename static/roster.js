function printRoster() { window.print(); }

const rosterRoot = document.getElementById('roster');
const rosterTable = document.querySelector('table.roster');
const siteHeader = document.querySelector('.site-header');
const rosterStickyShield = document.querySelector('[data-roster-sticky-shield]');
const updateRosterLayout = () => {
  if (!rosterTable) return;
  const headerHeight = siteHeader ? Math.ceil(siteHeader.getBoundingClientRect().height) : 0;
  if (rosterStickyShield) rosterStickyShield.style.height = `${headerHeight}px`;
  const scale = Number(rosterRoot?.dataset.scale) || 1;
  rosterTable.style.setProperty('--roster-sticky-top', `${Math.ceil(headerHeight / scale)}px`);
  const width = rosterTable.getBoundingClientRect().width;
  if (width && rosterRoot) rosterRoot.style.setProperty('--roster-status-frame-width', `${Math.ceil(width)}px`);
};
updateRosterLayout();
window.addEventListener('resize', updateRosterLayout, { passive: true });
if (siteHeader && 'ResizeObserver' in window) new ResizeObserver(updateRosterLayout).observe(siteHeader);
if (rosterRoot && 'MutationObserver' in window) {
  new MutationObserver(updateRosterLayout).observe(rosterRoot, { attributes: true, attributeFilter: ['data-scale'] });
}
window.addEventListener('load', () => {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const navigation = performance.getEntriesByType('navigation')[0];
    const csrfToken = rosterRoot?.querySelector('input[name="_csrf_token"]')?.value;
    const telemetryUrl = rosterRoot?.dataset.rosterTelemetryUrl;
    if (!navigation || !csrfToken || !telemetryUrl) return;
    fetch(telemetryUrl, {
      method: 'POST', keepalive: true, credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
      body: JSON.stringify({ render_ms: performance.now(), dom_ms: navigation.domInteractive - navigation.startTime,
        transfer_bytes: navigation.transferSize || navigation.encodedBodySize || 0, decoded_bytes: navigation.decodedBodySize || 0 }),
    }).catch(() => {});
  }));
}, { once: true });

const scrollKey = `atcroster:scroll:${window.location.pathname}`;
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
const rememberPosition = () => sessionStorage.setItem(scrollKey, JSON.stringify({ x: window.scrollX, y: window.scrollY }));
try {
  const saved = sessionStorage.getItem(scrollKey);
  if (saved) requestAnimationFrame(() => { const point = JSON.parse(saved); window.scrollTo(point.x || 0, point.y || 0); });
  sessionStorage.removeItem(scrollKey);
} catch (_error) {}

const saveStatus = document.querySelector('[data-roster-save-status]');
let saveStatusTimer;
const showStatus = (message, error = false) => {
  if (!saveStatus) return;
  clearTimeout(saveStatusTimer);
  saveStatus.textContent = message;
  saveStatus.classList.toggle('is-error', error);
  saveStatus.classList.add('is-visible');
  saveStatusTimer = setTimeout(() => saveStatus.classList.remove('is-visible'), error ? 4500 : 1600);
};
const updateDaySummary = (day, summary) => {
  const totalCell = document.querySelector(`.totcell[data-roster-day="${day}"]`);
  if (!totalCell || !summary) return;
  const total = totalCell.querySelector('.daily-total strong');
  if (total) total.textContent = `Total ${summary.total}`;
  ['M', 'D', 'A', 'N'].forEach((group) => {
    const row = totalCell.querySelector(`[data-roster-count-row="${group}"]`);
    if (!row) return;
    row.classList.remove('green', 'amber', 'red');
    row.classList.add(summary.rag[group]);
    const count = row.querySelector('[data-roster-count]');
    const required = row.querySelector('[data-roster-required]');
    count.textContent = summary.counts[group] || 0;
    required.textContent = summary.required[group] || 0;
    count.classList.toggle('rag-count--over', Number(count.textContent) > Number(required.textContent));
  });
};

const shiftOptions = document.getElementById('roster-shift-options');
const shiftDialog = document.getElementById('roster-shift-dialog');
const shiftForm = shiftDialog?.querySelector('[data-roster-shift-form]');
const shiftSelect = shiftDialog?.querySelector('[data-roster-shift-select]');
const shiftSubmit = shiftForm?.querySelector('button[type="submit"]');
const shiftVersion = shiftDialog?.querySelector('[data-roster-shift-version]');
const shiftTitle = shiftDialog?.querySelector('[data-roster-shift-title]');
let activeShift;
const rosterUndo = document.querySelector('[data-roster-undo]');
const rosterSessionStatus = document.querySelector('[data-roster-session-status]');
const latestRosterChanges = [];
const recordRosterChange = (button, code) => {
  latestRosterChanges.push({ button, code });
  if (latestRosterChanges.length > 20) latestRosterChanges.shift();
  if (rosterUndo) rosterUndo.disabled = false;
  if (rosterSessionStatus) rosterSessionStatus.textContent = 'Saved in this session';
};
const applyShiftPayload = (button, payload, baseline = false) => {
  const cell = button.closest('.cell');
  const code = payload.code || '';
  const prefix = code === 'EM' ? 'm' : (code === 'LA' ? 'a' : code.slice(0, 1).toLowerCase());
  button.textContent = code || '—'; button.dataset.code = code; button.dataset.version = payload.version;
  button.dataset.override = baseline ? '0' : button.dataset.override;
  button.className = `code-input roster-cell-button code-len-${code.length}${code ? ` ${code.toLowerCase()} group-${prefix}` : ''}`;
  button.setAttribute('aria-label', `${button.dataset.staffName} shift on ${button.dataset.dayLabel}: ${code || 'unassigned'}`);
  cell?.querySelectorAll('[data-roster-annotation-open]').forEach((annotationButton) => { annotationButton.dataset.version = payload.version; });
  cell?.classList.toggle('training', Boolean(payload.is_training)); cell?.classList.remove('request-applied'); cell?.querySelector('.request-applied-marker')?.remove();
  if (cell) cell.dataset.rosterCode = code;
  updateDaySummary(payload.day, payload.day_summary);
};
const closeShiftEditor = () => { if (shiftDialog?.open) shiftDialog.close(); activeShift?.focus({ preventScroll: true }); };
const loadShiftOptions = () => {
  if (!shiftSelect || shiftSelect.dataset.optionsLoaded || !shiftOptions) return;
  shiftSelect.append(new Option('—', ''), shiftOptions.content.cloneNode(true));
  shiftSelect.dataset.optionsLoaded = '1';
};
rosterRoot?.addEventListener('click', (event) => {
  const button = event.target.closest('[data-roster-shift-open]');
  if (!button || !shiftForm || !shiftSelect || !shiftVersion) return;
  activeShift = button;
  loadShiftOptions();
  shiftForm.action = button.dataset.action;
  shiftVersion.value = button.dataset.version || '0';
  shiftSelect.querySelector('option[value="__BASELINE__"]')?.remove();
  if (button.dataset.override === '1') shiftSelect.add(new Option('↺ Baseline', '__BASELINE__'));
  shiftSelect.value = button.dataset.code || '';
  if (shiftTitle) shiftTitle.textContent = `${button.dataset.staffName} · ${button.dataset.dayLabel}`;
  shiftDialog.showModal();
  shiftSelect.focus();
});
shiftDialog?.querySelectorAll('[data-roster-shift-close]').forEach((button) => button.addEventListener('click', closeShiftEditor));
shiftDialog?.addEventListener('click', (event) => { if (event.target === shiftDialog) closeShiftEditor(); });
shiftForm?.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!activeShift || shiftForm.dataset.saving === '1') return;
  const data = new FormData(shiftForm);
  const baseline = data.get('code') === '__BASELINE__';
  const previousCode = activeShift.dataset.code || '';
  shiftForm.dataset.saving = '1'; shiftForm.classList.add('is-saving'); shiftSelect.disabled = true; showStatus('Saving…');
  try {
    const response = await fetch(shiftForm.action, { method: 'POST', body: data, credentials: 'same-origin', headers: { Accept: 'application/json', 'X-Requested-With': 'XMLHttpRequest' } });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) throw new Error(payload.error || 'The roster change could not be saved.');
    applyShiftPayload(activeShift, payload, baseline); recordRosterChange(activeShift, previousCode); showStatus('Saved'); closeShiftEditor();
  } catch (error) { showStatus(error.message || 'The roster change could not be saved.', true); }
  finally {
    shiftSelect.disabled = false;
    // The generic form-progress handler disables submit controls. Restore this
    // async editor's control after every response so a retry/edit is possible.
    if (shiftSubmit) { shiftSubmit.disabled = false; shiftSubmit.removeAttribute('aria-disabled'); }
    shiftForm.dataset.saving = '0'; shiftForm.classList.remove('is-saving');
  }
});

const saveShiftFromCommand = async (button, code, recordChange = true) => {
  if (!button || !shiftForm || button.dataset.saving === '1') return;
  const previousCode = button.dataset.code || '';
  const data = new FormData(shiftForm);
  data.set('assignment_version', button.dataset.version || '0'); data.set('code', code);
  button.dataset.saving = '1'; showStatus('Saving…');
  try {
    const response = await fetch(button.dataset.action, { method: 'POST', body: data, credentials: 'same-origin', headers: { Accept: 'application/json', 'X-Requested-With': 'XMLHttpRequest' } });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) throw new Error(payload.error || 'The roster change could not be saved.');
    applyShiftPayload(button, payload, code === '__BASELINE__'); if (recordChange) recordRosterChange(button, previousCode); showStatus('Saved');
  } catch (error) { showStatus(error.message || 'The roster change could not be saved.', true); }
  finally { button.dataset.saving = '0'; }
};
rosterUndo?.addEventListener('click', async () => {
  const change = latestRosterChanges.pop(); if (!change) return;
  await saveShiftFromCommand(change.button, change.code, false);
  rosterUndo.disabled = latestRosterChanges.length === 0;
});

const inspector = document.querySelector('[data-roster-inspector]');
const inspectorTitle = document.querySelector('[data-roster-inspector-title]');
const inspectorDetail = document.querySelector('[data-roster-inspector-detail]');
let selectedRosterCell;
const editableCell = (cell) => cell?.querySelector('[data-roster-shift-open]:not(:disabled)');
const selectRosterCell = (cell) => {
  if (!editableCell(cell)) return;
  document.querySelectorAll('.cell.is-selected').forEach((item) => item.classList.remove('is-selected'));
  selectedRosterCell = cell; cell.classList.add('is-selected');
  if (inspector) { inspector.hidden = false; inspectorTitle.textContent = `${cell.dataset.rosterName} · ${cell.dataset.rosterDay}`; inspectorDetail.textContent = `Shift ${cell.dataset.rosterCode || 'not assigned'}. Press Enter or use Commands to change it.`; }
};
rosterRoot?.addEventListener('click', (event) => { const cell = event.target.closest('.cell'); if (cell && editableCell(cell) && !event.target.closest('[data-roster-shift-open]')) selectRosterCell(cell); });
document.querySelector('[data-roster-inspector-close]')?.addEventListener('click', () => { if (inspector) inspector.hidden = true; selectedRosterCell?.classList.remove('is-selected'); });

const readinessDialog = document.querySelector('[data-roster-readiness-dialog]');
const readinessSummary = document.querySelector('[data-roster-readiness-summary]');
const readinessList = document.querySelector('[data-roster-readiness-list]');
document.querySelectorAll('[data-roster-filter]').forEach((button) => button.addEventListener('click', () => {
  const filter = button.dataset.rosterFilter; const issues = rosterReadinessIssues[filter] || [];
  const matches = issues.map((issue) => issue.staff_id ? document.querySelector(`[data-roster-staff="${issue.staff_id}"][data-roster-day="${issue.date}"]`) : document.querySelector(`.totcell[data-roster-day="${issue.date}"][data-readiness~="${filter}"]`)).filter(Boolean);
  document.querySelectorAll('.cell.is-readiness-match,.totcell.is-readiness-match').forEach((cell) => cell.classList.remove('is-readiness-match'));
  matches.forEach((cell) => cell.classList.add('is-readiness-match'));
  if (readinessSummary && readinessList) {
    readinessSummary.textContent = `${issues.length} issue${issues.length === 1 ? '' : 's'} listed and ${matches.length} location${matches.length === 1 ? '' : 's'} highlighted.`;
    readinessList.replaceChildren(...issues.map((issue) => { const item = document.createElement('li'); const title = document.createElement('strong'); const severity = document.createElement('span'); const link = document.createElement('a'); title.textContent = `${issue.date} · ${issue.shift} · ${issue.controller}`; severity.textContent = ` (${issue.severity}) `; link.href = issue.href; link.textContent = issue.remediation; item.append(title, severity, link); return item; }));
    if (!readinessDialog.open) readinessDialog.showModal();
  }
  matches[0]?.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
}));

const commandPalette = document.querySelector('[data-roster-command-palette]');
const commandInput = document.querySelector('[data-roster-command-input]');
const openCommands = () => { if (!selectedRosterCell) { const first = [...document.querySelectorAll('.cell.editable')].find(editableCell); if (first) selectRosterCell(first); } commandPalette?.showModal(); commandInput?.focus(); };
document.querySelector('[data-roster-command-open]')?.addEventListener('click', openCommands);
commandInput?.addEventListener('keydown', (event) => { if (event.key !== 'Enter') return; event.preventDefault(); saveShiftFromCommand(editableCell(selectedRosterCell), commandInput.value.trim().toUpperCase()); commandPalette?.close(); });
document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k' && commandPalette) { event.preventDefault(); openCommands(); return; }
  if (!selectedRosterCell || event.target.matches('input,select,textarea,button')) return;
  const row = selectedRosterCell.closest('tr'); const offset = selectedRosterCell.cellIndex; let target;
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { const step = event.key === 'ArrowRight' ? 1 : -1; for (let index = offset + step; row?.cells[index]; index += step) { if (editableCell(row.cells[index])) { target = row.cells[index]; break; } } }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { const step = event.key === 'ArrowDown' ? 1 : -1; let next = row; while ((next = step > 0 ? next?.nextElementSibling : next?.previousElementSibling)) { const candidate = next.cells?.[offset]; if (editableCell(candidate)) { target = candidate; break; } } }
  if (target) { event.preventDefault(); selectRosterCell(target); editableCell(target)?.focus({ preventScroll: true }); }
  if (event.key === 'Enter') { event.preventDefault(); editableCell(selectedRosterCell)?.click(); }
});

const annotationDialog = document.getElementById('roster-annotation-dialog');
const annotationForm = annotationDialog?.querySelector('[data-roster-annotation-form]');
let activeAnnotation;
const closeAnnotationEditor = () => { if (annotationDialog?.open) annotationDialog.close(); activeAnnotation?.focus({ preventScroll: true }); };
rosterRoot?.addEventListener('click', (event) => {
  const button = event.target.closest('[data-roster-annotation-open]');
  if (!button || !annotationForm) return;
  activeAnnotation = button; annotationForm.action = button.dataset.action;
  annotationForm.querySelector('[data-roster-annotation-version]').value = button.dataset.version || '0';
  const select = annotationForm.querySelector('[data-annotation-select]'); select.value = '';
  const note = annotationForm.querySelector('.annotation-note'); const apply = annotationForm.querySelector('.annotation-apply');
  note.value = ''; note.hidden = true; note.required = false; apply.hidden = true;
  annotationForm.querySelector('[data-roster-annotation-title]').textContent = `${button.dataset.staffName} · ${button.dataset.dayLabel}`;
  annotationDialog.showModal(); select.focus();
});
annotationDialog?.querySelectorAll('[data-roster-annotation-close]').forEach((button) => button.addEventListener('click', closeAnnotationEditor));
annotationDialog?.addEventListener('click', (event) => { if (event.target === annotationDialog) closeAnnotationEditor(); });
document.addEventListener('submit', (event) => { if (event.target.matches('[data-roster-annotation-form]') || event.target.closest('#roster')) rememberPosition(); });
document.querySelectorAll('[data-annotation-select]').forEach((select) => select.addEventListener('change', () => {
  const form = select.form; const needsNote = select.selectedOptions[0]?.dataset.noteRequired === 'yes';
  const note = form.querySelector('.annotation-note'); const apply = form.querySelector('.annotation-apply');
  note.hidden = !needsNote; note.required = needsNote; apply.hidden = !needsNote;
  if (needsNote) note.focus(); else form.requestSubmit();
}));
document.querySelectorAll('[data-annotation-dialog-open]').forEach((button) => button.addEventListener('click', () => {
  const dialog = document.getElementById(button.dataset.annotationDialogOpen); dialog?.showModal();
  const textarea = dialog?.querySelector('textarea'); textarea?.focus(); textarea?.setSelectionRange(textarea.value.length, textarea.value.length);
}));
document.querySelectorAll('.annotation-dialog').forEach((dialog) => {
  dialog.querySelectorAll('[data-annotation-dialog-close]').forEach((button) => button.addEventListener('click', () => dialog.close()));
  dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
  const textarea = dialog.querySelector('textarea'); const counter = dialog.querySelector('[data-annotation-character-count]');
  textarea?.addEventListener('input', () => { if (counter) counter.textContent = textarea.value.length; });
});
