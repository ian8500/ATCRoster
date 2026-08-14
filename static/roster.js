function printRoster() { window.print(); }

const rosterRoot = document.getElementById('roster');
const rosterTable = document.querySelector('table.roster');
const siteHeader = document.querySelector('.site-header');
const rosterStickyShield = document.querySelector('[data-roster-sticky-shield]');
const assignmentUrlFor = (control) => {
  const cell = control?.closest('.cell');
  const template = rosterRoot?.dataset.assignmentUrlTemplate;
  if (!cell || !template) return '';
  return template
    .replace('__staff__', encodeURIComponent(cell.dataset.rosterStaff || ''))
    .replace('__day__', encodeURIComponent(cell.dataset.rosterDay || ''));
};
const cellLabel = (control) => {
  const cell = control?.closest('.cell');
  return {
    name: cell?.dataset.rosterName || 'Controller',
    day: cell?.dataset.rosterDay || 'selected day',
  };
};
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

const applyShiftPayload = (select, payload, baseline = false) => {
  const cell = select.closest('.cell');
  const code = payload.code || '';
  const prefix = code === 'EM' ? 'm' : (code === 'LA' ? 'a' : code.slice(0, 1).toLowerCase());
  select.value = code;
  select.dataset.code = code;
  select.dataset.version = payload.version;
  select.dataset.override = baseline ? '0' : select.dataset.override;
  select.className = `code-input roster-cell-select code-len-${code.length}${code ? ` ${code.toLowerCase()} group-${prefix}` : ''}`;
  const label = cellLabel(select);
  select.setAttribute('aria-label', `${label.name} shift on ${label.day}: ${code || 'unassigned'}`);
  cell?.querySelectorAll('[data-roster-annotation-open]').forEach((annotationButton) => { annotationButton.dataset.version = payload.version; });
  cell?.classList.toggle('training', Boolean(payload.is_training)); cell?.classList.remove('request-applied'); cell?.querySelector('.request-applied-marker')?.remove();
  if (cell) cell.dataset.rosterCode = code;
  updateDaySummary(payload.day, payload.day_summary);
};
rosterRoot?.addEventListener('change', async (event) => {
  const select = event.target.closest('[data-roster-shift-select]');
  if (!select || select.dataset.saving === '1') return;
  const previousCode = select.dataset.code || '';
  const code = select.value;
  const baseline = code === '__BASELINE__';
  const data = new FormData();
  data.set('_csrf_token', rosterRoot?.dataset.csrfToken || '');
  data.set('assignment_version', select.dataset.version || '0');
  data.set('code', code);
  select.dataset.saving = '1';
  select.disabled = true;
  showStatus('Saving…');
  try {
    const response = await fetch(assignmentUrlFor(select), { method: 'POST', body: data, credentials: 'same-origin', headers: { Accept: 'application/json', 'X-Requested-With': 'XMLHttpRequest' } });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) throw new Error(payload.error || 'The roster change could not be saved.');
    applyShiftPayload(select, payload, baseline);
    showStatus('Saved');
  } catch (error) {
    select.value = previousCode;
    showStatus(error.message || 'The roster change could not be saved.', true);
  }
  finally {
    select.disabled = false;
    select.dataset.saving = '0';
  }
});

const inspector = document.querySelector('[data-roster-inspector]');
const inspectorTitle = document.querySelector('[data-roster-inspector-title]');
const inspectorDetail = document.querySelector('[data-roster-inspector-detail]');
let selectedRosterCell;
const editableCell = (cell) => cell?.querySelector('[data-roster-shift-select]:not(:disabled)');
const selectRosterCell = (cell) => {
  if (!editableCell(cell)) return;
  document.querySelectorAll('.cell.is-selected').forEach((item) => item.classList.remove('is-selected'));
  selectedRosterCell = cell; cell.classList.add('is-selected');
  if (inspector) { inspector.hidden = false; inspectorTitle.textContent = `${cell.dataset.rosterName} · ${cell.dataset.rosterDay}`; inspectorDetail.textContent = `Shift ${cell.dataset.rosterCode || 'not assigned'}. Use the in-cell shift selector to change it.`; }
};
rosterRoot?.addEventListener('click', (event) => { const cell = event.target.closest('.cell'); if (cell && editableCell(cell) && !event.target.closest('[data-roster-shift-select]')) selectRosterCell(cell); });
document.querySelector('[data-roster-inspector-close]')?.addEventListener('click', () => { if (inspector) inspector.hidden = true; selectedRosterCell?.classList.remove('is-selected'); });

document.addEventListener('keydown', (event) => {
  if (!selectedRosterCell || event.target.matches('input,select,textarea,button')) return;
  const row = selectedRosterCell.closest('tr'); const offset = selectedRosterCell.cellIndex; let target;
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { const step = event.key === 'ArrowRight' ? 1 : -1; for (let index = offset + step; row?.cells[index]; index += step) { if (editableCell(row.cells[index])) { target = row.cells[index]; break; } } }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { const step = event.key === 'ArrowDown' ? 1 : -1; let next = row; while ((next = step > 0 ? next?.nextElementSibling : next?.previousElementSibling)) { const candidate = next.cells?.[offset]; if (editableCell(candidate)) { target = candidate; break; } } }
  if (target) { event.preventDefault(); selectRosterCell(target); editableCell(target)?.focus({ preventScroll: true }); }
});

const annotationDialog = document.getElementById('roster-annotation-dialog');
const annotationForm = annotationDialog?.querySelector('[data-roster-annotation-form]');
let activeAnnotation;
const closeAnnotationEditor = () => { if (annotationDialog?.open) annotationDialog.close(); activeAnnotation?.focus({ preventScroll: true }); };
rosterRoot?.addEventListener('click', (event) => {
  const button = event.target.closest('[data-roster-annotation-open]');
  if (!button || !annotationForm) return;
  activeAnnotation = button; annotationForm.action = assignmentUrlFor(button);
  annotationForm.querySelector('[data-roster-annotation-version]').value = button.dataset.version || '0';
  const select = annotationForm.querySelector('[data-annotation-select]'); select.value = '';
  const note = annotationForm.querySelector('.annotation-note'); const apply = annotationForm.querySelector('.annotation-apply');
  note.value = ''; note.hidden = true; note.required = false; apply.hidden = true;
  { const label = cellLabel(button); annotationForm.querySelector('[data-roster-annotation-title]').textContent = `${label.name} · ${label.day}`; }
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
