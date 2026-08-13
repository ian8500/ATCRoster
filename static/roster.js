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
  const cell = activeShift.closest('.cell');
  const data = new FormData(shiftForm);
  const baseline = data.get('code') === '__BASELINE__';
  shiftForm.dataset.saving = '1'; shiftForm.classList.add('is-saving'); shiftSelect.disabled = true; showStatus('Saving…');
  try {
    const response = await fetch(shiftForm.action, { method: 'POST', body: data, credentials: 'same-origin', headers: { Accept: 'application/json', 'X-Requested-With': 'XMLHttpRequest' } });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) throw new Error(payload.error || 'The roster change could not be saved.');
    const code = payload.code || ''; const prefix = code === 'EM' ? 'm' : (code === 'LA' ? 'a' : code.slice(0, 1).toLowerCase());
    activeShift.textContent = code || '—'; activeShift.dataset.code = code; activeShift.dataset.version = payload.version;
    activeShift.dataset.override = baseline ? '0' : activeShift.dataset.override;
    activeShift.className = `code-input roster-cell-button code-len-${code.length}${code ? ` ${code.toLowerCase()} group-${prefix}` : ''}`;
    activeShift.setAttribute('aria-label', `${activeShift.dataset.staffName} shift on ${activeShift.dataset.dayLabel}: ${code || 'unassigned'}`);
    cell?.querySelectorAll('[data-roster-annotation-open]').forEach((button) => { button.dataset.version = payload.version; });
    cell?.classList.toggle('training', Boolean(payload.is_training)); cell?.classList.remove('request-applied'); cell?.querySelector('.request-applied-marker')?.remove();
    updateDaySummary(payload.day, payload.day_summary); showStatus('Saved'); closeShiftEditor();
  } catch (error) { showStatus(error.message || 'The roster change could not be saved.', true); }
  finally {
    shiftSelect.disabled = false;
    // The generic form-progress handler disables submit controls. Restore this
    // async editor's control after every response so a retry/edit is possible.
    if (shiftSubmit) { shiftSubmit.disabled = false; shiftSubmit.removeAttribute('aria-disabled'); }
    shiftForm.dataset.saving = '0'; shiftForm.classList.remove('is-saving');
  }
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
