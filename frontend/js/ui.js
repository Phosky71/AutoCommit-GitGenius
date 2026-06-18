import { invoke, repos, history, smartMode, stateControls } from './api.js';

let currentSection = 'repos';
export let editingRepoId = null;
export let dryRunPath = '';
export let approvalPath = '';
export let approvalUsedLlm = false;

/* ── SETTERS ──────────────────────────────────────────────── */
export function setEditingRepoId(id) { editingRepoId = id; }
export function setDryRunPath(path) { dryRunPath = path; }

/* ── THEME ────────────────────────────────────────────────── */
export function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeIcon(theme);
}

export async function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try {
        const cfg = await invoke('load_config_from_file');
        if (cfg) {
            cfg.theme = next;
            await invoke('save_config', { config: cfg });
        }
    } catch (e) { /* silencioso */ }
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('theme-btn');
    if (!btn) return;
    btn.innerHTML = theme === 'dark'
        ? '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
        : '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
}

/* ── NAVIGATION ───────────────────────────────────────────── */
export function navigateTo(section) {
    currentSection = section;
    document.querySelectorAll('.nav-item').forEach(i =>
        i.classList.toggle('active', i.dataset.section === section));
    document.querySelectorAll('.section-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${section}`).classList.add('active');

    const titles = { repos: 'Repositories', settings: 'Settings', history: 'Commit History' };
    document.getElementById('topbar-title').textContent = titles[section] || section;

    // Disparamos evento para que main.js recargue datos si hace falta
    window.dispatchEvent(new CustomEvent('navigated', { detail: { section } }));
}

/* ── TOAST ────────────────────────────────────────────────── */
export function toast(msg, type = 'info', duration = 3500) {
    const tc = document.getElementById('toast-container');
    const t  = document.createElement('div');
    t.className = `toast toast-${type}`;
    const icons = {
        success: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>',
        error:   '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        info:    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    };
    t.innerHTML = `${icons[type] || icons.info} <span>${msg}</span>`;
    tc.appendChild(t);
    const remove = () => {
        t.classList.add('removing');
        t.addEventListener('animationend', () => t.remove(), { once: true });
    };
    const timer = setTimeout(remove, duration);
    t.addEventListener('click', () => { clearTimeout(timer); remove(); });
}

/* ── UTILS ────────────────────────────────────────────────── */
export function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function formatRelTime(unixSecs) {
    const diff = Math.floor(Date.now() / 1000 - unixSecs);
    if (diff < 60)    return `${diff}s ago`;
    if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

export function downloadText(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

/* ── MODALS (REPO) ────────────────────────────────────────── */
export function openAddRepoModal() {
    editingRepoId = null;
    document.getElementById('modal-repo-title').textContent = 'Add repository';
    document.getElementById('btn-save-repo').innerHTML =
        `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg> Add repository`;

    document.getElementById('repo-edit-id').value   = '';
    document.getElementById('repo-path').value      = '';
    document.getElementById('repo-path-hint').textContent = '';
    document.getElementById('repo-path-hint').className   = 'form-hint';
    document.getElementById('repo-interval').value  = document.getElementById('cfg-interval')?.value || 30;
    document.getElementById('repo-cooldown').value  = 5;
    document.getElementById('repo-prefix').value    = '';
    document.getElementById('repo-push-enabled').checked = true;
    document.getElementById('repo-push-remote').value    = 'origin';
    document.getElementById('repo-push-branch').value    = 'main';

    document.getElementById('repo-preview').style.display = 'none';
    document.getElementById('push-branch-group').style.display = '';
    document.getElementById('modal-repo').classList.add('open');
}

export function openEditRepoModal(id, loadBranchFn) {
    const repo = repos.find(r => r.id === id);
    if (!repo) return;
    editingRepoId = id;

    document.getElementById('modal-repo-title').textContent = 'Edit repository';
    document.getElementById('btn-save-repo').innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
      <polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>
    </svg> Save changes`;

    document.getElementById('repo-edit-id').value           = id;
    document.getElementById('repo-path').value              = repo.path;
    document.getElementById('repo-interval').value          = repo.interval_minutes;
    document.getElementById('repo-cooldown').value          = repo.cooldown_minutes;
    document.getElementById('repo-prefix').value            = repo.commit_prefix;
    document.getElementById('repo-push-enabled').checked    = repo.push_enabled;
    document.getElementById('repo-push-remote').value       = repo.push_remote || 'origin';
    document.getElementById('repo-push-branch').value       = repo.push_branch || 'main';

    document.getElementById('push-branch-group').style.display = repo.push_enabled ? '' : 'none';
    document.getElementById('repo-preview').style.display   = 'none';

    if(loadBranchFn) loadBranchFn(repo.path);
    document.getElementById('modal-repo').classList.add('open');
}

export function closeRepoModal() {
    document.getElementById('modal-repo').classList.remove('open');
}

export function togglePushBranch() {
    const enabled = document.getElementById('repo-push-enabled').checked;
    document.getElementById('push-branch-group').style.display = enabled ? '' : 'none';
}

/* ── DROPDOWNS ────────────────────────────────────────────── */
export function toggleDropdown(id) {
    const menu   = document.getElementById(`ddm-${id}`);
    const isOpen = menu.classList.contains('open');
    document.querySelectorAll('.dropdown-menu.open').forEach(m => m.classList.remove('open'));
    if (!isOpen) menu.classList.add('open');
}

export function closeDropdown(id) {
    document.getElementById(`ddm-${id}`)?.classList.remove('open');
}

/* ── MODALS (DRY RUN) ─────────────────────────────────────── */
export function openDryRunModal() {
    document.getElementById('dryrun-result').innerHTML = '<span class="spin"></span>';
    document.getElementById('dryrun-diff').style.display = 'none';
    document.getElementById('modal-dryrun').classList.add('open');
}

export function closeDryRunModal() {
    document.getElementById('modal-dryrun').classList.remove('open');
}

/* ── MODALS (APPROVAL / HITL) ─────────────────────────────── */
export function openApprovalModal(path, pending) {
    approvalPath = path;
    approvalUsedLlm = pending.used_llm;

    document.getElementById('approval-message').value        = pending.message || '';
    document.getElementById('approval-push-enabled').checked = true;
    document.getElementById('approval-tag').value            = '';

    const s = pending.diff_stats;
    if (s) {
        document.getElementById('approval-diff-info').innerHTML =
            `<span class="diff-files">${s.files_changed} file${s.files_changed !== 1 ? 's' : ''}</span>
       <span class="diff-add">+${s.insertions}</span>
       <span class="diff-del">-${s.deletions}</span>
       <span class="diff-files" style="color:var(--color-text-faint)">${s.estimated_tokens} est. tokens</span>`;
    } else {
        document.getElementById('approval-diff-info').textContent = '—';
    }

    const fileList = document.getElementById('approval-files-list');
    if (pending.files_changed_list && pending.files_changed_list.length) {
        fileList.innerHTML = pending.files_changed_list
            .map(f => `<div class="approval-file-item">${escHtml(f)}</div>`).join('');
        document.getElementById('approval-files-section').style.display = '';
    } else {
        document.getElementById('approval-files-section').style.display = 'none';
    }

    if (pending.diff_preview) {
        document.getElementById('approval-diff-preview').textContent = pending.diff_preview;
        document.getElementById('approval-diff-section').style.display = '';
    } else {
        document.getElementById('approval-diff-section').style.display = 'none';
    }

    document.getElementById('modal-approval').classList.add('open');
}

export function closeApprovalModal() {
    document.getElementById('modal-approval').classList.remove('open');
    approvalPath = '';
}

/* ── RENDERERS ────────────────────────────────────────────── */
export function updateSmartModeUI(val) {
    document.querySelectorAll('.smart-mode-ctrl .segment-btn').forEach(btn =>
        btn.classList.toggle('active', btn.dataset.val === val));

    const hints = {
        always: 'LLM is always used to generate commit messages',
        smart:  'LLM only when diff is significant (recommended)',
        never:  'Always use heuristic messages — no LLM calls',
    };
    document.getElementById('smart-mode-hint').textContent = hints[val] || '';
    document.getElementById('threshold-group').style.display = val === 'smart' ? '' : 'none';
}

export function renderRepos(loadBranchForRepoFn) {
    const grid = document.getElementById('repos-grid');
    const sub  = document.getElementById('repos-count-sub');
    if (!grid) return;

    let empty = document.getElementById('repos-empty');
    if (!empty) {
        empty = document.createElement('div');
        empty.id = 'repos-empty';
        empty.className = 'empty-state';
        empty.innerHTML = `
      <div class="empty-state-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M3 3h18v18H3z"/><path d="M3 9h18"/><path d="M9 21V9"/>
        </svg>
      </div>
      <h3>No repositories yet</h3>
      <p>Add a Git repository to start automating commits with AI-generated messages.</p>
      <button id="btn-add-repo-empty" class="btn btn-primary">Add repository</button>
    `;
    }

    if (empty.parentNode === grid) grid.removeChild(empty);

    if (!repos.length) {
        grid.innerHTML = '';
        grid.appendChild(empty);
        empty.style.display = '';
        if (sub) sub.textContent = 'No repositories configured';
        // El event listener se delega en main.js
        return;
    }

    empty.style.display = 'none';
    if (sub) sub.textContent = `${repos.length} repositor${repos.length === 1 ? 'y' : 'ies'} configured`;

    grid.innerHTML = repos.map(repo => {
        const parts    = repo.path.replace(/\\/g, '/').split('/');
        const repoName = parts.pop();
        const dirPath  = parts.join('/');
        const pushLabel = `${repo.push_remote || 'origin'}/${repo.push_branch || 'main'}`;
        return `
    <div class="repo-card" id="repo-card-${repo.id}">
      <div class="repo-card-header">
        <div class="repo-path">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;color:var(--color-text-faint)"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
          <div><span class="repo-path-dir">${dirPath}/</span><span class="repo-path-name">${repoName}</span></div>
        </div>
        <div class="repo-actions">
          <label class="repo-toggle" title="${repo.enabled ? 'Enabled' : 'Paused'}">
            <input type="checkbox" class="repo-enabled-toggle" data-id="${repo.id}" ${repo.enabled ? 'checked' : ''}>
            <span class="repo-toggle-slider"></span>
          </label>
          <div class="dropdown" id="dd-${repo.id}">
            <button class="btn btn-ghost btn-icon repo-dropdown-btn" data-id="${repo.id}" title="More options">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1" fill="currentColor"/><circle cx="12" cy="12" r="1" fill="currentColor"/><circle cx="12" cy="19" r="1" fill="currentColor"/></svg>
            </button>
            <div class="dropdown-menu" id="ddm-${repo.id}">
              <div class="dropdown-item repo-edit-btn" data-id="${repo.id}">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg> Edit
              </div>
              <div class="dropdown-divider"></div>
              <div class="dropdown-item danger repo-delete-btn" data-id="${repo.id}">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg> Remove
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="repo-meta">
        <div class="repo-meta-item" id="branch-${repo.id}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg><span>—</span>
        </div>
        <div class="repo-meta-item">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> Every ${repo.interval_minutes}m
        </div>
        ${repo.push_enabled ? `<div class="repo-meta-item"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>${pushLabel}</div>` : ''}
        ${repo.last_commit_time ? `<div class="repo-meta-item"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>Last ${formatRelTime(repo.last_commit_time)}</div>` : ''}
        <span class="badge ${repo.enabled ? 'badge-running' : 'badge-paused'}">${repo.enabled ? '&nbsp;Running&nbsp;' : 'Paused'}</span>
      </div>
      <div class="repo-card-actions">
        <button class="btn btn-primary btn-sm repo-commit-btn" data-id="${repo.id}" data-path="${repo.path}" id="commit-btn-${repo.id}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Commit now
        </button>
        <button class="btn btn-secondary btn-sm repo-dryrun-btn" data-path="${repo.path}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg> Dry run
        </button>
        <button class="btn btn-secondary btn-sm repo-diff-btn" data-path="${repo.path}" data-target="diff-${repo.id}" title="Check diff">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg> Diff
        </button>
        <div class="diff-info" id="diff-${repo.id}"></div>
      </div>
    </div>`;
    }).join('');

    repos.forEach(r => loadBranchForRepoFn(r));
}

export function updateHistoryStats() {
    const ai     = history.filter(h => h.used_llm).length;
    const tokens = history.reduce((s, h) => s + (h.estimated_tokens || 0), 0);
    animateNum('stat-total',     history.length);
    animateNum('stat-ai',        ai);
    animateNum('stat-heuristic', history.length - ai);
    animateNum('stat-tokens',    tokens);
}

function animateNum(id, target) {
    const el = document.getElementById(id);
    if (!el) return;
    const start = parseInt(el.textContent) || 0;
    const diff  = target - start;
    if (diff === 0) return;
    const steps = 20, stepMs = 15;
    let i = 0;
    const t = setInterval(() => {
        i++;
        el.textContent = Math.round(start + diff * i / steps);
        if (i >= steps) clearInterval(t);
    }, stepMs);
}

export function updateHistoryRepoFilter() {
    const sel   = document.getElementById('history-filter-repo');
    const cur   = sel.value;
    const paths = [...new Set(history.map(h => h.repo_path))];
    sel.innerHTML = '<option value="">All repositories</option>'
        + paths.map(p => {
            const name = p.replace(/\\/g, '/').split('/').pop();
            return `<option value="${escHtml(p)}"${cur === p ? ' selected' : ''}>${escHtml(name)}</option>`;
        }).join('');
}

export function renderHistory() {
    const filterRepo = document.getElementById('history-filter-repo').value;
    const filterType = document.getElementById('history-filter-type').value;
    const list       = document.getElementById('history-list');
    const empty      = document.getElementById('history-empty');
    let filtered = history;

    if (filterRepo) filtered = filtered.filter(h => h.repo_path === filterRepo);
    if (filterType === 'ai')        filtered = filtered.filter(h =>  h.used_llm);
    if (filterType === 'heuristic') filtered = filtered.filter(h => !h.used_llm);

    if (!filtered.length) {
        list.innerHTML = '';
        list.appendChild(empty);
        empty.style.display = '';
        return;
    }
    empty.style.display = 'none';
    list.innerHTML = filtered.map(entry => {
        const repoName = entry.repo_path.replace(/\\/g, '/').split('/').pop();
        const date     = new Date((entry.timestamp || 0) * 1000).toLocaleString();
        return `
    <div class="history-item">
      <div class="history-item-header">
        <span class="history-msg">${escHtml(entry.message)}</span>
        <span class="badge ${entry.used_llm ? 'badge-ai' : 'badge-heuristic'}">${entry.used_llm ? 'AI' : 'Heuristic'}</span>
      </div>
      <div class="history-meta">
        <div class="history-meta-item"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>${escHtml(repoName)}</div>
        <div class="history-meta-item"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>${date}</div>
        <div class="history-meta-item diff-files">${entry.files_changed} file${entry.files_changed !== 1 ? 's' : ''}</div>
        <span class="diff-add">+${entry.insertions}</span>
        <span class="diff-del">-${entry.deletions}</span>
      </div>
    </div>`;
    }).join('');
}