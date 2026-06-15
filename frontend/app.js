/* ── TAURI BRIDGE ─────────────────────────────────────────── */
const invoke = (cmd, args) => window.__TAURI__?.core?.invoke(cmd, args) ?? Promise.resolve(null);
const listen  = (ev, cb)   => window.__TAURI__?.event?.listen(ev, cb)   ?? (() => {});

/* ── STATE ────────────────────────────────────────────────── */
let repos          = [];
let history        = [];
let timerRunning   = false;
let currentSection = 'repos';
let editingRepoId  = null;
let smartMode      = 'smart';
let humanInTheLoop = true;   // ← NUEVO

/* ── THEME ────────────────────────────────────────────────── */
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  updateThemeIcon(theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  saveThemePreference(next);
}

async function saveThemePreference(theme) {
  try {
    const cfg = await invoke('load_config_from_file');
    if (cfg) {
      cfg.theme = theme;
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
function navigateTo(section) {
    currentSection = section;
    document.querySelectorAll('.nav-item').forEach(i =>
        i.classList.toggle('active', i.dataset.section === section));
    document.querySelectorAll('.section-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${section}`).classList.add('active');
    const titles = { repos: 'Repositories', settings: 'Settings', history: 'Commit History' };
    document.getElementById('topbar-title').textContent = titles[section] || section;
    if (section === 'history') loadHistory();
}

/* ── TOAST ────────────────────────────────────────────────── */
function toast(msg, type = 'info', duration = 3500) {
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

/* ── TIMER ────────────────────────────────────────────────── */
async function toggleTimer() {
    if (timerRunning) {
        try {
            await invoke('stop_auto_commit');
            setTimerState(false);
            toast('Timer stopped', 'info');
        } catch (e) { toast('Failed to stop timer: ' + e, 'error'); }
    } else {
        try {
            await invoke('start_auto_commit');
            setTimerState(true);
            toast('Timer started', 'success');
        } catch (e) { toast('Failed to start timer: ' + e, 'error'); }
    }
}
function setTimerState(running) {
    timerRunning = running;
    // document.getElementById('timer-dot').classList.toggle('running', running);
    // document.getElementById('timer-status-text').textContent = running ? 'Running' : 'Stopped';
    // document.getElementById('btn-start-stop').innerHTML = running
    //     ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop'
    //     : '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start';
}

/* ── BACKEND EVENTS ───────────────────────────────────────── */
listen('commit-status', ev => {
    const r = ev.payload;
    // ← CORREGIDO: si hay pending_approval, abrir modal HiTL
    if (r.pending_approval) {
        const repoPath = repos.find(rp => rp.enabled)?.path ?? '';
        openApprovalModal(repoPath, r.pending_approval);
        return;
    }
    const tag = r.used_llm ? '🤖 ' : '⚙️ ';
    toast(`${tag}${r.message}`, 'success', 5000);
    loadRepos();
    if (currentSection === 'history') loadHistory();
});
listen('commit-error', ev => {
    toast('Commit error: ' + ev.payload, 'error', 6000);
});

/* ── REPOS ────────────────────────────────────────────────── */
async function loadRepos() {
    try {
        repos = await invoke('get_repos') ?? [];
        renderRepos();
    } catch (e) { console.error(e); }
}

function renderRepos() {
    const grid = document.getElementById('repos-grid');
    const sub  = document.getElementById('repos-count-sub');
    if (!grid) return;

    // Buscar o crear el empty-state
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

    // Extraerlo del DOM antes de limpiar
    if (empty.parentNode === grid) grid.removeChild(empty);

    if (!repos.length) {
        grid.innerHTML = '';
        grid.appendChild(empty);
        empty.style.display = '';
        if (sub) sub.textContent = 'No repositories configured';
        // Re-enlazar el botón del empty state
        document.getElementById('btn-add-repo-empty')
            ?.addEventListener('click', openAddRepoModal);
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
    repos.forEach(r => loadBranchForRepo(r));
}

/* ── EVENT DELEGATION ─────────────────────────────────────── */
function setupRepoDelegation() {
    const grid = document.getElementById('repos-grid');
    grid.addEventListener('change', e => {
        const toggle = e.target.closest('.repo-enabled-toggle');
        if (toggle) toggleRepo(toggle.dataset.id, toggle.checked);
    });
    grid.addEventListener('click', e => {
        const ddBtn     = e.target.closest('.repo-dropdown-btn');
        const editBtn   = e.target.closest('.repo-edit-btn');
        const delBtn    = e.target.closest('.repo-delete-btn');
        const commitBtn = e.target.closest('.repo-commit-btn');
        const dryBtn    = e.target.closest('.repo-dryrun-btn');
        const diffBtn   = e.target.closest('.repo-diff-btn');
        if (ddBtn)     { e.stopPropagation(); toggleDropdown(ddBtn.dataset.id); return; }
        if (editBtn)   { openEditRepoModal(editBtn.dataset.id); closeDropdown(editBtn.dataset.id); return; }
        if (delBtn)    { deleteRepo(delBtn.dataset.id); closeDropdown(delBtn.dataset.id); return; }
        if (commitBtn) { commitNow(commitBtn.dataset.id, commitBtn.dataset.path); return; }
        if (dryBtn)    { openDryRun(dryBtn.dataset.path); return; }
        if (diffBtn)   { previewDiff(diffBtn.dataset.path, diffBtn.dataset.target);  }
    });
}

async function loadBranchForRepo(repo) {
    try {
        const branch = await invoke('get_current_branch', { path: repo.path });
        const el = document.querySelector(`#branch-${repo.id} span`);
        if (el) el.textContent = branch;
    } catch (e) {}
}

async function previewDiff(path, targetId) {
    try {
        const stats = await invoke('get_diff_preview', { path });
        const el = document.getElementById(targetId);
        if (!el) return;
        if (!stats || stats.files_changed === 0) {
            el.innerHTML = `<span style="color:var(--color-text-faint);font-size:var(--text-xs)">No changes</span>`;
            return;
        }
        el.innerHTML = `<span class="diff-files">${stats.files_changed} file${stats.files_changed !== 1 ? 's' : ''}</span>
      <span class="diff-add">+${stats.insertions}</span>
      <span class="diff-del">-${stats.deletions}</span>
      <span class="diff-files" style="color:var(--color-text-faint)">${stats.estimated_tokens} est. tokens</span>`;
    } catch (e) {}
}

async function toggleRepo(id, enabled) {
    const repo = repos.find(r => r.id === id);
    if (!repo) return;
    repo.enabled = enabled;
    try {
        await invoke('update_repo', { repo });
        toast(`Repository ${enabled ? 'enabled' : 'paused'}`, 'info');
        loadRepos();
    } catch (e) { toast('Failed to update: ' + e, 'error'); }
}

async function deleteRepo(id) {
    if (!confirm('Remove this repository from AutoCommit? Git history is not affected.')) return;
    try {
        await invoke('remove_repo', { id });
        toast('Repository removed', 'info');
        loadRepos();
    } catch (e) { toast('Failed to remove: ' + e, 'error'); }
}

/* ── COMMIT NOW ───────────────────────────────────────────── */
async function commitNow(id, path) {
    const btn = document.getElementById(`commit-btn-${id}`);
    if (!btn) return;
    btn.disabled = true;
    btn.innerHTML = `<span class="spin"></span> Committing`;
    try {
        const result = await invoke('run_commit', { path });
        if (result.message === 'No changes to commit') {
            toast('No changes to commit', 'info');
        } else if (result.message.startsWith('Cooldown')) {
            toast(result.message, 'info');
        } else if (result.pending_approval) {
            // ← CORREGIDO: Human in the Loop
            openApprovalModal(path, result.pending_approval);
        } else {
            toast(`Committed: ${result.message}`, 'success', 5000);
            loadRepos();
        }
    } catch (e) { toast(`Commit failed: ${e}`, 'error'); }
    finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Commit now`;
    }
}

/* ── DROPDOWN ─────────────────────────────────────────────── */
function toggleDropdown(id) {
    const menu   = document.getElementById(`ddm-${id}`);
    const isOpen = menu.classList.contains('open');
    document.querySelectorAll('.dropdown-menu.open').forEach(m => m.classList.remove('open'));
    if (!isOpen) menu.classList.add('open');
}
function closeDropdown(id) {
    document.getElementById(`ddm-${id}`)?.classList.remove('open');
}
document.addEventListener('click', e => {
    if (!e.target.closest('.dropdown'))
        document.querySelectorAll('.dropdown-menu.open').forEach(m => m.classList.remove('open'));
});

/* ── ADD / EDIT REPO MODAL ────────────────────────────────── */
function openAddRepoModal() {
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

function openEditRepoModal(id) {
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
    document.getElementById('repo-push-enabled').checked   = repo.push_enabled;
    document.getElementById('repo-push-remote').value       = repo.push_remote || 'origin';
    document.getElementById('repo-push-branch').value       = repo.push_branch || 'main';
    document.getElementById('push-branch-group').style.display = repo.push_enabled ? '' : 'none';
    document.getElementById('repo-preview').style.display   = 'none';
    loadBranchOptions(repo.path);
    document.getElementById('modal-repo').classList.add('open');
}

function closeRepoModal() {
    document.getElementById('modal-repo').classList.remove('open');
}

async function pickDirectory() {
    try {
        const path = await invoke('select_directory');
        if (!path) return;
        document.getElementById('repo-path').value = path;
        validateRepoPath(path);
        previewRepoDiff(path);
    } catch (e) { toast('Failed to pick directory: ' + e, 'error'); }
}

async function validateRepoPath(path) {
    const hint = document.getElementById('repo-path-hint');
    try {
        const valid = await invoke('validate_repo_path', { path });
        hint.textContent = valid ? 'Valid Git repository' : 'No .git folder found';
        hint.className   = 'form-hint ' + (valid ? 'success' : 'error');
    } catch (e) { hint.textContent = ''; hint.className = 'form-hint'; }
}

async function loadBranchOptions(path) {
    const hint = document.getElementById('push-branch-hint');
    try {
        const branches = await invoke('list_remote_branches', { path });
        if (branches && branches.length)
            hint.textContent = 'Remote branches: ' + branches.slice(0, 5).join(', ') + (branches.length > 5 ? '…' : '');
        else hint.textContent = '';
    } catch (e) { hint.textContent = ''; }
}

async function previewRepoDiff(path) {
    const preview = document.getElementById('repo-preview');
    const box     = document.getElementById('repo-diff-preview');
    preview.style.display = '';
    box.textContent = 'Checking diff…';
    try {
        const stats = await invoke('get_diff_preview', { path });
        if (!stats || stats.files_changed === 0) {
            box.textContent = 'No uncommitted changes';
        } else {
            box.innerHTML = `${stats.files_changed} file${stats.files_changed !== 1 ? 's' : ''} changed &nbsp;`
                + `<span class="diff-add">+${stats.insertions}</span> `
                + `<span class="diff-del">-${stats.deletions}</span> &nbsp;${stats.estimated_tokens} tokens`;
        }
    } catch (e) { box.textContent = 'Could not read diff'; }
}

function togglePushBranch() {
    const enabled = document.getElementById('repo-push-enabled').checked;
    document.getElementById('push-branch-group').style.display = enabled ? '' : 'none';
}

async function saveRepo() {
    const path = document.getElementById('repo-path').value.trim();
    if (!path) { toast('Please select a repository path', 'error'); return; }

    const interval    = parseInt(document.getElementById('repo-interval').value) || 30;
    const cooldown    = parseInt(document.getElementById('repo-cooldown').value) || 0;
    const prefix      = document.getElementById('repo-prefix').value.trim();
    const push_enabled = document.getElementById('repo-push-enabled').checked;
    const pushRemote  = document.getElementById('repo-push-remote').value.trim() || 'origin';
    const pushBranch  = document.getElementById('repo-push-branch').value.trim() || 'main';

    const repoObj = {
        id:               editingRepoId || crypto.randomUUID(),
        path,
        interval_minutes: interval,
        timer_enabled:    false,
        enabled:          true,
        push_enabled:     push_enabled,
        push_remote:      pushRemote,
        push_branch:      pushBranch,
        commit_prefix:    prefix,
        last_commit_time: 0,
        cooldown_minutes: cooldown,
    };

    try {
        if (editingRepoId) {
            repoObj.enabled          = repos.find(r => r.id === editingRepoId)?.enabled ?? true;
            repoObj.last_commit_time = repos.find(r => r.id === editingRepoId)?.last_commit_time ?? 0;
            await invoke('update_repo', { repo: repoObj });
            toast('Repository updated', 'success');
        } else {
            await invoke('add_repo', { repo: repoObj });
            toast('Repository added', 'success');
        }
        closeRepoModal();
        loadRepos();
    } catch (e) { toast('Failed to save: ' + e, 'error'); }
}
/* ── DRY RUN MODAL ────────────────────────────────────────── */
let dryRunPath = '';
function openDryRun(path) {
    dryRunPath = path;
    document.getElementById('dryrun-result').innerHTML = '<span class="spin"></span>';
    document.getElementById('dryrun-diff').style.display = 'none';
    document.getElementById('modal-dryrun').classList.add('open');
    runDryRun(path);
}
async function runDryRun(path) {
    try {
        const result = await invoke('dry_run_commit', { path });
        document.getElementById('dryrun-result').textContent = result.message;
        if (result.diff_stats) {
            const s = result.diff_stats;
            document.getElementById('dryrun-diff').style.display = '';
            document.getElementById('dryrun-diff-info').innerHTML =
                `<span class="diff-files">${s.files_changed} file${s.files_changed !== 1 ? 's' : ''}</span>
         <span class="diff-add">+${s.insertions}</span>
         <span class="diff-del">-${s.deletions}</span>
         <span class="diff-files" style="color:var(--color-text-faint)">${s.estimated_tokens} est. tokens</span>`;
        }
        const commitBtn = document.getElementById('btn-commit-from-dry');
        commitBtn.removeEventListener('click', commitBtn._handler);
        commitBtn._handler = () => { closeDryRunModal(); commitNowByPath(path); };
        commitBtn.addEventListener('click', commitBtn._handler);
    } catch (e) { document.getElementById('dryrun-result').textContent = 'Error: ' + e; }
}
function closeDryRunModal() {
    document.getElementById('modal-dryrun').classList.remove('open');
}
async function commitNowByPath(path) {
    try {
        const result = await invoke('run_commit', { path });
        if (result.message === 'No changes to commit') {
            toast('No changes to commit', 'info');
        } else if (result.message.startsWith('Cooldown')) {
            toast(result.message, 'info');
        } else if (result.pending_approval) {
            // ← CORREGIDO: Human in the Loop
            openApprovalModal(path, result.pending_approval);
        } else {
            toast(`Committed: ${result.message}`, 'success', 5000);
            loadRepos();
        }
    } catch (e) { toast(`Commit failed: ${e}`, 'error'); }
}

/* ── SETTINGS ─────────────────────────────────────────────── */
async function loadSettings() {
    try {
        const cfg = await invoke('load_config_from_file');
        if (!cfg) return;

        const sel = document.getElementById('cfg-provider');
        if (sel) sel.value = cfg.provider || 'lmstudio';

        document.getElementById('cfg-base-url').value         = cfg.llm_base_url    || '';
        document.getElementById('cfg-model').value            = cfg.llm_model_name  || '';
        document.getElementById('cfg-threshold').value        = cfg.smart_threshold_lines || 10;

        // Human in the loop
        humanInTheLoop = cfg.human_in_the_loop !== undefined ? cfg.human_in_the_loop : true;
        const hitlEl = document.getElementById('cfg-human-in-the-loop');
        if (hitlEl) hitlEl.checked = humanInTheLoop;

        const maskedKey = await invoke('get_masked_api_key');
        if (maskedKey && maskedKey !== 'sk-...') {
            document.getElementById('cfg-api-key').placeholder = maskedKey;
        }

        setSmartMode(cfg.smart_mode || 'smart');

        // ELIMINADO: La lógica de auto_start ya no se usa porque arranca por defecto.
    } catch (e) {
        console.error('loadSettings', e);
    }
}

async function onProviderChange() {
    const provider = document.getElementById('cfg-provider').value;
    try {
        const [baseUrl, model] = await invoke('get_provider_defaults', { provider });
        document.getElementById('cfg-base-url').value = baseUrl;
        document.getElementById('cfg-model').value    = model;
    } catch (e) {}
}

function setSmartMode(val) {
    smartMode = val;
    // ANTES: document.querySelectorAll('#smart-mode-ctrl .segment-btn').forEach(...)
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
function toggleApiKeyVisibility() {
    const input = document.getElementById('cfg-api-key');
    input.type  = input.type === 'password' ? 'text' : 'password';
}

async function testConnection() {
    const provider = document.getElementById('cfg-provider').value;
    const baseUrl  = document.getElementById('cfg-base-url').value.trim();
    const model    = document.getElementById('cfg-model').value.trim();
    const apiKey   = document.getElementById('cfg-api-key').value.trim();
    const result   = document.getElementById('test-result');
    const btn      = document.getElementById('btn-test');
    btn.disabled   = true;
    btn.innerHTML  = '<span class="spin"></span>';
    result.style.display = 'none';
    try {
        await invoke('test_connection', { provider, baseUrl: baseUrl, model, apiKey: apiKey });
        result.style.display = '';
        result.innerHTML = '<span class="test-status ok">Connection successful</span>';
    } catch (e) {
        result.style.display = '';
        result.innerHTML = `<span class="test-status fail">${e}</span>`;
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Test';
    }
}


async function saveSettings() {
    const btn = document.getElementById('btn-save-settings');
    btn.disabled = true;

    try {
        const cfg = await invoke('get_config');

        cfg.provider              = document.getElementById('cfg-provider').value;
        cfg.llm_base_url          = document.getElementById('cfg-base-url').value.trim();
        cfg.llm_model_name        = document.getElementById('cfg-model').value.trim();

        const apiKey = document.getElementById('cfg-api-key').value.trim();
        if (apiKey) {
            cfg.llm_api_key = apiKey;
        }

        cfg.smart_mode            = smartMode;
        cfg.smart_threshold_lines = parseInt(document.getElementById('cfg-threshold').value) || 10;

        // Solo leemos el Human in the loop (ya no existe el auto-start)
        cfg.human_in_the_loop     = document.getElementById('cfg-human-in-the-loop').checked;
        humanInTheLoop = cfg.human_in_the_loop;

        await invoke('save_config', { config: cfg });
        toast('Settings saved', 'success');

        if (apiKey) {
            document.getElementById('cfg-api-key').value       = '';
            document.getElementById('cfg-api-key').placeholder = 'sk-…';
        }
    } catch (e) {
        toast('Failed to save settings: ' + e, 'error');
    } finally {
        btn.disabled = false;
    }
}
// async function saveSettings() {
//     const provider  = document.getElementById('cfg-provider').value;
//     const baseUrl   = document.getElementById('cfg-base-url').value.trim();
//     const model     = document.getElementById('cfg-model').value.trim();
//     const apiKey    = document.getElementById('cfg-api-key').value.trim();
//     const threshold = parseInt(document.getElementById('cfg-threshold').value) || 10;
//     const interval  = 30;
//     const autoStart = document.getElementById('cfg-auto-start').checked;
//     // ← CORREGIDO: leer y guardar human_in_the_loop
//     const humanItl  = document.getElementById('cfg-human-in-the-loop')?.checked ?? true;
//     humanInTheLoop  = humanItl;
//
//     const cfg = {
//         repo_path:             '',
//         auto_commit_enabled:   false,
//         interval_minutes:      interval,
//         auto_start:            autoStart,
//         provider,
//         llm_base_url:          baseUrl,
//         llm_model_name:        model,
//         llm_api_key:           apiKey,
//         smart_mode:            smartMode,
//         smart_threshold_lines: threshold,
//         human_in_the_loop:     humanItl,
//         // Campos legado — se mantienen por compatibilidad con AppConfig pero no se usan
//         push_enabled:   true,
//         push_remote:    'origin',
//         push_branch:    'main',
//         commit_prefix:  '',
//         cooldown_minutes: 0,
//         last_successful_commit: 0,
//         repos:          repos,
//         commit_history: [],
//     };
//     try {
//         await invoke('save_config', { config: cfg });
//         toast('Settings saved', 'success');
//         if (apiKey) {
//             document.getElementById('cfg-api-key').value       = '';
//             document.getElementById('cfg-api-key').placeholder = 'sk-…';
//         }
//     } catch (e) { toast('Failed to save settings: ' + e, 'error'); }
// }

/* ── HISTORY ──────────────────────────────────────────────── */
async function loadHistory() {
    try {
        history = await invoke('get_commit_history') ?? [];
        updateHistoryStats();
        updateHistoryRepoFilter();
        renderHistory();
    } catch (e) { console.error(e); }
}

function updateHistoryStats() {
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

function updateHistoryRepoFilter() {
    const sel   = document.getElementById('history-filter-repo');
    const cur   = sel.value;
    const paths = [...new Set(history.map(h => h.repo_path))];
    sel.innerHTML = '<option value="">All repositories</option>'
        + paths.map(p => {
            const name = p.replace(/\\/g, '/').split('/').pop();
            return `<option value="${escHtml(p)}"${cur === p ? ' selected' : ''}>${escHtml(name)}</option>`;
        }).join('');
}

function renderHistory() {
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

async function exportCSV() {
    try {
        const csv = await invoke('export_history_csv');
        downloadText(csv, 'autocommit-history.csv', 'text/csv');
    } catch (e) { toast('Export failed: ' + e, 'error'); }
}
function exportJSON() {
    downloadText(JSON.stringify(history, null, 2), 'autocommit-history.json', 'application/json');
}
async function clearHistory() {
    if (!confirm('Clear all commit history? This cannot be undone.')) return;
    try {
        await invoke('clear_commit_history');
        history = [];
        updateHistoryStats();
        renderHistory();
        toast('History cleared', 'info');
    } catch (e) { toast('Failed to clear: ' + e, 'error'); }
}
function downloadText(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

/* ── APPROVAL MODAL (Human in the Loop) ──────────────────── */
let approvalPath = '';
let approvalUsedLlm = false;

function openApprovalModal(path, pending) {
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

function closeApprovalModal() {
    document.getElementById('modal-approval').classList.remove('open');
    approvalPath = '';
}

async function confirmApproval() {
    const message     = document.getElementById('approval-message').value.trim();
    const push_enabled = document.getElementById('approval-push-enabled').checked;
    const tag         = document.getElementById('approval-tag').value.trim();
    if (!message) { toast('Commit message cannot be empty', 'error'); return; }

    const btn = document.getElementById('btn-confirm-approval');
    btn.disabled  = true;
    btn.innerHTML = `<span class="spin"></span> Committing`;
    try {
        const result = await invoke('confirm_commit', {
            path: approvalPath,
            message,
            pushEnabled: Boolean(push_enabled),
            usedLlm: approvalUsedLlm,
            tag: tag ? tag : null
        });
        closeApprovalModal();
        toast(`Committed: ${result.message}`, 'success', 5000);
        loadRepos();
        if (currentSection === 'history') loadHistory();
    } catch (e) { toast(`Commit failed: ${e}`, 'error'); }
    finally {
        btn.disabled  = false;
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Confirm &amp; Commit`;
    }
}

/* ── UTILS ────────────────────────────────────────────────── */
function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function formatRelTime(unixSecs) {
    const diff = Math.floor(Date.now() / 1000 - unixSecs);
    if (diff < 60)    return `${diff}s ago`;
    if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

/* ── INIT ─────────────────────────────────────────────────── */
async function init() {
    updateThemeIcon(document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light');

    document.getElementById('theme-btn').addEventListener('click', toggleTheme);
    const logo = document.getElementById('brand-logo');
    if (logo) logo.addEventListener('error', () => { logo.style.display = 'none'; });

    document.querySelectorAll('.nav-item').forEach(item =>
        item.addEventListener('click', () => navigateTo(item.dataset.section)));

    document.getElementById('btn-add-repo').addEventListener('click', openAddRepoModal);
    document.getElementById('btn-add-repo-empty').addEventListener('click', openAddRepoModal);
    document.getElementById('btn-close-repo-modal').addEventListener('click', closeRepoModal);
    document.getElementById('btn-cancel-repo-modal').addEventListener('click', closeRepoModal);
    document.getElementById('btn-save-repo').addEventListener('click', saveRepo);
    document.getElementById('btn-browse-repo').addEventListener('click', pickDirectory);
    document.getElementById('repo-path').addEventListener('click', pickDirectory);
    document.getElementById('repo-push-enabled').addEventListener('change', togglePushBranch);
    document.getElementById('modal-repo').addEventListener('click', e => {
        if (e.target === e.currentTarget) closeRepoModal();
    });

    document.getElementById('btn-close-dryrun-modal').addEventListener('click', closeDryRunModal);
    document.getElementById('btn-cancel-dryrun').addEventListener('click', closeDryRunModal);
    document.getElementById('modal-dryrun').addEventListener('click', e => {
        if (e.target === e.currentTarget) closeDryRunModal();
    });

    // ← NUEVO: eventos modal aprobación
    document.getElementById('btn-close-approval-modal').addEventListener('click', closeApprovalModal);
    document.getElementById('btn-cancel-approval').addEventListener('click', closeApprovalModal);
    document.getElementById('btn-confirm-approval').addEventListener('click', confirmApproval);
    document.getElementById('modal-approval').addEventListener('click', e => {
        if (e.target === e.currentTarget) closeApprovalModal();
    });

    document.getElementById('btn-save-settings').addEventListener('click', saveSettings);
    document.getElementById('cfg-provider').addEventListener('change', onProviderChange);
    document.getElementById('btn-toggle-apikey').addEventListener('click', toggleApiKeyVisibility);
    document.getElementById('btn-test').addEventListener('click', testConnection);
    // ANTES: document.querySelectorAll('#smart-mode-ctrl .segment-btn').forEach(...)
    document.querySelectorAll('.smart-mode-ctrl .segment-btn').forEach(btn =>
        btn.addEventListener('click', () => setSmartMode(btn.dataset.val)));

    document.getElementById('btn-export-csv').addEventListener('click', exportCSV);
    document.getElementById('btn-export-json').addEventListener('click', exportJSON);
    document.getElementById('btn-clear-history').addEventListener('click', clearHistory);
    document.getElementById('history-filter-repo').addEventListener('change', renderHistory);
    document.getElementById('history-filter-type').addEventListener('change', renderHistory);

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') { closeRepoModal(); closeDryRunModal(); closeApprovalModal(); }
    });


    try {
        await invoke('start_auto_commit');
    } catch (e) {

    }

    setupRepoDelegation();
    await loadSettings();
    await loadRepos();
}

document.addEventListener('DOMContentLoaded', init);
