import * as api from './api.js';
import * as dom from './dom.js';
import { openEditRepoModal, openDryRun, openApprovalModal } from './modals.js';

export let reposList = [];

export async function initRepos() {
    setupRepoDelegation();
    await loadRepos();
}

export async function loadRepos() {
    try {
        reposList = await api.getRepos() ?? [];
        renderRepos();
    } catch (e) { console.error(e); }
}

function renderRepos() {
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
      <p>Add a Git repository to start automating commits.</p>
      <button id="btn-add-repo-empty" class="btn btn-primary">Add repository</button>
    `;
    }

    if (empty.parentNode === grid) grid.removeChild(empty);

    if (!reposList.length) {
        grid.innerHTML = '';
        grid.appendChild(empty);
        empty.style.display = '';
        if (sub) sub.textContent = 'No repositories configured';
        document.getElementById('btn-add-repo-empty')?.addEventListener('click', () => openEditRepoModal(null));
        return;
    }

    empty.style.display = 'none';
    if (sub) sub.textContent = `${reposList.length} repositor${reposList.length === 1 ? 'y' : 'ies'} configured`;

    grid.innerHTML = reposList.map(repo => {
        const parts    = repo.path.replace(/\\/g, '/').split('/');
        const repoName = parts.pop();
        const dirPath  = parts.join('/');
        const pushLabel = dom.escHtml(`${repo.push_remote || 'origin'}/${repo.push_branch || 'main'}`);
        return `
    <div class="repo-card" id="repo-card-${repo.id}">
      <div class="repo-card-header">
        <div class="repo-path">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;color:var(--color-text-faint)"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
          <div><span class="repo-path-dir">${dom.escHtml(dirPath)}/</span><span class="repo-path-name">${dom.escHtml(repoName)}</span></div>
        </div>
        <div class="repo-actions">
          <label class="repo-toggle" title="${repo.enabled ? 'Enabled' : 'Paused'}">
            <input type="checkbox" class="repo-enabled-toggle" data-id="${repo.id}" ${repo.enabled ? 'checked' : ''}>
            <span class="repo-toggle-slider"></span>
          </label>
          <div class="dropdown" id="dd-${repo.id}">
            <button class="btn btn-ghost btn-icon repo-dropdown-btn" data-id="${repo.id}">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" pointer-events="none"><circle cx="12" cy="5" r="1" fill="currentColor"/><circle cx="12" cy="12" r="1" fill="currentColor"/><circle cx="12" cy="19" r="1" fill="currentColor"/></svg>
            </button>
            <div class="dropdown-menu" id="ddm-${repo.id}">
              <div class="dropdown-item repo-edit-btn" data-id="${repo.id}">Edit</div>
              <div class="dropdown-divider"></div>
              <div class="dropdown-item danger repo-delete-btn" data-id="${repo.id}">Remove</div>
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
        ${repo.last_commit_time ? `<div class="repo-meta-item"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>Last ${dom.formatRelTime(repo.last_commit_time)}</div>` : ''}
        <span class="badge ${repo.enabled ? 'badge-running' : 'badge-paused'}">${repo.enabled ? '&nbsp;Running&nbsp;' : 'Paused'}</span>
      </div>
      <div class="repo-card-actions">
        <button class="btn btn-primary btn-sm repo-commit-btn" data-id="${repo.id}" data-path="${repo.path}" id="commit-btn-${repo.id}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" pointer-events="none"><polyline points="20 6 9 17 4 12"/></svg> Commit now
        </button>
        <button class="btn btn-secondary btn-sm repo-dryrun-btn" data-path="${repo.path}">Dry run</button>
        <button class="btn btn-secondary btn-sm repo-diff-btn" data-path="${repo.path}" data-target="diff-${repo.id}">Diff</button>
        <div class="diff-info" id="diff-${repo.id}"></div>
      </div>
    </div>`;
    }).join('');

    reposList.forEach(r => loadBranchForRepo(r));
}

function setupRepoDelegation() {
    const grid = document.getElementById('repos-grid');
    grid.addEventListener('change', e => {
        const toggle = e.target.closest('.repo-enabled-toggle');
        if (toggle) toggleRepo(toggle.dataset.id, toggle.checked);
    });
    grid.addEventListener('click', e => {
        const target = e.target;
        const ddBtn = target.closest('.repo-dropdown-btn');
        if (ddBtn) { e.stopPropagation(); toggleDropdown(ddBtn.dataset.id); return; }

        const editBtn = target.closest('.repo-edit-btn');
        if (editBtn) { openEditRepoModal(editBtn.dataset.id); closeDropdown(editBtn.dataset.id); return; }

        const delBtn = target.closest('.repo-delete-btn');
        if (delBtn) { deleteRepo(delBtn.dataset.id); closeDropdown(delBtn.dataset.id); return; }

        const commitBtn = target.closest('.repo-commit-btn');
        if (commitBtn) { commitNow(commitBtn.dataset.id, commitBtn.dataset.path); return; }

        const dryBtn = target.closest('.repo-dryrun-btn');
        if (dryBtn) { openDryRun(dryBtn.dataset.path); return; }

        const diffBtn = target.closest('.repo-diff-btn');
        if (diffBtn) { previewDiff(diffBtn.dataset.path, diffBtn.dataset.target); }
    });

    document.addEventListener('click', e => {
        if (!e.target.closest('.dropdown')) {
            document.querySelectorAll('.dropdown-menu.open').forEach(m => m.classList.remove('open'));
        }
    });
}

function toggleDropdown(id) {
    const menu = document.getElementById(`ddm-${id}`);
    const isOpen = menu.classList.contains('open');
    document.querySelectorAll('.dropdown-menu.open').forEach(m => m.classList.remove('open'));
    if (!isOpen) menu.classList.add('open');
}

function closeDropdown(id) {
    document.getElementById(`ddm-${id}`)?.classList.remove('open');
}

async function loadBranchForRepo(repo) {
    try {
        const branch = await api.getCurrentBranch(repo.path);
        const el = document.querySelector(`#branch-${repo.id} span`);
        if (el) el.textContent = dom.escHtml(branch);
    } catch (e) { }
}

async function previewDiff(path, targetId) {
    try {
        const stats = await api.getDiffPreview(path);
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
    const repo = reposList.find(r => r.id === id);
    if (!repo) return;
    repo.enabled = enabled;
    try {
        await api.updateRepo(repo);
        dom.toast(`Repository ${enabled ? 'enabled' : 'paused'}`, 'info');
        loadRepos();
    } catch (e) { dom.toast('Failed to update: ' + e, 'error'); }
}

async function deleteRepo(id) {
    if (!confirm('Remove this repository from AutoCommit? Git history is not affected.')) return;
    try {
        await api.removeRepo(id);
        dom.toast('Repository removed', 'info');
        loadRepos();
    } catch (e) { dom.toast('Failed to remove: ' + e, 'error'); }
}

export async function commitNow(id, path) {
    const btn = id ? document.getElementById(`commit-btn-${id}`) : null;
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spin"></span> Committing`;
    }

    try {
        const result = await api.runCommit(path);
        if (result.message === 'No changes to commit' || result.message.startsWith('Cooldown')) {
            dom.toast(result.message, 'info');
        } else if (result.pending_approval) {
            openApprovalModal(path, result.pending_approval);
        } else {
            const shortMessage = result.message.split('\n')[0];
            dom.toast(`Committed: ${shortMessage}`, 'success', 5000);
            loadRepos();
        }
    } catch (e) { dom.toast(`Commit failed: ${e}`, 'error'); }
    finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" pointer-events="none"><polyline points="20 6 9 17 4 12"/></svg> Commit now`;
        }
    }
}