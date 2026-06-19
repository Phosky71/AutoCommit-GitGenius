import * as api from './api.js';
import * as dom from './dom.js';
import { reposList, loadRepos, commitNow } from './repos.js';
import { loadHistory } from './history.js';

let editingRepoId = null;
let dryRunPath = '';
let approvalPath = '';
let approvalUsedLlm = false;

export function initModals() {
    // Add/Edit Repo
    document.getElementById('btn-add-repo').addEventListener('click', () => openEditRepoModal(null));
    document.getElementById('btn-save-repo').addEventListener('click', saveRepo);
    document.getElementById('btn-browse-repo').addEventListener('click', pickDirectory);
    document.getElementById('repo-path').addEventListener('click', pickDirectory);
    document.getElementById('repo-push-enabled').addEventListener('change', togglePushBranch);

    // Approval
    document.getElementById('btn-confirm-approval').addEventListener('click', confirmApproval);

    // Global close handlers for all modals
    document.querySelectorAll('.btn-icon, .btn-secondary').forEach(btn => {
        if (btn.id.includes('cancel') || btn.id.includes('close')) {
            btn.addEventListener('click', e => {
                const modal = e.target.closest('.modal-backdrop');
                if (modal) modal.classList.remove('open');
            });
        }
    });

    document.querySelectorAll('.modal-backdrop').forEach(modal => {
        modal.addEventListener('click', e => {
            if (e.target === e.currentTarget) modal.classList.remove('open');
        });
    });

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal-backdrop.open').forEach(m => m.classList.remove('open'));
        }
    });
}

// --- ADD / EDIT REPO ---
export function openEditRepoModal(id) {
    editingRepoId = id;
    const isEdit = !!id;
    const repo = isEdit ? reposList.find(r => r.id === id) : null;
    if (isEdit && !repo) return;

    document.getElementById('modal-repo-title').textContent = isEdit ? 'Edit repository' : 'Add repository';
    document.getElementById('repo-edit-id').value = id || '';
    document.getElementById('repo-path').value = repo ? repo.path : '';
    document.getElementById('repo-path-hint').textContent = '';
    document.getElementById('repo-interval').value = repo ? repo.interval_minutes : 30; // Bug fix #2
    document.getElementById('repo-cooldown').value = repo ? repo.cooldown_minutes : 5;
    document.getElementById('repo-prefix').value = repo ? repo.commit_prefix : '';

    const pushEnabled = repo ? repo.push_enabled : true;
    document.getElementById('repo-push-enabled').checked = pushEnabled;
    document.getElementById('repo-push-remote').value = repo?.push_remote || 'origin';
    document.getElementById('repo-push-branch').value = repo?.push_branch || 'main';

    document.getElementById('push-branch-group').style.display = pushEnabled ? '' : 'none';
    document.getElementById('repo-preview').style.display = 'none';

    if (isEdit) loadBranchOptions(repo.path);

    document.getElementById('modal-repo').classList.add('open');
}

async function pickDirectory() {
    try {
        const path = await api.selectDirectory();
        if (!path) return;
        document.getElementById('repo-path').value = path;
        validateRepoPath(path);
        previewRepoDiff(path);
    } catch (e) { dom.toast('Failed to pick directory: ' + e, 'error'); }
}

async function validateRepoPath(path) {
    const hint = document.getElementById('repo-path-hint');
    try {
        const valid = await api.validateRepoPath(path);
        hint.textContent = valid ? 'Valid Git repository' : 'No .git folder found';
        hint.className = 'form-hint ' + (valid ? 'success' : 'error');
    } catch (e) { hint.textContent = ''; hint.className = 'form-hint'; }
}

async function loadBranchOptions(path) {
    const hint = document.getElementById('push-branch-hint');
    try {
        const branches = await api.listRemoteBranches(path);
        if (branches && branches.length)
            hint.textContent = 'Remote branches: ' + branches.slice(0, 5).join(', ') + (branches.length > 5 ? '…' : '');
        else hint.textContent = '';
    } catch (e) { hint.textContent = ''; }
}

async function previewRepoDiff(path) {
    const preview = document.getElementById('repo-preview');
    const box = document.getElementById('repo-diff-preview');
    preview.style.display = '';
    box.textContent = 'Checking diff…';
    try {
        const stats = await api.getDiffPreview(path);
        if (!stats || stats.files_changed === 0) {
            box.textContent = 'No uncommitted changes';
        } else {
            box.innerHTML = `${stats.files_changed} file(s) changed &nbsp;`
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
    if (!path) { dom.toast('Please select a repository path', 'error'); return; }

    if (!editingRepoId) {
        const alreadyExists = reposList.some(r => r.path === path);
        if (alreadyExists) {
            dom.toast('This repository is already added', 'error');
            return; // Cortamos la ejecución
        }
    }

    const repoObj = {
        id: editingRepoId || crypto.randomUUID(),
        path,
        interval_minutes: parseInt(document.getElementById('repo-interval').value) || 30,
        timer_enabled: false,
        enabled: true,
        push_enabled: document.getElementById('repo-push-enabled').checked,
        push_remote: document.getElementById('repo-push-remote').value.trim() || 'origin',
        push_branch: document.getElementById('repo-push-branch').value.trim() || 'main',
        commit_prefix: document.getElementById('repo-prefix').value.trim(),
        last_commit_time: 0,
        cooldown_minutes: parseInt(document.getElementById('repo-cooldown').value) || 0,
    };

    try {
        if (editingRepoId) {
            const current = reposList.find(r => r.id === editingRepoId);
            repoObj.enabled = current?.enabled ?? true;
            repoObj.last_commit_time = current?.last_commit_time ?? 0;
            await api.updateRepo(repoObj);
            dom.toast('Repository updated', 'success');
        } else {
            await api.addRepo(repoObj);
            dom.toast('Repository added', 'success');
        }
        document.getElementById('modal-repo').classList.remove('open');
        loadRepos();
    } catch (e) { dom.toast('Failed to save: ' + e, 'error'); }
}

// --- DRY RUN ---
export async function openDryRun(path) {
    dryRunPath = path;
    document.getElementById('dryrun-result').innerHTML = '<span class="spin"></span>';
    document.getElementById('dryrun-diff').style.display = 'none';
    document.getElementById('modal-dryrun').classList.add('open');

    try {
        const result = await api.dryRunCommit(path);
        document.getElementById('dryrun-result').textContent = result.message;
        if (result.diff_stats) {
            const s = result.diff_stats;
            document.getElementById('dryrun-diff').style.display = '';
            document.getElementById('dryrun-diff-info').innerHTML =
                `<span class="diff-files">${s.files_changed} file(s)</span>
                 <span class="diff-add">+${s.insertions}</span>
                 <span class="diff-del">-${s.deletions}</span>
                 <span class="diff-files" style="color:var(--color-text-faint)">${s.estimated_tokens} est. tokens</span>`;
        }
        const commitBtn = document.getElementById('btn-commit-from-dry');
        commitBtn.onclick = () => {
            document.getElementById('modal-dryrun').classList.remove('open');
            commitNow(null, path);
        };
    } catch (e) { document.getElementById('dryrun-result').textContent = 'Error: ' + e; }
}

// --- HUMAN IN THE LOOP (APPROVAL) ---
export function openApprovalModal(path, pending) {
    approvalPath = path;
    approvalUsedLlm = pending.used_llm;

    document.getElementById('approval-message').value = pending.message || '';
    document.getElementById('approval-push-enabled').checked = true;
    document.getElementById('approval-tag').value = '';

    const s = pending.diff_stats;
    if (s) {
        document.getElementById('approval-diff-info').innerHTML =
            `<span class="diff-files">${s.files_changed} file(s)</span>
             <span class="diff-add">+${s.insertions}</span>
             <span class="diff-del">-${s.deletions}</span>
             <span class="diff-files" style="color:var(--color-text-faint)">${s.estimated_tokens} est. tokens</span>`;
    } else {
        document.getElementById('approval-diff-info').textContent = '—';
    }

    const fileList = document.getElementById('approval-files-list');
    if (pending.files_changed_list?.length) {
        fileList.innerHTML = pending.files_changed_list.map(f => `<div class="approval-file-item">${dom.escHtml(f)}</div>`).join('');
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

async function confirmApproval() {
    const message = document.getElementById('approval-message').value.trim();
    if (!message) { dom.toast('Commit message cannot be empty', 'error'); return; }

    const btn = document.getElementById('btn-confirm-approval');
    btn.disabled = true;
    btn.innerHTML = `<span class="spin"></span> Committing`;

    try {
        const result = await api.confirmCommit({
            path: approvalPath,
            message,
            pushEnabled: document.getElementById('approval-push-enabled').checked,
            usedLlm: approvalUsedLlm,
            tag: document.getElementById('approval-tag').value.trim() || null
        });

        document.getElementById('modal-approval').classList.remove('open');
        const shortMessage = result.message.split('\n')[0];
        dom.toast(`Committed: ${shortMessage}`, 'success', 5000);
        loadRepos();

        // Comprobar la sección activa desde el main es algo complejo por los imports circulares,
        // así que forzamos la carga del history de forma segura:
        if (document.getElementById('panel-history').classList.contains('active')) {
            loadHistory();
        }
    } catch (e) { dom.toast(`Commit failed: ${e}`, 'error'); }
    finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Confirm &amp; Commit`;
    }
}