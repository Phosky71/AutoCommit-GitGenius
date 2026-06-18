import * as api from './api.js';
import * as dom from './dom.js';

let commitHistory = [];

export function initHistory() {
    document.getElementById('btn-export-csv').addEventListener('click', exportCSV);
    document.getElementById('btn-export-json').addEventListener('click', exportJSON);
    document.getElementById('btn-clear-history').addEventListener('click', clearHistory);
    document.getElementById('history-filter-repo').addEventListener('change', renderHistory);
    document.getElementById('history-filter-type').addEventListener('change', renderHistory);
}

export async function loadHistory() {
    try {
        commitHistory = await api.getCommitHistory() ?? [];
        updateHistoryStats();
        updateHistoryRepoFilter();
        renderHistory();
    } catch (e) { console.error(e); }
}

function updateHistoryStats() {
    const ai = commitHistory.filter(h => h.used_llm).length;
    const tokens = commitHistory.reduce((s, h) => s + (h.estimated_tokens || 0), 0);
    dom.animateNum('stat-total', commitHistory.length);
    dom.animateNum('stat-ai', ai);
    dom.animateNum('stat-heuristic', commitHistory.length - ai);
    dom.animateNum('stat-tokens', tokens);
}

function updateHistoryRepoFilter() {
    const sel = document.getElementById('history-filter-repo');
    const cur = sel.value;
    const paths = [...new Set(commitHistory.map(h => h.repo_path))];
    sel.innerHTML = '<option value="">All repositories</option>'
        + paths.map(p => {
            const name = p.replace(/\\/g, '/').split('/').pop();
            return `<option value="${dom.escHtml(p)}"${cur === p ? ' selected' : ''}>${dom.escHtml(name)}</option>`;
        }).join('');
}

function renderHistory() {
    const filterRepo = document.getElementById('history-filter-repo').value;
    const filterType = document.getElementById('history-filter-type').value;
    const list = document.getElementById('history-list');
    const empty = document.getElementById('history-empty');

    let filtered = commitHistory;
    if (filterRepo) filtered = filtered.filter(h => h.repo_path === filterRepo);
    if (filterType === 'ai') filtered = filtered.filter(h => h.used_llm);
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
        const date = new Date((entry.timestamp || 0) * 1000).toLocaleString();
        return `
    <div class="history-item">
      <div class="history-item-header">
        <span class="history-msg">${dom.escHtml(entry.message)}</span>
        <span class="badge ${entry.used_llm ? 'badge-ai' : 'badge-heuristic'}">${entry.used_llm ? 'AI' : 'Heuristic'}</span>
      </div>
      <div class="history-meta">
        <div class="history-meta-item"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>${dom.escHtml(repoName)}</div>
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
        const csv = await api.exportHistoryCsv();
        dom.downloadText(csv, 'autocommit-history.csv', 'text/csv');
    } catch (e) { dom.toast('Export failed: ' + e, 'error'); }
}

function exportJSON() {
    dom.downloadText(JSON.stringify(commitHistory, null, 2), 'autocommit-history.json', 'application/json');
}

async function clearHistory() {
    if (!confirm('Clear all commit history? This cannot be undone.')) return;
    try {
        await api.clearCommitHistory();
        commitHistory = [];
        updateHistoryStats();
        renderHistory();
        dom.toast('History cleared', 'info');
    } catch (e) { dom.toast('Failed to clear: ' + e, 'error'); }
}