import * as api from './api.js';
import * as dom from './dom.js';
import { initRepos, loadRepos, reposList } from './repos.js';
import { initSettings } from './settings.js';
import { initHistory, loadHistory } from './history.js';
import { initModals, openApprovalModal } from './modals.js';

let currentSection = 'repos';

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const btn = document.getElementById('theme-btn');
    if (btn) {
        btn.innerHTML = theme === 'dark'
            ? '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
            : '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
    }
}

async function toggleTheme() {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try {
        const cfg = await api.getConfig();
        if (cfg) {
            cfg.theme = next;
            await api.saveConfig(cfg);
        }
    } catch (e) { }
}

function navigateTo(section) {
    currentSection = section;
    document.querySelectorAll('.nav-item').forEach(i => i.classList.toggle('active', i.dataset.section === section));
    document.querySelectorAll('.section-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${section}`).classList.add('active');

    const titles = { repos: 'Repositories', settings: 'Settings', history: 'Commit History' };
    document.getElementById('topbar-title').textContent = titles[section] || section;

    if (section === 'history') loadHistory();
}

api.listen('commit-status', ev => {
    const r = ev.payload;
    if (r.pending_approval) {
        const activeRepoPath = reposList.find(rp => rp.enabled)?.path ?? '';
        openApprovalModal(activeRepoPath, r.pending_approval);
        return;
    }
    const tag = r.used_llm ? '🤖 ' : '⚙️ ';
    const shortMessage = r.message.split('\n')[0];
    dom.toast(`${tag}${shortMessage}`, 'success', 5000);

    loadRepos();
    if (currentSection === 'history') loadHistory();
});

api.listen('commit-error', ev => dom.toast('Commit error: ' + ev.payload, 'error', 6000));

async function init() {
    applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light');
    document.getElementById('theme-btn').addEventListener('click', toggleTheme);

    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => navigateTo(item.dataset.section));
    });

    initModals();
    await initSettings();
    initHistory();
    await initRepos();

    try { await api.startAutoCommit(); } catch (e) { }
}

document.addEventListener('DOMContentLoaded', init);