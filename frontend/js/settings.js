import * as api from './api.js';
import * as dom from './dom.js';

let smartMode = 'smart';
export let humanInTheLoop = true;

export async function initSettings() {
    document.getElementById('btn-save-settings').addEventListener('click', saveSettings);
    document.getElementById('cfg-provider').addEventListener('change', onProviderChange);
    document.getElementById('btn-toggle-apikey').addEventListener('click', toggleApiKeyVisibility);
    document.getElementById('btn-test').addEventListener('click', testConnection);

    // FIX 1: Evento para mostrar/ocultar el Token de Git
    document.getElementById('btn-toggle-gittoken').addEventListener('click', () => {
        const input = document.getElementById('cfg-git-token');
        input.type = input.type === 'password' ? 'text' : 'password';
    });

    const openLink = (url) => {
        if (window.__TAURI__ && window.__TAURI__.core) {
            window.__TAURI__.core.invoke('open_url', { url }).catch(() => window.open(url, '_blank'));
        } else {
            window.open(url, '_blank');
        }
    };

    document.getElementById('link-github-token').addEventListener('click', (e) => {
        e.preventDefault();
        openLink("https://github.com/settings/tokens/new?scopes=repo&description=AutoCommit+App");
    });

    document.getElementById('link-gitlab-token').addEventListener('click', (e) => {
        e.preventDefault();
        openLink("https://gitlab.com/-/profile/personal_access_tokens");
    });

    document.getElementById('link-bitbucket-token').addEventListener('click', (e) => {
        e.preventDefault();
        openLink("https://bitbucket.org/account/settings/app-passwords/new");
    });

    document.querySelectorAll('.smart-mode-ctrl .segment-btn').forEach(btn =>
        btn.addEventListener('click', () => setSmartMode(btn.dataset.val))
    );

    await loadSettings();
}

async function loadSettings() {
    try {
        const cfg = await api.loadConfig();
        if (!cfg) return;

        document.getElementById('cfg-provider').value = cfg.provider || 'lmstudio';
        document.getElementById('cfg-base-url').value = cfg.llm_base_url || '';
        document.getElementById('cfg-model').value = cfg.llm_model_name || '';
        document.getElementById('cfg-threshold').value = cfg.smart_threshold_lines || 10;

        // FIX 1: Cargar Git Token
        document.getElementById('cfg-git-token').value = cfg.git_token || '';

        humanInTheLoop = cfg.human_in_the_loop !== undefined ? cfg.human_in_the_loop : true;
        const hitlEl = document.getElementById('cfg-human-in-the-loop');
        if (hitlEl) hitlEl.checked = humanInTheLoop;

        // const maskedKey = await api.getMaskedApiKey();
        // if (maskedKey && maskedKey !== 'sk-...') {
        //     document.getElementById('cfg-api-key').placeholder = maskedKey;
        // }
        document.getElementById('cfg-api-key').value = cfg.llm_api_key || '';

        setSmartMode(cfg.smart_mode || 'smart');
    } catch (e) { console.error('loadSettings', e); }
}

async function saveSettings() {
    const btn = document.getElementById('btn-save-settings');
    btn.disabled = true;

    try {
        const cfg = await api.getConfig();

        cfg.provider = document.getElementById('cfg-provider').value;
        cfg.llm_base_url = document.getElementById('cfg-base-url').value.trim();
        cfg.llm_model_name = document.getElementById('cfg-model').value.trim();

        const apiKey = document.getElementById('cfg-api-key').value.trim();
        if (apiKey) cfg.llm_api_key = apiKey;

        cfg.git_token = document.getElementById('cfg-git-token').value.trim();

        cfg.smart_mode = smartMode;
        cfg.smart_threshold_lines = parseInt(document.getElementById('cfg-threshold').value) || 10;
        cfg.human_in_the_loop = document.getElementById('cfg-human-in-the-loop').checked;
        humanInTheLoop = cfg.human_in_the_loop;

        await api.saveConfig(cfg);
        dom.toast('Settings saved', 'success');

        // if (apiKey) {
        //     document.getElementById('cfg-api-key').value = '';
        //     document.getElementById('cfg-api-key').placeholder = 'sk-…';
        // }
    } catch (e) { dom.toast('Failed to save settings: ' + e, 'error'); }
    finally { btn.disabled = false; }
}

async function onProviderChange() {
    const provider = document.getElementById('cfg-provider').value;
    try {
        const [baseUrl, model] = await api.getProviderDefaults(provider);
        document.getElementById('cfg-base-url').value = baseUrl;
        document.getElementById('cfg-model').value = model;
    } catch (e) { }
}

function toggleApiKeyVisibility() {
    const input = document.getElementById('cfg-api-key');
    input.type = input.type === 'password' ? 'text' : 'password';
}

function setSmartMode(val) {
    smartMode = val;
    document.querySelectorAll('.smart-mode-ctrl .segment-btn').forEach(btn =>
        btn.classList.toggle('active', btn.dataset.val === val));

    const hints = {
        always: 'LLM is always used to generate commit messages',
        smart: 'LLM only when diff is significant (recommended)',
        never: 'Always use heuristic messages — no LLM calls',
    };
    document.getElementById('smart-mode-hint').textContent = hints[val] || '';
    document.getElementById('threshold-group').style.display = val === 'smart' ? '' : 'none';
}

async function testConnection() {
    const provider = document.getElementById('cfg-provider').value;
    const baseUrl = document.getElementById('cfg-base-url').value.trim();
    const model = document.getElementById('cfg-model').value.trim();
    const apiKey = document.getElementById('cfg-api-key').value.trim();
    const result = document.getElementById('test-result');
    const btn = document.getElementById('btn-test');

    btn.disabled = true;
    btn.innerHTML = '<span class="spin"></span>';
    result.style.display = 'none';

    try {
        await api.testConnection({ provider, baseUrl, model, apiKey });
        result.style.display = '';
        result.innerHTML = '<span class="test-status ok">Connection successful</span>';
    } catch (e) {
        result.style.display = '';
        result.innerHTML = `<span class="test-status fail">${dom.escHtml(e)}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Test connection';
    }
}