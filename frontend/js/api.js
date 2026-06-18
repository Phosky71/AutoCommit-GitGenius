/* ── TAURI BRIDGE ─────────────────────────────────────────── */
export const invoke = (cmd, args) => window.__TAURI__?.core?.invoke(cmd, args) ?? Promise.resolve(null);
export const listen = (ev, cb)    => window.__TAURI__?.event?.listen(ev, cb)   ?? (() => {});

/* ── STATE EXPORTS ────────────────────────────────────────── */
export let repos = [];
export let history = [];
export let timerRunning = false;
export let smartMode = 'smart';
export let humanInTheLoop = true;

export const stateControls = {
    setRepos: (data) => repos = data,
    setHistory: (data) => history = data,
    setTimerRunning: (val) => timerRunning = val,
    setSmartMode: (val) => smartMode = val,
    setHitl: (val) => humanInTheLoop = val
};