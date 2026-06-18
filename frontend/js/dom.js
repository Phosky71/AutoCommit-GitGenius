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

export function toast(msg, type = 'info', duration = 3500) {
    const tc = document.getElementById('toast-container');
    const t  = document.createElement('div');
    t.className = `toast toast-${type}`;
    const icons = {
        success: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>',
        error:   '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        info:    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    };
    t.innerHTML = `${icons[type] || icons.info} <span>${escHtml(msg)}</span>`;
    tc.appendChild(t);
    const remove = () => {
        t.classList.add('removing');
        t.addEventListener('animationend', () => t.remove(), { once: true });
    };
    const timer = setTimeout(remove, duration);
    t.addEventListener('click', () => { clearTimeout(timer); remove(); });
}

export function downloadText(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

export function animateNum(id, target) {
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