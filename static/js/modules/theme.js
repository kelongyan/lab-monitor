/**
 * 主题切换模块 (Dark / Light)
 */
export function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  
  const icon = document.getElementById('theme-icon');
  const text = document.getElementById('theme-text');
  
  if (next === 'light') {
    if (icon) icon.textContent = '☀️';
    if (text) text.textContent = '日间视效';
  } else {
    if (icon) icon.textContent = '🌙';
    if (text) text.textContent = '暗夜模式';
  }
}

export function initTheme() {
  const btn = document.getElementById('btn-toggle-theme');
  if (btn) {
    btn.addEventListener('click', toggleTheme);
  }
}
