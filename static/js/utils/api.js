/**
 * API 配置与全局 Toast 提示
 */
export const BASE = location.protocol + '//' + location.host;

export function showToast(message, type = 'success', duration = 3000) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }

  const icons = {
    success: '✅',
    warning: '⚠️',
    error: '❌',
    info: 'ℹ️'
  };

  const toast = document.createElement('div');
  toast.className = `toast-message ${type}`;
  
  const iconSpan = document.createElement('span');
  iconSpan.style.fontSize = '16px';
  iconSpan.textContent = icons[type] || 'ℹ️';
  
  const msgSpan = document.createElement('span');
  msgSpan.textContent = message;
  
  toast.appendChild(iconSpan);
  toast.appendChild(msgSpan);

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast-fadeOut');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}
