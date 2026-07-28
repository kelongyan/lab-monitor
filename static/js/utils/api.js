/**
 * API 配置与全局 Toast 提示
 */
export const BASE = location.protocol + '//' + location.host;

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export async function fetchJson(url, options = {}) {
  const { timeoutMs = 5000, signal: externalSignal, ...fetchOptions } = options;
  const controller = new AbortController();
  const abortFromExternal = () => controller.abort(externalSignal?.reason);
  if (externalSignal) {
    if (externalSignal.aborted) abortFromExternal();
    else externalSignal.addEventListener('abort', abortFromExternal, { once: true });
  }
  const timeoutId = setTimeout(() => controller.abort('timeout'), timeoutMs);
  try {
    const response = await fetch(url, { ...fetchOptions, signal: controller.signal });
    const text = await response.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        throw new ApiError(`接口返回了无效 JSON (HTTP ${response.status})`, response.status);
      }
    }
    if (!response.ok || payload?.error) {
      throw new ApiError(
        payload?.error || `请求失败 (HTTP ${response.status})`,
        response.status,
      );
    }
    return payload;
  } catch (error) {
    if (controller.signal.aborted && !externalSignal?.aborted) {
      throw new ApiError('请求超时', 0);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
    externalSignal?.removeEventListener('abort', abortFromExternal);
  }
}

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
