/**
 * API 配置与全局 Toast 提示
 */
export const BASE = location.protocol + '//' + location.host;

// 后端对写请求（POST/PUT/PATCH/DELETE）设有守卫：缺少该自定义头一律 403，
// 用于挡掉跨站表单/img 之类的简单请求。所有写接口都经 fetchJson 统一出口注入。
export const WRITE_GUARD_HEADER = 'X-Lab-Monitor-Request';
const WRITE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

function withWriteGuardHeader(fetchOptions) {
  const method = String(fetchOptions.method || 'GET').toUpperCase();
  if (!WRITE_METHODS.has(method)) return fetchOptions;
  const headers = fetchOptions.headers;
  if (typeof Headers !== 'undefined' && headers instanceof Headers) {
    headers.set(WRITE_GUARD_HEADER, '1');
    return fetchOptions;
  }
  return {
    ...fetchOptions,
    headers: { ...(headers || {}), [WRITE_GUARD_HEADER]: '1' },
  };
}

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export async function fetchJson(url, options = {}) {
  const { timeoutMs = 5000, signal: externalSignal, ...rawOptions } = options;
  const fetchOptions = withWriteGuardHeader(rawOptions);
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
