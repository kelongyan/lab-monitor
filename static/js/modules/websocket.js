/**
 * WebSocket 与数据轮询模块
 */
import { escapeHtml, escapeAttr, playAlertBeep } from '../utils/formatter.js';
import { fetchJson } from '../utils/api.js';
import { renderCamGrid, updateCamStatus } from './grid.js';
import { openAlertDetailModal, showTrajectoryModal, pushAlertToModalHistory, clearModalAlertHistory, updateFocusCameraStatus } from './modals.js';

let ws = null;
let totalAlerts = 0;
let totalWarnings = 0;
let visibleAlerts = 0;
let lastAlertId = '';
const seenAlertIds = new Set();
const seenAlertOrder = [];
const MAX_SEEN_ALERT_IDS = 1000;

// 全局告警历史缓存（最近 100 条）
const alertHistoryCache = [];

export function getAlertHistoryCache() {
  return alertHistoryCache;
}

export function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const since = lastAlertId ? `?since=${encodeURIComponent(lastAlertId)}` : '';
  ws = new WebSocket(`${proto}://${location.host}/ws${since}`);
  const statusEl = document.getElementById('ws-status');

  ws.onopen = () => {
    if (statusEl) {
      statusEl.innerHTML = `<div class="pulse-dot"></div><span>WS 实时联线</span>`;
      statusEl.className = 'ws-badge connected';
    }
  };
  ws.onmessage = e => {
    try { addAlert(JSON.parse(e.data)); } catch {}
  };
  ws.onclose = () => {
    if (statusEl) {
      statusEl.innerHTML = `<div class="pulse-dot"></div><span>WS 已断开</span>`;
      statusEl.className = 'ws-badge disconnected';
    }
    setTimeout(connectWS, 1000);
  };
  ws.onerror = () => ws.close();
}

export function createAlertCardElement(alert) {
  const isWarning = alert.stage === 'WARNING';
  const isIntrusion = alert.alert_type === 'INTRUSION';
  const isCrowd = alert.alert_type === 'CROWD_DENSITY';
  const isSceneExit = alert.alert_type === 'SCENE_EXIT';

  const timeStr = new Date(alert.timestamp * 1000).toLocaleTimeString('zh-CN');
  const stage = alert.stage || 'ALERT';
  const card = document.createElement('div');

  const alertClass = isIntrusion ? 'INTRUSION' : (stage === 'WARNING' ? 'WARNING' : alert.risk_level);
  card.className = `alert-card ${alertClass}`;
  card.setAttribute('role', 'button');
  card.tabIndex = 0;
  card.addEventListener('click', () => openAlertDetailModal(alert));
  card.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      openAlertDetailModal(alert);
    }
  });

  let titleText;
  if (isIntrusion)      titleText = 'ROI 区域非法越界入侵';
  else if (isCrowd)     titleText = '区域人流密度超限预警';
  else if (isSceneExit) titleText = '目标全域场景消失失联';
  else if (isWarning)   titleText = '路径通行超时预警';
  else                  titleText = '目标通行超时失联';

  const gidHtml = escapeHtml(alert.global_id);
  const gidAttr = escapeAttr(alert.global_id);
  const riskHtml = escapeHtml(alert.risk_level);
  const camHtml = escapeHtml(alert.last_camera);
  const expectedHtml = (alert.expected_cameras || []).map(c => escapeHtml(c)).join(', ') || '无';
  const elapsedHtml = escapeHtml(alert.elapsed_seconds);

  card.innerHTML = `
    <div class="alert-card-header">
      <div class="alert-type-title">
        <span>${titleText}</span>
      </div>
      <div class="alert-time-tag">${timeStr}</div>
    </div>
    <div class="alert-body-grid">
      <div class="alert-item">目标 ID: <val class="id-link" data-action="trajectory" data-gid="${gidAttr}">#${gidHtml}</val></div>
      <div class="alert-item">风险等级: <val>${riskHtml}</val></div>
      <div class="alert-item">位置: <val>${camHtml}</val></div>
      <div class="alert-item">${isIntrusion ? '状态' : '已用时长'}: <val>${isIntrusion ? '越界入侵' : elapsedHtml + 's'}</val></div>
      <div class="alert-item" style="grid-column: span 2;">预期/区域: <val>${expectedHtml}</val></div>
    </div>
  `;

  const trajLink = card.querySelector('[data-action="trajectory"]');
  if (trajLink && !isCrowd) {
    trajLink.addEventListener('click', (e) => {
      e.stopPropagation();
      showTrajectoryModal(alert.global_id);
    });
  } else if (trajLink) {
    trajLink.style.cursor = 'default';
    trajLink.style.textDecoration = 'none';
  }

  return card;
}

export function addAlert(alert, silent = false) {
  const alertId = alert?.alert_id || [
    alert?.timestamp, alert?.stage, alert?.alert_type,
    alert?.global_id, alert?.last_camera,
  ].join('|');
  if (seenAlertIds.has(alertId)) return false;
  seenAlertIds.add(alertId);
  seenAlertOrder.push(alertId);
  if (seenAlertOrder.length > MAX_SEEN_ALERT_IDS) {
    seenAlertIds.delete(seenAlertOrder.shift());
  }
  if (alert?.alert_id) lastAlertId = alert.alert_id;

  // 缓存至全局告警数组（保留最近 100 条）
  alertHistoryCache.unshift(alert);
  if (alertHistoryCache.length > 100) alertHistoryCache.pop();

  // 同步推送至 Modal 告警历史缓存与焦点视窗
  pushAlertToModalHistory(alert);

  const list = document.getElementById('alert-list');
  const noAlert = document.getElementById('no-alert');
  if (noAlert) noAlert.remove();

  const isWarning = alert.stage === 'WARNING';
  if (!isWarning) visibleAlerts++;
  if (!silent) {
    if (isWarning) {
      totalWarnings++;
      const statWarn = document.getElementById('stat-warnings');
      if (statWarn) statWarn.textContent = totalWarnings;
    } else {
      totalAlerts++;
      const statAlert = document.getElementById('stat-alerts');
      const badge = document.getElementById('alert-badge');
      if (statAlert) statAlert.textContent = totalAlerts;
      playAlertBeep();
    }
  }
  const card = createAlertCardElement(alert);
  if (list) list.insertBefore(card, list.firstChild);
  if (list) {
    while (list.children.length > 30) list.removeChild(list.lastChild);
    visibleAlerts = list.querySelectorAll('.alert-card:not(.WARNING)').length;
  }
  const badge = document.getElementById('alert-badge');
  if (badge) badge.textContent = Math.min(visibleAlerts, 99);
  return true;
}

export function hydrateAlerts(alerts, summary = {}) {
  [...alerts].reverse().forEach(alert => addAlert(alert, true));
  const loadedAlerts = alerts.filter(alert => alert.stage !== 'WARNING').length;
  const loadedWarnings = alerts.length - loadedAlerts;
  totalAlerts = Number.isFinite(summary.alerts) ? summary.alerts : loadedAlerts;
  totalWarnings = Number.isFinite(summary.warnings) ? summary.warnings : loadedWarnings;
  const statAlerts = document.getElementById('stat-alerts');
  const statWarnings = document.getElementById('stat-warnings');
  if (statAlerts) statAlerts.textContent = totalAlerts;
  if (statWarnings) statWarnings.textContent = totalWarnings;
}

export async function pollCalibStats() {
  try {
    const res = await fetchJson('/api/stats', { timeoutMs: 5000 });
    const data = res.calibration || {};
    const container = document.getElementById('calib-content');
    if (!container) return;
    const entries = Object.entries(data);
    if (entries.length === 0) {
      container.innerHTML = '<div class="empty-state" style="padding: 10px 0;">数据积累中（需 5 条记录生效）</div>';
      return;
    }
    container.innerHTML = entries.map(([path, s]) => `
      <div class="calib-row">
        <span class="calib-path">${escapeHtml(path)}</span>
        <span class="calib-stats-val">
          均值 <highlight>${escapeHtml(s.mean_seconds)}s</highlight> (±<highlight>${escapeHtml(s.std_seconds)}s</highlight>)
          样本 <highlight>${escapeHtml(s.count)}</highlight>
        </span>
      </div>
    `).join('');
  } catch (error) {
    console.warn('通行校准统计暂不可用:', error.message);
  }
}

export async function pollStatus() {
  try {
    const [statusRes, idsRes] = await Promise.all([
      fetchJson('/api/status', { timeoutMs: 4000 }),
      fetchJson('/api/identities', { timeoutMs: 4000 }),
    ]);
    const cameras = statusRes.cameras || [];
    renderCamGrid(cameras);
    updateCamStatus(cameras);
    cameras.forEach(updateFocusCameraStatus);
    const statIds = document.getElementById('stat-ids');
    if (statIds) statIds.textContent = idsRes.count || 0;
  } catch (error) {
    console.warn('摄像头状态暂不可用，保留上一轮画面:', error.message);
  }
}

export async function pollReidMetrics() {
  try {
    const res = await fetchJson('/api/metrics/reid', { timeoutMs: 4000 });
    
    const simPct = (res.avg_top1_similarity * 100).toFixed(1);
    const matchRatePct = (res.match_rate * 100).toFixed(1);
    
    const top1 = document.getElementById('reid-top1-sim');
    const matchRate = document.getElementById('reid-match-rate');
    const simBar = document.getElementById('reid-sim-bar');
    if (top1) top1.textContent = res.total_searches > 0 ? `${simPct}%` : '--%';
    if (matchRate) matchRate.textContent = `(匹配率 ${matchRatePct}%)`;
    if (simBar) simBar.style.width = `${Math.min(100, Math.max(0, simPct))}%`;

    const blocked = document.getElementById('reid-ratio-blocked');
    const marginText = document.getElementById('reid-margin-text');
    if (blocked) blocked.textContent = res.ratio_blocked_count || 0;
    const marginPct = (res.avg_ratio_margin * 100).toFixed(1);
    if (marginText) marginText.textContent = `确信裕度: ${res.total_searches > 0 ? marginPct + '%' : '--'}`;

    const latency = document.getElementById('reid-latency');
    const searchesText = document.getElementById('reid-searches-text');
    if (latency) latency.textContent = res.avg_latency_ms ? res.avg_latency_ms.toFixed(1) : '0.0';
    if (searchesText) searchesText.textContent = `总比对: ${res.total_searches || 0} 次`;

    const gallerySize = document.getElementById('reid-gallery-size');
    const qualityText = document.getElementById('reid-quality-text');
    if (gallerySize) gallerySize.textContent = res.gallery_size || 0;
    const qual = res.avg_feature_quality ? res.avg_feature_quality.toFixed(2) : '--';
    if (qualityText) qualityText.textContent = `平均质量: ${qual}`;
    
    const statIds = document.getElementById('stat-ids');
    if (statIds) statIds.textContent = res.gallery_size || 0;
  } catch (e) {
    console.error('获取 ReID 检索指标失败:', e);
  }
}

export function clearAlerts() {
  const list = document.getElementById('alert-list');
  if (list) {
    list.innerHTML = `
      <div class="secure-empty-box" id="no-alert">
        <div class="secure-shield-icon">🛡️</div>
        <div class="secure-empty-title">全域防线感知安防正常</div>
        <div class="secure-empty-sub">
          <span>全网节点巡检中</span>
          <span>·</span>
          <span style="color: var(--success); font-weight: 700;">0 风险隐患 detected</span>
        </div>
        <div class="secure-live-tag">SYSTEM OPERATIONAL</div>
      </div>
    `;
  }
  alertHistoryCache.length = 0;
  clearModalAlertHistory();
  visibleAlerts = 0;
  const alertBadge = document.getElementById('alert-badge');
  if (alertBadge) alertBadge.textContent = '0';
}
