/**
 * 前端 App 主入口脚本
 */
import { initTheme } from './modules/theme.js';
import { setGridMode, setFocusCamera } from './modules/grid.js';
import { connectWS, pollStatus, pollCalibStats, pollReidMetrics, clearAlerts, addAlert } from './modules/websocket.js';
import { 
  closeModal, 
  openIdentitySearchModal, 
  openTopologyModal, 
  openHistoryExportModal 
} from './modules/modals.js';
import { 
  initRoiCanvas, 
  openRoiDrawModal, 
  clearRoiPoints, 
  saveRoiPoints, 
  removeRoi, 
  setRoiDrawMode 
} from './modules/roi.js';

// 初始化全局数字时钟
function startClock() {
  const clockEl = document.getElementById('realtime-clock');
  function update() {
    if (clockEl) {
      const now = new Date();
      clockEl.textContent = now.toLocaleTimeString('zh-CN', { hour12: false }) + '.' + String(now.getMilliseconds()).padStart(3, '0').slice(0, 2);
    }
  }
  update();
  setInterval(update, 500);
}

// 事件代理绑定
function bindEvents() {
  // 主题切换初始化
  initTheme();

  // 数字时钟启动
  startClock();

  // ROI 画布初始化
  initRoiCanvas();

  // 网格模式切换按钮
  ['auto', '1', '2', '1n'].forEach(mode => {
    const btn = document.getElementById(`btn-grid-${mode}`);
    if (btn) {
      btn.addEventListener('click', () => setGridMode(mode));
    }
  });

  // 顶部卡片 & 按钮绑定
  const cardPersons = document.getElementById('card-stat-persons');
  if (cardPersons) {
    cardPersons.addEventListener('click', openIdentitySearchModal);
  }

  const btnTopology = document.getElementById('btn-open-topology');
  if (btnTopology) {
    btnTopology.addEventListener('click', openTopologyModal);
  }

  const btnHistory = document.getElementById('btn-open-history');
  if (btnHistory) {
    btnHistory.addEventListener('click', openHistoryExportModal);
  }

  const btnClearAlerts = document.getElementById('btn-clear-alerts');
  if (btnClearAlerts) {
    btnClearAlerts.addEventListener('click', clearAlerts);
  }

  // Modal 关闭按钮 (所有 data-close 属性的按钮)
  document.querySelectorAll('[data-close-modal]').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-close-modal');
      closeModal(targetId);
    });
  });

  // ROI 画板模式 & 按钮
  const btnRoiFree = document.getElementById('roi-mode-freehand');
  if (btnRoiFree) btnRoiFree.addEventListener('click', () => setRoiDrawMode('freehand'));

  const btnRoiClick = document.getElementById('roi-mode-click');
  if (btnRoiClick) btnRoiClick.addEventListener('click', () => setRoiDrawMode('click'));

  const btnClearRoi = document.getElementById('btn-clear-roi-points');
  if (btnClearRoi) btnClearRoi.addEventListener('click', clearRoiPoints);

  const btnSaveRoi = document.getElementById('btn-save-roi');
  if (btnSaveRoi) btnSaveRoi.addEventListener('click', saveRoiPoints);

  const btnRemoveRoi = document.getElementById('btn-remove-roi');
  if (btnRemoveRoi) btnRemoveRoi.addEventListener('click', removeRoi);

  // 事件代理：捕捉动态生成的相机控件按钮（如 focus, roi 按钮等）
  document.addEventListener('click', (e) => {
    const target = e.target.closest('[data-action]');
    if (!target) return;

    const action = target.getAttribute('data-action');
    const cam = target.getAttribute('data-cam');

    if (action === 'focus' && cam) {
      setFocusCamera(cam);
    } else if (action === 'roi' && cam) {
      openRoiDrawModal(cam);
    }
  });
}

// 主初始化逻辑
async function init() {
  bindEvents();

  try {
    const res = await fetch('/api/alerts/history?limit=30').then(r => r.json());
    if (res.alerts && res.alerts.length > 0) {
      // silent=true：历史回填不触发计数器累加和告警音效
      [...res.alerts].reverse().forEach(a => addAlert(a, true));
    }
  } catch (e) {
    console.error('加载历史告警失败:', e);
  }

  connectWS();
  pollStatus();
  pollCalibStats();
  pollReidMetrics();

  setInterval(pollStatus, 3000);
  setInterval(pollCalibStats, 15000);
  setInterval(pollReidMetrics, 2000);
}

// 页面加载完成后启动
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
