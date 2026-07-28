/**
 * 前端 App 主入口脚本
 */
import { initTheme } from './modules/theme.js';
import { setGridMode, setFocusCamera } from './modules/grid.js';
import { connectWS, pollStatus, pollCalibStats, pollReidMetrics, clearAlerts, hydrateAlerts } from './modules/websocket.js';
import { 
  closeModal, 
  openIdentitySearchModal, 
  openTopologyModal, 
  openHistoryExportModal,
  openFocusModal,
  closeFocusModal
} from './modules/modals.js';
import { 
  initRoiCanvas, 
  openRoiDrawModal, 
  clearRoiPoints, 
  saveRoiPoints, 
  removeRoi, 
  setRoiDrawMode 
} from './modules/roi.js';
import { fetchJson } from './utils/api.js';
import { unlockAlertAudio } from './utils/formatter.js';

const stopPollingTasks = [];

function startPolling(task, intervalMs) {
  let stopped = false;
  let timerId = null;
  const run = async () => {
    if (stopped) return;
    await task();
    if (!stopped) timerId = setTimeout(run, intervalMs);
  };
  run();
  return () => {
    stopped = true;
    if (timerId !== null) clearTimeout(timerId);
  };
}

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

  const unlockAudio = () => {
    unlockAlertAudio();
    document.removeEventListener('pointerdown', unlockAudio);
    document.removeEventListener('keydown', unlockAudio);
  };
  document.addEventListener('pointerdown', unlockAudio, { once: true });
  document.addEventListener('keydown', unlockAudio, { once: true });

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

  // 顶部卡片 & 按钮绑定（stat card 与 header 按钮都绑定同一个处理器）
  ['card-stat-persons', 'card-stat-persons-btn'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('click', openIdentitySearchModal);
      if (el.getAttribute('role') === 'button') {
        el.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            openIdentitySearchModal();
          }
        });
      }
    }
  });

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

  document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (event) => {
      if (event.target === overlay && overlay.id !== 'focus-modal-overlay') {
        closeModal(overlay.id);
      }
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
      openFocusModal(cam);
    } else if (action === 'roi' && cam) {
      openRoiDrawModal(cam);
    } else if (action === 'reset-grid') {
      setGridMode('auto');
      closeFocusModal();
    }
  });
}

// 主初始化逻辑
async function init() {
  bindEvents();

  try {
    const res = await fetchJson('/api/alerts/history?limit=30', { timeoutMs: 5000 });
    // 历史回填初始化累计统计，但不播放告警音效。
    hydrateAlerts(res.alerts || [], res.summary || {});
  } catch (e) {
    console.error('加载历史告警失败:', e);
  }

  connectWS();
  stopPollingTasks.push(startPolling(pollStatus, 1000));
  stopPollingTasks.push(startPolling(pollCalibStats, 15000));
  stopPollingTasks.push(startPolling(pollReidMetrics, 1000));
}

window.addEventListener('beforeunload', () => {
  stopPollingTasks.forEach(stop => stop());
});

// 页面加载完成后启动
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
