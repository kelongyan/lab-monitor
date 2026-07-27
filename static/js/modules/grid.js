/**
 * 摄像头画面网格与视角控制
 */
import { BASE, showToast } from '../utils/api.js';
import { escapeHtml, escapeAttr } from '../utils/formatter.js';

let currentGridMode = 'auto'; // 'auto', 1, 2, '1n'
let currentFocusCamId = '';
let cachedCameras = [];
let camIds = [];

export function getCachedCameras() {
  return cachedCameras;
}

export function setFocusCamera(camId) {
  if (!camId) return;
  currentFocusCamId = camId;
  if (currentGridMode === 'auto' || currentGridMode === '2') {
    setGridMode('1n');
  } else {
    renderCamGrid(cachedCameras, true);
  }
  showToast(`已将 ${camId.toUpperCase()} 切换为主焦点监视视角`, 'info');
}

export function setGridMode(mode) {
  currentGridMode = mode;
  document.querySelectorAll('.layout-mode-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`btn-grid-${mode}`);
  if (btn) btn.classList.add('active');
  renderCamGrid(cachedCameras, true);
}

export function buildSingleCamCardHTML(cam, isFocus = false) {
  const isReconnecting = cam.status_text === 'RECONNECTING';
  const camIdAttr = escapeAttr(cam.camera_id);
  const camIdHtml = escapeHtml(cam.camera_id);
  return `
    <div class="cam-card ${isFocus ? 'theater-focus-card' : ''}" id="card-${camIdHtml}">
      <div class="cam-header">
        <div class="cam-title-box">
          <div class="cam-status-dot ${cam.is_online ? '' : (isReconnecting ? 'reconnecting' : 'offline')}" id="dot-${camIdHtml}"></div>
          <span class="cam-name">${camIdHtml.toUpperCase()}</span>
        </div>
        <div class="cam-controls">
          <button class="cam-control-btn" title="手动画围栏" data-action="roi" data-cam="${camIdAttr}">ROI 围栏</button>
          <button class="cam-control-btn" title="焦点主视角" data-action="focus" data-cam="${camIdAttr}">焦点</button>
        </div>
        <div class="cam-badges">
          ${cam.reconnect_count > 0 ? `<div class="rtsp-diag-badge" id="reconn-${camIdHtml}">RTSP 重连: ${escapeHtml(cam.reconnect_count)}次</div>` : ''}
          <div class="cam-rec-badge"><div class="rec-circle"></div>REC</div>
          <div class="cam-fps-badge" id="fps-${camIdHtml}">${escapeHtml(cam.fps)} FPS</div>
        </div>
      </div>
      <div class="cam-view">
        <img src="${BASE}/stream/${camIdHtml}" id="stream-img-${camIdHtml}" alt="${camIdHtml}" loading="lazy">
        <div class="cam-hud-tag">LIVE | 1080P MJPEG</div>
      </div>
    </div>
  `;
}

export function renderCamGrid(cameras, forceRefresh = false) {
  cachedCameras = cameras;
  const grid = document.getElementById('cam-grid');
  if (!cameras || cameras.length === 0) return;

  if (!currentFocusCamId || !cameras.find(c => c.camera_id === currentFocusCamId)) {
    currentFocusCamId = cameras[0].camera_id;
  }

  const newIds = cameras.map(c => c.camera_id);
  if (!forceRefresh && JSON.stringify(newIds) === JSON.stringify(camIds)) {
    updateCamStatus(cameras);
    const statCams = document.getElementById('stat-cams');
    if (statCams) statCams.textContent = cameras.filter(c => c.is_online).length;
    return;
  }
  camIds = newIds;

  grid.innerHTML = '';
  grid.removeAttribute('style');

  if (currentGridMode === '1') {
    grid.style.display = 'flex';
    grid.style.flexDirection = 'column';

    const focusCam = cameras.find(c => c.camera_id === currentFocusCamId) || cameras[0];
    const container = document.createElement('div');
    container.className = 'theater-mode-container';
    
    let html = buildSingleCamCardHTML(focusCam, true);

    html += `<div class="cam-carousel-bar">`;
    cameras.forEach(c => {
      const activeCls = c.camera_id === focusCam.camera_id ? 'active' : '';
      const cIdAttr = escapeAttr(c.camera_id);
      const cIdHtml = escapeHtml(c.camera_id);
      html += `
        <div class="carousel-thumb-item ${activeCls}" data-action="focus" data-cam="${cIdAttr}">
          <img src="${BASE}/stream/${cIdHtml}" alt="${cIdHtml}">
          <div class="carousel-thumb-tag">${cIdHtml.toUpperCase()}</div>
        </div>
      `;
    });
    html += `</div>`;

    container.innerHTML = html;
    grid.appendChild(container);

  } else if (currentGridMode === '1n') {
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = '2.3fr 1fr';

    const focusCam = cameras.find(c => c.camera_id === currentFocusCamId) || cameras[0];
    const sideCams = cameras.filter(c => c.camera_id !== focusCam.camera_id);

    const leftBox = document.createElement('div');
    leftBox.innerHTML = buildSingleCamCardHTML(focusCam, true);
    grid.appendChild(leftBox.firstElementChild);

    const sideBox = document.createElement('div');
    sideBox.className = 'layout-1n-side';
    sideCams.forEach(c => {
      const temp = document.createElement('div');
      temp.innerHTML = buildSingleCamCardHTML(c, false);
      const card = temp.firstElementChild;   // cam-card div
      // 直接把 data-action/data-cam 写到 cam-card 上，避免包裹层被 firstElementChild 丢弃
      card.setAttribute('data-action', 'focus');
      card.setAttribute('data-cam', escapeAttr(c.camera_id));
      card.style.cursor = 'pointer';
      sideBox.appendChild(card);
    });
    grid.appendChild(sideBox);

  } else {
    grid.style.display = 'grid';
    if (currentGridMode === 'auto') {
      const cols = cameras.length <= 2 ? 2 : cameras.length <= 4 ? 2 : 3;
      grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    } else {
      grid.style.gridTemplateColumns = `repeat(${currentGridMode}, 1fr)`;
    }

    cameras.forEach(cam => {
      const wrap = document.createElement('div');
      wrap.innerHTML = buildSingleCamCardHTML(cam, false);
      grid.appendChild(wrap.firstElementChild);
    });
  }

  const statCams = document.getElementById('stat-cams');
  if (statCams) statCams.textContent = cameras.filter(c => c.is_online).length;
}

export function updateCamStatus(cameras) {
  cameras.forEach(cam => {
    const dot = document.getElementById(`dot-${cam.camera_id}`);
    const fps = document.getElementById(`fps-${cam.camera_id}`);
    const isReconnecting = cam.status_text === 'RECONNECTING';
    if (dot) dot.className = 'cam-status-dot' + (cam.is_online ? '' : (isReconnecting ? ' reconnecting' : ' offline'));
    if (fps) fps.textContent = cam.fps + ' FPS';
  });
}
