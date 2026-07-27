/**
 * ROI 电子围栏绘制交互模块
 */
import { BASE, showToast } from '../utils/api.js';
import { openModal, closeModal } from './modals.js';

let currentDrawingCamId = '';
let roiPoints = [];
let roiDrawMode = 'freehand';
let canvas = null;
let ctx = null;

export function setRoiDrawMode(mode) {
  roiDrawMode = mode;
  const btnFree = document.getElementById('roi-mode-freehand');
  const btnClick = document.getElementById('roi-mode-click');
  if (btnFree) btnFree.classList.toggle('active', mode === 'freehand');
  if (btnClick) btnClick.classList.toggle('active', mode === 'click');
  redrawRoiCanvas();
}

export function initRoiCanvas() {
  canvas = document.getElementById('roi-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');

  let isMouseDown = false;
  let hasDragged = false;

  canvas.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    hasDragged = false;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = parseFloat((clickX / rect.width).toFixed(4));
    const normY = parseFloat((clickY / rect.height).toFixed(4));

    if (roiDrawMode === 'freehand') {
      roiPoints = [[normX, normY]];
      redrawRoiCanvas();
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isMouseDown) return;
    hasDragged = true;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;
    const normX = parseFloat((curX / rect.width).toFixed(4));
    const normY = parseFloat((curY / rect.height).toFixed(4));

    if (roiDrawMode === 'freehand') {
      const last = roiPoints[roiPoints.length - 1];
      if (last) {
        const dx = normX - last[0];
        const dy = normY - last[1];
        if (Math.sqrt(dx * dx + dy * dy) > 0.012) {
          roiPoints.push([normX, normY]);
          redrawRoiCanvas();
        }
      }
    }
  });

  const handleDrawEnd = (e) => {
    if (!isMouseDown) return;
    isMouseDown = false;

    if (roiDrawMode === 'click' && !hasDragged && e.type === 'mouseup') {
      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;
      const normX = parseFloat((clickX / rect.width).toFixed(4));
      const normY = parseFloat((clickY / rect.height).toFixed(4));
      roiPoints.push([normX, normY]);
      redrawRoiCanvas();
    } else if (roiDrawMode === 'freehand') {
      redrawRoiCanvas();
    }
  };

  canvas.addEventListener('mouseup', handleDrawEnd);
  canvas.addEventListener('mouseleave', () => { isMouseDown = false; });
}

export async function openRoiDrawModal(cam_id) {
  currentDrawingCamId = cam_id;
  roiPoints = [];
  
  const title = document.getElementById('roi-modal-title');
  const bgImg = document.getElementById('roi-preview-bg');
  const nameInput = document.getElementById('roi-name-input');
  
  if (title) title.textContent = `绘制围栏: ${cam_id.toUpperCase()}`;
  if (bgImg) bgImg.src = `${BASE}/stream/${cam_id}`;
  if (nameInput) nameInput.value = '核心机房防护区';
  
  openModal('roi-modal');
  
  try {
    const res = await fetch('/api/roi').then(r => r.json());
    if (res[cam_id] && res[cam_id].length > 0) {
      const existing = res[cam_id][0];
      if (nameInput) nameInput.value = existing.name || '核心机房防护区';
      roiPoints = existing.polygon || [];
    }
  } catch(e) {}

  setTimeout(redrawRoiCanvas, 150);
}

export function clearRoiPoints() {
  roiPoints = [];
  redrawRoiCanvas();
}

export function redrawRoiCanvas() {
  if (!canvas || !ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (roiPoints.length === 0) {
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(20, 20, w - 40, h - 40);
    ctx.setLineDash([]);

    ctx.fillStyle = '#38bdf8';
    ctx.font = '13px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'center';
    const tip = roiDrawMode === 'freehand' 
      ? '🎨 按住鼠标左键在画面上拖拽自由画出防区曲线' 
      : '📍 鼠标依次点击画面添加顶点标定多边形';
    ctx.fillText(tip, w / 2, h / 2);
    return;
  }

  ctx.strokeStyle = '#f59e0b';
  ctx.fillStyle = 'rgba(245, 158, 11, 0.28)';
  ctx.lineWidth = 2.5;

  ctx.beginPath();
  roiPoints.forEach((p, idx) => {
    const px = p[0] * w;
    const py = p[1] * h;
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });

  if (roiPoints.length >= 3) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();

  roiPoints.forEach((p, idx) => {
    const px = p[0] * w;
    const py = p[1] * h;
    
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = '700 10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(idx + 1, px, py - 9);
  });
}

export async function saveRoiPoints() {
  if (!currentDrawingCamId) {
    showToast('未识别到目标摄像头，请重试打开围栏画板！', 'error');
    return;
  }

  if (roiPoints.length > 0 && roiPoints.length < 3) {
    showToast('多边形电子围栏至少需要 3 个顶点！', 'warning');
    return;
  }

  const MAX_ROI_VERTICES = 64;
  const finalPoints = roiPoints.length > MAX_ROI_VERTICES
    ? roiPoints.slice(0, MAX_ROI_VERTICES)
    : roiPoints;
  if (roiPoints.length > MAX_ROI_VERTICES) {
    showToast(`顶点数已超过上限，已自动截取前 ${MAX_ROI_VERTICES} 个顶点。`, 'warning');
  }

  const nameInput = document.getElementById('roi-name-input');
  const name = (nameInput ? nameInput.value.trim() : '') || '自定义电子围栏';
  try {
    const resp = await fetch('/api/roi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        camera_id: currentDrawingCamId,
        polygon: finalPoints,
        name: name
      })
    });
    
    const res = await resp.json();

    if (resp.ok && res.status === 'success') {
      showToast(`摄像头 ${currentDrawingCamId.toUpperCase()} 的 ROI 围栏保存并实时生效成功！`, 'success');
      closeModal('roi-modal');
    } else {
      showToast('保存失败: ' + (res.error || `HTTP ${resp.status}`), 'error');
    }
  } catch(e) {
    showToast('保存请求异常: ' + e.message, 'error');
  }
}

export async function removeRoi() {
  if (!currentDrawingCamId) {
    showToast('未识别到目标摄像头！', 'error');
    return;
  }

  roiPoints = [];
  redrawRoiCanvas();
  const nameInput = document.getElementById('roi-name-input');
  const name = nameInput ? nameInput.value.trim() : '';
  try {
    const resp = await fetch('/api/roi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        camera_id: currentDrawingCamId,
        polygon: [],
        name: name
      })
    });
    
    const res = await resp.json();

    if (resp.ok && res.status === 'success') {
      showToast(`已成功清除摄像头 ${currentDrawingCamId.toUpperCase()} 的电子围栏配置！`, 'success');
      closeModal('roi-modal');
    } else {
      showToast('清除失败: ' + (res.error || `HTTP ${resp.status}`), 'error');
    }
  } catch(e) {
    showToast('清除操作异常: ' + e.message, 'error');
  }
}
