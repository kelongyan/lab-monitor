/**
 * ROI 电子围栏绘制交互模块
 */
import { BASE, fetchJson, showToast } from '../utils/api.js';
import { openModal, closeModal } from './modals.js';

// ─── Douglas-Peucker 多边形简化 ─────────────────────────────────────────────
// 计算点 p 到线段 (a, b) 的垂直距离（归一化坐标）
function _ptLineDistance(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  if (dx === 0 && dy === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1,
    ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
  ));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

/**
 * Douglas-Peucker 递归简化
 * @param {number[][]} points  - 归一化坐标点数组 [[x,y], ...]
 * @param {number}     epsilon - 容差（归一化坐标，0.008 ≈ 5px@640px画布）
 * @returns {number[][]} 简化后的点数组
 */
function simplifyPolygon(points, epsilon = 0.008) {
  if (points.length <= 3) return points;
  function dp(pts) {
    if (pts.length <= 2) return pts;
    let maxDist = 0, maxIdx = 0;
    const first = pts[0], last = pts[pts.length - 1];
    for (let i = 1; i < pts.length - 1; i++) {
      const d = _ptLineDistance(pts[i], first, last);
      if (d > maxDist) { maxDist = d; maxIdx = i; }
    }
    if (maxDist > epsilon) {
      const left  = dp(pts.slice(0, maxIdx + 1));
      const right = dp(pts.slice(maxIdx));
      return [...left.slice(0, -1), ...right];
    }
    return [first, last];
  }
  return dp(points);
}
// ────────────────────────────────────────────────────────────────────────────

let currentDrawingCamId = '';
let roiPoints = [];
let roiDrawMode = 'freehand';
let canvas = null;
let ctx = null;
let roiRequestController = null;
let roiSessionGeneration = 0;
let roiRequestInFlight = false;

function setRoiBusy(isBusy) {
  roiRequestInFlight = isBusy;
  ['btn-save-roi', 'btn-remove-roi'].forEach(id => {
    const button = document.getElementById(id);
    if (button) button.disabled = isBusy;
  });
}

function getImageRect(targetWidth, targetHeight) {
  const image = document.getElementById('roi-preview-bg');
  const sourceWidth = image?.naturalWidth || 16;
  const sourceHeight = image?.naturalHeight || 9;
  const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return {
    left: (targetWidth - width) / 2,
    top: (targetHeight - height) / 2,
    width,
    height,
  };
}

function eventToNormalizedPoint(event) {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const imageRect = getImageRect(rect.width, rect.height);
  const x = event.clientX - rect.left - imageRect.left;
  const y = event.clientY - rect.top - imageRect.top;
  if (x < 0 || y < 0 || x > imageRect.width || y > imageRect.height) return null;
  return [
    Number((x / imageRect.width).toFixed(4)),
    Number((y / imageRect.height).toFixed(4)),
  ];
}

function simplifyToVertexLimit(points, maxVertices) {
  let epsilon = 0.008;
  let simplified = simplifyPolygon(points, epsilon);
  while (simplified.length > maxVertices && epsilon < 0.5) {
    epsilon *= 1.5;
    simplified = simplifyPolygon(points, epsilon);
  }
  return simplified;
}

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
  document.getElementById('roi-modal')?.addEventListener('modal:closed', () => {
    roiRequestController?.abort();
    roiRequestController = null;
    roiSessionGeneration++;
    setRoiBusy(false);
  });

  let isMouseDown = false;
  let hasDragged = false;

  canvas.addEventListener('mousedown', (e) => {
    const point = eventToNormalizedPoint(e);
    if (!point) return;
    isMouseDown = true;
    hasDragged = false;

    if (roiDrawMode === 'freehand') {
      roiPoints = [point];
      redrawRoiCanvas();
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isMouseDown) return;
    hasDragged = true;
    const point = eventToNormalizedPoint(e);
    if (!point) return;
    const [normX, normY] = point;

    if (roiDrawMode === 'freehand') {
      const last = roiPoints[roiPoints.length - 1];
      if (last) {
        const dx = normX - last[0];
        const dy = normY - last[1];
        // 阈值从 0.012 降到 0.003（约 2px），仅过滤原地微抖动
        // 保存时由 Douglas-Peucker 自动精简冗余点
        if (Math.sqrt(dx * dx + dy * dy) > 0.003) {
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
      const point = eventToNormalizedPoint(e);
      if (point) roiPoints.push(point);
      redrawRoiCanvas();
    } else if (roiDrawMode === 'freehand') {
      redrawRoiCanvas();
    }
  };

  canvas.addEventListener('mouseup', handleDrawEnd);
  canvas.addEventListener('mouseleave', () => { isMouseDown = false; });
}

export async function openRoiDrawModal(cam_id) {
  roiRequestController?.abort();
  roiRequestController = new AbortController();
  const controller = roiRequestController;
  const generation = ++roiSessionGeneration;
  currentDrawingCamId = cam_id;
  roiPoints = [];
  
  const title = document.getElementById('roi-modal-title');
  const bgImg = document.getElementById('roi-preview-bg');
  const nameInput = document.getElementById('roi-name-input');
  
  if (title) title.textContent = `绘制围栏: ${cam_id.toUpperCase()}`;
  if (bgImg) bgImg.src = `${BASE}/stream/${encodeURIComponent(cam_id)}`;
  if (nameInput) nameInput.value = '核心机房防护区';
  
  openModal('roi-modal');
  
  try {
    const res = await fetchJson('/api/roi', { signal: controller.signal });
    if (
      generation !== roiSessionGeneration
      || currentDrawingCamId !== cam_id
      || !document.getElementById('roi-modal')?.classList.contains('active')
    ) return;
    if (res[cam_id] && res[cam_id].length > 0) {
      const existing = res[cam_id][0];
      if (nameInput) nameInput.value = existing.name || '核心机房防护区';
      roiPoints = existing.polygon || [];
    }
  } catch(e) {
    if (!controller.signal.aborted) {
      showToast(`加载 ROI 失败: ${e.message}`, 'error');
    }
  }

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
  const imageRect = getImageRect(w, h);
  ctx.clearRect(0, 0, w, h);

  if (roiPoints.length === 0) {
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(imageRect.left, imageRect.top, imageRect.width, imageRect.height);
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
    const px = imageRect.left + p[0] * imageRect.width;
    const py = imageRect.top + p[1] * imageRect.height;
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });

  if (roiPoints.length >= 3) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();

  if (roiDrawMode === 'click') {
    // Click 模式：每个顶点画编号红点，方便精确定位
    roiPoints.forEach((p, idx) => {
      const px = imageRect.left + p[0] * imageRect.width;
      const py = imageRect.top + p[1] * imageRect.height;

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
  } else {
    // Freehand 模式：顶点可能几百个，只标记起点和当前端点
    const first = roiPoints[0];
    const last  = roiPoints[roiPoints.length - 1];

    // 起点：绿色
    ctx.fillStyle = '#22c55e';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(imageRect.left + first[0] * imageRect.width, imageRect.top + first[1] * imageRect.height, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // 当前端点：蓝色（绘制中动态提示）
    if (last !== first) {
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(imageRect.left + last[0] * imageRect.width, imageRect.top + last[1] * imageRect.height, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // 右下角显示采集点数与保存后预估顶点数
    const simplified = simplifyPolygon(roiPoints);
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(w - 170, h - 34, 162, 26);
    ctx.fillStyle = '#f59e0b';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`采集 ${roiPoints.length} 点 → 保存后约 ${simplified.length} 顶点`, w - 164, h - 16);
  }
}

export async function saveRoiPoints() {
  if (roiRequestInFlight) return;
  if (!currentDrawingCamId) {
    showToast('未识别到目标摄像头，请重试打开围栏画板！', 'error');
    return;
  }

  if (roiPoints.length > 0 && roiPoints.length < 3) {
    showToast('多边形电子围栏至少需要 3 个顶点！', 'warning');
    return;
  }

  // Freehand 模式：用 Douglas-Peucker 自动精简冗余轨迹点
  // Click 模式：点少，直接使用
  const rawCount = roiPoints.length;
  const MAX_ROI_VERTICES = 64;
  const simplified = roiDrawMode === 'freehand'
    ? simplifyToVertexLimit(roiPoints, MAX_ROI_VERTICES)
    : roiPoints.map(point => [...point]);

  if (simplified.length > 0 && simplified.length < 3) {
    showToast('多边形电子围栏至少需要 3 个顶点，请画更完整的区域！', 'warning');
    return;
  }

  if (simplified.length > MAX_ROI_VERTICES) {
    showToast(`围栏最多允许 ${MAX_ROI_VERTICES} 个顶点，请减少标记点。`, 'warning');
    return;
  }
  const finalPoints = simplified.map(point => [...point]);
  if (roiDrawMode === 'freehand') {
    console.info(`[ROI] Douglas-Peucker 简化: ${rawCount} → ${finalPoints.length} 个顶点`);
  }
  const nameInput = document.getElementById('roi-name-input');
  const name = (nameInput ? nameInput.value.trim() : '') || '自定义电子围栏';
  const cameraId = currentDrawingCamId;
  const generation = roiSessionGeneration;
  roiRequestController?.abort();
  roiRequestController = new AbortController();
  const controller = roiRequestController;
  setRoiBusy(true);
  try {
    const res = await fetchJson('/api/roi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        camera_id: cameraId,
        polygon: finalPoints,
        name: name
      }),
      signal: controller.signal,
      timeoutMs: 8000,
    });
    if (generation !== roiSessionGeneration || cameraId !== currentDrawingCamId) return;
    if (res.status === 'success') {
      showToast(`摄像头 ${cameraId.toUpperCase()} 的 ROI 围栏保存并实时生效成功！`, 'success');
      closeModal('roi-modal');
    } else {
      showToast('保存失败: 服务端未确认配置生效', 'error');
    }
  } catch(e) {
    if (generation === roiSessionGeneration && !controller.signal.aborted) {
      showToast('保存请求异常: ' + e.message, 'error');
    }
  } finally {
    if (generation === roiSessionGeneration) setRoiBusy(false);
  }
}

export async function removeRoi() {
  if (roiRequestInFlight) return;
  if (!currentDrawingCamId) {
    showToast('未识别到目标摄像头！', 'error');
    return;
  }

  const nameInput = document.getElementById('roi-name-input');
  const name = nameInput ? nameInput.value.trim() : '';
  const cameraId = currentDrawingCamId;
  const generation = roiSessionGeneration;
  roiRequestController?.abort();
  roiRequestController = new AbortController();
  const controller = roiRequestController;
  setRoiBusy(true);
  try {
    const res = await fetchJson('/api/roi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        camera_id: cameraId,
        polygon: [],
        name: name
      }),
      signal: controller.signal,
      timeoutMs: 8000,
    });
    if (generation !== roiSessionGeneration || cameraId !== currentDrawingCamId) return;
    if (res.status === 'success') {
      roiPoints = [];
      redrawRoiCanvas();
      showToast(`已成功清除摄像头 ${cameraId.toUpperCase()} 的电子围栏配置！`, 'success');
      closeModal('roi-modal');
    } else {
      showToast('清除失败: 服务端未确认配置生效', 'error');
    }
  } catch(e) {
    if (generation === roiSessionGeneration && !controller.signal.aborted) {
      showToast('清除操作异常: ' + e.message, 'error');
    }
  } finally {
    if (generation === roiSessionGeneration) setRoiBusy(false);
  }
}
