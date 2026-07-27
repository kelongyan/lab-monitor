/**
 * 弹窗与模态框交互模块
 */
import { BASE, showToast } from '../utils/api.js';
import { escapeHtml, escapeAttr } from '../utils/formatter.js';
import { getCachedCameras } from './grid.js';

export function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
}

export function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('active');
}

export function openAlertDetailModal(alert) {
  const title = document.getElementById('modal-alert-title');
  const body = document.getElementById('modal-alert-body');
  if (!title || !body) return;

  title.textContent = `告警事件排查 (ID: ${alert.alert_id || 'Alert'})`;

  const timeStr = new Date(alert.timestamp * 1000).toLocaleString('zh-CN');
  const isIntrusion = alert.alert_type === 'INTRUSION';

  const gidHtml = escapeHtml(alert.global_id);
  const gidAttr = escapeAttr(alert.global_id);
  const camHtml = escapeHtml(alert.last_camera);
  const camAttr = escapeAttr(alert.last_camera);
  const stageHtml = escapeHtml(alert.stage || 'ALERT');
  const riskHtml = escapeHtml(alert.risk_level || 'HIGH');
  const elapsedHtml = escapeHtml(alert.elapsed_seconds);
  const alertIdHtml = encodeURIComponent(alert.alert_id || '');
  const expectedHtml = (alert.expected_cameras || []).map(c => escapeHtml(c)).join(', ') || '无';

  body.innerHTML = `
    <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5;">
      摄像头 <b style="color: var(--primary);">${camHtml}</b> 于 <b style="color: var(--text-main);">${timeStr}</b> 检测到预警事件：
    </div>

    <div class="meta-grid">
      <div class="meta-card">
        <div class="label">目标 Global ID</div>
        <div class="val" style="color: var(--primary); cursor: pointer;" id="btn-alert-traj">
          #${gidHtml} (点击看轨迹)
        </div>
      </div>
      <div class="meta-card">
        <div class="label">告警类型/阶段</div>
        <div class="val" style="color: ${isIntrusion ? '#ff0055' : (alert.stage === 'WARNING' ? 'var(--warning)' : 'var(--danger)')};">
          ${isIntrusion ? '🚨 危险区闯入' : stageHtml} (${riskHtml})
        </div>
      </div>
      <div class="meta-card">
        <div class="label">${isIntrusion ? '闯入状态' : '已停留/穿越用时'}</div>
        <div class="val">${isIntrusion ? '触发电子围栏' : elapsedHtml + ' 秒'}</div>
      </div>
      <div class="meta-card">
        <div class="label">预计/区域描述</div>
        <div class="val">${expectedHtml}</div>
      </div>
    </div>

    <div style="margin-top: 8px;">
      <div style="font-size: 12px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px;">📷 现场最后快照:</div>
      <img src="${BASE}/screenshots/${alertIdHtml}.jpg" class="snapshot-preview"
           onerror="this.onerror=null; this.src='${BASE}/stream/${camAttr}';" alt="现场快照">
    </div>
  `;

  const btnTraj = document.getElementById('btn-alert-traj');
  if (btnTraj) {
    btnTraj.addEventListener('click', () => {
      closeModal('alert-detail-modal');
      showTrajectoryModal(alert.global_id);
    });
  }

  openModal('alert-detail-modal');
}

export async function showTrajectoryModal(global_id) {
  const title = document.getElementById('modal-traj-title');
  const body = document.getElementById('modal-traj-body');
  if (!title || !body) return;

  title.textContent = `目标 #${global_id} 跨镜头通行轨迹链`;
  body.innerHTML = `<div class="empty-state">加载人员轨迹数据中...</div>`;
  openModal('trajectory-modal');

  try {
    const res = await fetch(`/api/identities/${global_id}`).then(r => r.json());
    if (res.error) {
      body.innerHTML = `<div class="empty-state">暂无该目标移动轨迹历史</div>`;
      return;
    }

    const traj = res.trajectory || [];
    if (traj.length === 0) {
      body.innerHTML = `<div class="empty-state">未在任何摄像头中形成有效路线</div>`;
      return;
    }

    let nodesHtml = traj.map((t, idx) => `
      <div class="timeline-node">
        <div class="timeline-dot"></div>
        <div class="timeline-content">
          <div class="timeline-header">
            <span class="timeline-cam">节点 ${idx + 1}: ${escapeHtml(t.camera).toUpperCase()}</span>
            <span class="timeline-time">${escapeHtml(t.time_str)} ${t.end_time_str ? '➔ ' + escapeHtml(t.end_time_str) : ''}</span>
          </div>
          <div style="font-size: 11px; color: var(--text-muted);">
            坐标范围 (BBox): [${(t.bbox || []).map(n => Math.round(Number(n))).join(', ')}]
          </div>
        </div>
      </div>
    `).join('');

    body.innerHTML = `
      <div class="meta-grid" style="margin-bottom: 12px;">
        <div class="meta-card">
          <div class="label">累计出现次数</div>
          <div class="val">${escapeHtml(res.total_appearances)} 次抓拍</div>
        </div>
        <div class="meta-card">
          <div class="label">当前所在/最后镜头</div>
          <div class="val" style="color: var(--primary);">${escapeHtml(res.last_camera).toUpperCase()}</div>
        </div>
      </div>

      <div style="font-size: 12px; font-weight: 700; color: var(--text-muted);">📍 移动路线时序链:</div>
      <div class="timeline">
        ${nodesHtml}
      </div>
    `;
  } catch (e) {
    body.innerHTML = `<div class="empty-state">获取轨迹失败: ${e.message}</div>`;
  }
}

export async function openIdentitySearchModal() {
  const title = document.getElementById('modal-traj-title');
  const body = document.getElementById('modal-traj-body');
  if (!title || !body) return;

  title.textContent = `ReID 全局身份档案库`;
  body.innerHTML = `<div class="empty-state">读取全量 ReID 身份中...</div>`;
  openModal('trajectory-modal');

  try {
    const res = await fetch(`/api/identities`).then(r => r.json());
    const ids = res.ids || [];
    if (ids.length === 0) {
      body.innerHTML = `<div class="empty-state">当前尚未登记任何 ReID 人员身份</div>`;
      return;
    }

    body.innerHTML = `
      <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">
        系统已自动提炼注册 <b style="color: var(--purple);">${escapeHtml(ids.length)}</b> 个全局独一无二的人员 ID。点击查看其路径：
      </div>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
        ${ids.map(id => {
          const idAttr = escapeAttr(id);
          const idHtml = escapeHtml(id);
          return `
          <div class="meta-card id-card-btn" style="cursor: pointer;" data-id="${idAttr}">
            <div class="label">Global ID</div>
            <div class="val" style="color: var(--primary);">#${idHtml}</div>
          </div>`;
        }).join('')}
      </div>
    `;

    body.querySelectorAll('.id-card-btn').forEach(card => {
      card.addEventListener('click', () => {
        showTrajectoryModal(card.getAttribute('data-id'));
      });
    });
  } catch(e) {
    body.innerHTML = `<div class="empty-state">加载失败</div>`;
  }
}

export async function openTopologyModal() {
  const body = document.getElementById('modal-topo-body');
  if (!body) return;
  body.innerHTML = `<div class="empty-state">读取与渲染拓扑网络中...</div>`;
  openModal('topology-modal');

  try {
    const [topoRes, statsRes] = await Promise.all([
      fetch('/api/topology').then(r => r.json()),
      fetch('/api/stats').then(r => r.json()).catch(() => ({ calibration: {} }))
    ]);

    const statsData = statsRes.calibration || {};
    let edgesList = [];
    let camNodesSet = new Set();

    if (Array.isArray(topoRes.edges)) {
      edgesList = topoRes.edges;
      edgesList.forEach(e => { camNodesSet.add(e.from); camNodesSet.add(e.to); });
    } else if (typeof topoRes === 'object' && topoRes !== null) {
      for (const [fromCam, hops] of Object.entries(topoRes)) {
        camNodesSet.add(fromCam);
        if (Array.isArray(hops)) {
          hops.forEach(h => {
            const toCam = h.next || h.to;
            camNodesSet.add(toCam);
            edgesList.push({
              from: fromCam,
              to: toCam,
              expected_seconds: h.expected_seconds,
              tolerance_seconds: h.tolerance_seconds || 15
            });
          });
        }
      }
    }

    const camNodes = Array.from(camNodesSet);
    if (camNodes.length === 0) {
      body.innerHTML = `<div class="empty-state">未配置拓扑节点 (config/topology.json)</div>`;
      return;
    }

    const nodeCoords = {};
    const W = 700, H = 340;
    const count = camNodes.length;

    camNodes.forEach((id, idx) => {
      if (count === 4) {
        const coords4 = [
          { x: 160, y: 90 },
          { x: 540, y: 90 },
          { x: 540, y: 250 },
          { x: 160, y: 250 }
        ];
        nodeCoords[id] = coords4[idx] || { x: 350, y: 170 };
      } else {
        const angle = (idx / count) * 2 * Math.PI - Math.PI / 2;
        nodeCoords[id] = {
          x: Math.round(350 + 220 * Math.cos(angle)),
          y: Math.round(170 + 110 * Math.sin(angle))
        };
      }
    });

    let pathsSvg = '';
    let labelsSvg = '';

    edgesList.forEach((e) => {
      const fromPos = nodeCoords[e.from] || { x: 100, y: 100 };
      const toPos = nodeCoords[e.to] || { x: 200, y: 200 };

      const dx = toPos.x - fromPos.x;
      const dy = toPos.y - fromPos.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const nx = -dy / (dist || 1);
      const ny = dx / (dist || 1);
      
      const curveOffset = 25;
      const ctrlX = (fromPos.x + toPos.x) / 2 + nx * curveOffset;
      const ctrlY = (fromPos.y + toPos.y) / 2 + ny * curveOffset;

      const pathD = `M ${fromPos.x} ${fromPos.y} Q ${ctrlX} ${ctrlY} ${toPos.x} ${toPos.y}`;

      const statKey = `${e.from}->${e.to}`;
      const statItem = statsData[statKey];
      const aiTimeText = statItem ? `AI校准: ${statItem.mean_seconds}s` : `预期: ${e.expected_seconds}s`;

      pathsSvg += `
        <path d="${pathD}" stroke="rgba(56, 189, 248, 0.2)" stroke-width="3" fill="none" />
        <path d="${pathD}" stroke="url(#topo-line-grad)" stroke-width="2.5" stroke-dasharray="8 6" fill="none" marker-end="url(#arrow)" class="topo-flow-path" />
      `;

      const labelX = (fromPos.x + toPos.x) / 2 + nx * (curveOffset * 0.75);
      const labelY = (fromPos.y + toPos.y) / 2 + ny * (curveOffset * 0.75);

      labelsSvg += `
        <g transform="translate(${labelX}, ${labelY})">
          <rect x="-42" y="-11" width="84" height="22" rx="11" fill="rgba(15, 23, 42, 0.9)" stroke="rgba(56, 189, 248, 0.4)" stroke-width="1"/>
          <text x="0" y="3" text-anchor="middle" fill="#38bdf8" font-size="10" font-family="JetBrains Mono" font-weight="700">${aiTimeText}</text>
        </g>
      `;
    });

    let nodesSvg = '';
    const cachedCams = getCachedCameras();
    camNodes.forEach(id => {
      const pos = nodeCoords[id];
      const isOnline = cachedCams.some(c => c.camera_id === id && c.is_online);
      const statusColor = isOnline ? '#10b981' : '#ef4444';

      nodesSvg += `
        <g class="topo-node-group" transform="translate(${pos.x}, ${pos.y})" cursor="pointer" data-cam="${escapeAttr(id)}">
          <circle r="32" fill="none" stroke="${statusColor}" stroke-opacity="0.3" stroke-width="2">
            <animate attributeName="r" values="28;38;28" dur="3s" repeatCount="indefinite"/>
            <animate attributeName="stroke-opacity" values="0.4;0.0;0.4" dur="3s" repeatCount="indefinite"/>
          </circle>
          <circle r="26" fill="rgba(15, 23, 42, 0.95)" stroke="${statusColor}" stroke-width="2" box-shadow="0 0 16px ${statusColor}"/>
          <text x="0" y="-4" text-anchor="middle" fill="#38bdf8" font-size="10" font-family="JetBrains Mono" font-weight="800">CAM</text>
          <text x="0" y="12" text-anchor="middle" fill="#f8fafc" font-size="11" font-family="JetBrains Mono" font-weight="800">${id.toUpperCase()}</text>
        </g>
      `;
    });

    body.innerHTML = `
      <style>
        .topo-flow-path { animation: topoFlow 2.5s linear infinite; }
        @keyframes topoFlow { from { stroke-dashoffset: 28; } to { stroke-dashoffset: 0; } }
        .topo-node-group:hover circle { stroke: #38bdf8 !important; filter: drop-shadow(0 0 8px #38bdf8); }
      </style>
      
      <div style="display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">
        <span><b>空间拓扑感知网络</b> (点击节点图标调出摄像头监视流)</span>
        <span style="display: flex; gap: 12px; font-size: 11px;">
          <span style="color: var(--success);">在线节点</span>
          <span style="color: var(--primary-light);">动态通道流向</span>
        </span>
      </div>

      <div style="background: rgba(6, 9, 19, 0.85); border: 1px solid var(--border-card-glow); border-radius: 12px; overflow: hidden; position: relative;">
        <svg width="100%" height="340" viewBox="0 0 ${W} ${H}">
          <defs>
            <linearGradient id="topo-line-grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#38bdf8" />
              <stop offset="50%" stop-color="#a855f7" />
              <stop offset="100%" stop-color="#38bdf8" />
            </linearGradient>
            <marker id="arrow" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8"/>
            </marker>
          </defs>
          ${pathsSvg}
          ${labelsSvg}
          ${nodesSvg}
        </svg>
      </div>

      <div style="margin-top: 14px; border-top: 1px solid var(--border-card); padding-top: 10px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
          <span style="font-size: 12px; font-weight: 700; color: var(--text-main);">拓扑通道与期望通行秒数在线管理</span>
          <button class="action-btn active" id="btn-save-topo">保存修改并实时生效</button>
        </div>
        <div class="tech-topo-table-wrap">
          <table class="tech-topo-table">
            <thead>
              <tr>
                <th>起始摄像头 (From)</th>
                <th>到达摄像头 (To)</th>
                <th>期望耗时 (秒)</th>
                <th>容忍窗口 (秒)</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody id="topo-edit-tbody">
              ${edgesList.map((e) => `
                <tr class="topo-edge-row">
                  <td><input type="text" class="tech-cam-input topo-from" value="${escapeAttr(e.from)}"></td>
                  <td><input type="text" class="tech-cam-input topo-to" value="${escapeAttr(e.to)}"></td>
                  <td>
                    <div class="tech-num-box">
                      <button class="tech-num-btn btn-step-sub" type="button">-</button>
                      <input type="number" class="tech-num-input topo-exp" value="${e.expected_seconds}" min="1">
                      <button class="tech-num-btn btn-step-add" type="button">+</button>
                    </div>
                  </td>
                  <td>
                    <div class="tech-num-box">
                      <button class="tech-num-btn btn-step-sub" type="button">-</button>
                      <input type="number" class="tech-num-input topo-tol" value="${e.tolerance_seconds || 15}" min="1">
                      <button class="tech-num-btn btn-step-add" type="button">+</button>
                    </div>
                  </td>
                  <td><button class="clear-btn btn-delete-row" type="button" style="color: var(--danger);">删除</button></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        <div style="display: flex; justify-content: flex-start; margin-top: 8px;">
          <button class="action-btn" type="button" id="btn-add-topo-row">+ 新增通道连线</button>
        </div>
      </div>
    `;

    // 绑定事件处理器
    body.querySelectorAll('.topo-node-group').forEach(group => {
      group.addEventListener('click', () => {
        openCamZoomModal(group.getAttribute('data-cam'));
      });
    });

    const btnSaveTopo = document.getElementById('btn-save-topo');
    if (btnSaveTopo) btnSaveTopo.addEventListener('click', saveTopologyConfig);

    const btnAddTopoRow = document.getElementById('btn-add-topo-row');
    if (btnAddTopoRow) btnAddTopoRow.addEventListener('click', addTopologyRow);

    bindTopoTableButtons(body);

  } catch(e) {
    body.innerHTML = `<div class="empty-state">加载拓扑图表失败: ${e.message}</div>`;
  }
}

function bindTopoTableButtons(container) {
  container.querySelectorAll('.btn-step-sub').forEach(btn => {
    btn.addEventListener('click', () => stepNumInput(btn, -5));
  });
  container.querySelectorAll('.btn-step-add').forEach(btn => {
    btn.addEventListener('click', () => stepNumInput(btn, 5));
  });
  container.querySelectorAll('.btn-delete-row').forEach(btn => {
    btn.addEventListener('click', () => btn.closest('tr').remove());
  });
}

export function stepNumInput(btn, delta) {
  const box = btn.closest('.tech-num-box');
  if (!box) return;
  const input = box.querySelector('.tech-num-input');
  if (!input) return;
  let val = (parseFloat(input.value) || 0) + delta;
  if (val < 1) val = 1;
  input.value = val;
}

export function addTopologyRow() {
  const tbody = document.getElementById('topo-edit-tbody');
  if (!tbody) return;
  const tr = document.createElement('tr');
  tr.className = 'topo-edge-row';
  tr.innerHTML = `
    <td><input type="text" class="tech-cam-input topo-from" value="cam_01"></td>
    <td><input type="text" class="tech-cam-input topo-to" value="cam_02"></td>
    <td>
      <div class="tech-num-box">
        <button class="tech-num-btn btn-step-sub" type="button">-</button>
        <input type="number" class="tech-num-input topo-exp" value="30" min="1">
        <button class="tech-num-btn btn-step-add" type="button">+</button>
      </div>
    </td>
    <td>
      <div class="tech-num-box">
        <button class="tech-num-btn btn-step-sub" type="button">-</button>
        <input type="number" class="tech-num-input topo-tol" value="15" min="1">
        <button class="tech-num-btn btn-step-add" type="button">+</button>
      </div>
    </td>
    <td><button class="clear-btn btn-delete-row" type="button" style="color: var(--danger);">🗑️ 删除</button></td>
  `;
  tbody.appendChild(tr);
  bindTopoTableButtons(tr);
}

export async function saveTopologyConfig() {
  const rows = document.querySelectorAll('.topo-edge-row');
  const graph = {};

  rows.forEach(tr => {
    const fromCam = tr.querySelector('.topo-from').value.trim();
    const toCam = tr.querySelector('.topo-to').value.trim();
    const expSec = parseFloat(tr.querySelector('.topo-exp').value) || 30;
    const tolSec = parseFloat(tr.querySelector('.topo-tol').value) || 15;

    if (!fromCam || !toCam) return;

    if (!graph[fromCam]) graph[fromCam] = [];
    graph[fromCam].push({
      next: toCam,
      expected_seconds: expSec,
      tolerance_seconds: tolSec
    });
  });

  try {
    const resp = await fetch('/api/topology', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graph)
    });
    const res = await resp.json();
    if (resp.ok && res.status === 'success') {
      showToast('摄像头拓扑结构与期望通行时间保存并即时生效成功！', 'success');
      openTopologyModal();
    } else {
      showToast('保存拓扑失败: ' + (res.error || `HTTP ${resp.status}`), 'error');
    }
  } catch(e) {
    showToast('保存拓扑发生网络错误: ' + e.message, 'error');
  }
}

export async function openHistoryExportModal() {
  const body = document.getElementById('modal-history-body');
  if (!body) return;
  body.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px;">
      <div style="font-size: 13px; color: var(--text-muted);">检索审计历史日志或一键导出报表：</div>
      <a href="${BASE}/api/alerts/export" class="action-btn active" style="text-decoration: none;" download>
        📥 导出 CSV 报表 (Excel)
      </a>
    </div>
    <div id="history-logs-container">
      <div class="empty-state">拉取日志数据中...</div>
    </div>
  `;
  openModal('history-modal');

  try {
    const res = await fetch('/api/alerts/history?limit=50').then(r => r.json());
    const alerts = res.alerts || [];
    const container = document.getElementById('history-logs-container');

    if (alerts.length === 0) {
      container.innerHTML = `<div class="empty-state">暂无历史告警日志记录</div>`;
      return;
    }

    container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 8px; max-height: 400px; overflow-y: auto;">
        ${alerts.map(a => {
          const tStr = new Date(a.timestamp * 1000).toLocaleString('zh-CN');
          const stageHtml = escapeHtml(a.stage || 'ALERT');
          const gidHtml = escapeHtml(a.global_id);
          const camHtml = escapeHtml(a.last_camera);
          const elapsedHtml = escapeHtml(a.elapsed_seconds);
          const riskHtml = escapeHtml(a.risk_level);
          return `
            <div class="calib-row" style="padding: 10px; flex-direction: column; align-items: flex-start; gap: 4px;">
              <div style="display: flex; justify-content: space-between; width: 100%;">
                <span style="font-weight: 700; color: ${a.alert_type === 'INTRUSION' ? '#ff0055' : 'var(--primary)'}; font-family: 'JetBrains Mono', monospace;">
                  ${a.alert_type === 'INTRUSION' ? '🚨 ROI 电子围栏闯入' : '⚠️ ' + stageHtml}
                </span>
                <span style="font-size: 10px; color: var(--text-dim);">${tStr}</span>
              </div>
              <div style="font-size: 11px; color: var(--text-muted);">
                目标: <b style="color: var(--text-main);">#${gidHtml}</b> |
                相机: <b style="color: var(--text-main);">${camHtml}</b> |
                用时: <b>${elapsedHtml}s</b> |
                风险等级: <b style="color: ${a.risk_level === 'HIGH' ? 'var(--danger)' : 'var(--warning)'}">${riskHtml}</b>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  } catch(e) {
    document.getElementById('history-logs-container').innerHTML = `<div class="empty-state">获取历史日志失败</div>`;
  }
}

export function openCamZoomModal(cam_id) {
  const title = document.getElementById('modal-alert-title');
  const body = document.getElementById('modal-alert-body');
  if (!title || !body) return;
  title.textContent = `摄像头高清放大监视器: ${cam_id.toUpperCase()}`;

  body.innerHTML = `
    <div class="cam-view" style="height: 380px; border-radius: 12px;">
      <img src="${BASE}/stream/${cam_id}" alt="${cam_id}">
    </div>
    <div style="display: flex; justify-content: flex-end; margin-top: 10px;">
      <button class="action-btn" id="btn-close-zoom-modal">关闭放大视角</button>
    </div>
  `;

  const btnClose = document.getElementById('btn-close-zoom-modal');
  if (btnClose) {
    btnClose.addEventListener('click', () => closeModal('alert-detail-modal'));
  }
  openModal('alert-detail-modal');
}
