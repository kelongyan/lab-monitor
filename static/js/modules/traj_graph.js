/**
 * traj_graph.js — 人员轨迹路线图（站点图）
 *
 * 把「某身份走过哪些摄像头」画成一张图：节点 = 物理站点，边 = 站点之间的通行路径。
 *
 * 为什么要合并「站点」而不是直接画 22 台相机
 * ------------------------------------------
 * config/topology.json 是一张巡检路线图，其中 `expected_seconds <= 10` 的边连接的是
 * **同一条走廊上两个朝向互拍的相机**（共位反向对），物理上是同一个位置。实测 7 组，
 * 加上 rnd_21/rnd_22 这类「desc 完全相同且拓扑并列」的同视点相机，22 台相机只剩 14 个
 * 物理站点。不合并的话节点数翻倍、边全部交叉，图必然糊成一团。
 *
 * ⚠️ 站点判据必须与 `scripts/derive_camera_map.py` 保持一致（后端坐标由它生成）。
 *    两处判据一旦漂移，会出现「节点画出来了但坐标指向别处」的错位。
 *
 * 坐标系
 * ------
 * 节点坐标来自 `/api/floorplan`（或 trajectory 响应里的 floorplan 包）的 `map_xy`：
 * 归一化 [0,1]，原点左上，x 向右、y 向下。渲染时乘画布尺寸。
 * `map_xy` 缺失时降级为「区域锚点 + 簇内水平排布」，保证空坐标也能出图。
 *
 * 约定
 * ----
 * - 纯函数（buildSiteGraph / layoutSites / routePath）不碰 DOM，便于 Node 用例覆盖。
 * - 渲染函数支持同一页面多实例（marker id 与动画类名都带实例号）。
 * - 所有文本插值过 escapeHtml —— 相机 desc 来自配置文件，仍按不可信内容处理。
 */

import { escapeHtml } from '../utils/formatter.js';

/* ------------------------------------------------------------------ *
 * 常量
 * ------------------------------------------------------------------ */

/** 判定共位反向相机对的时延上限（秒），与后端推导脚本一致。 */
export const COPLANAR_MAX_SECONDS = 10;

/** 画布逻辑尺寸（SVG viewBox 内部单位）。 */
const VIEW_W = 1000;
const VIEW_H = 720;

/** 站点节点尺寸（像素，viewBox 单位）。
 *  宽 158 是实测定的：最长站点名 `RND_04 + RND_05` 共 15 字符，12px 等宽
 *  在 viewBox 单位下约占 108，加左右各 16 内边距后 140，留出余量；
 *  再宽会挤压「中间走廊」3 站点的带内间距（带宽 0.50 / 3 站，间距已接近节点宽）。 */
export const NODE_W = 158;
export const NODE_H = 54;

/**
 * 区域带：`[x_start, x_end, y]`，区域内站点在该带中水平等距展开。
 *
 * 为什么是「带」而不是「单点锚点」：单点锚点只约束区域中心，西侧(2 站)/
 * 中间(3 站)/东侧(2 站) 三个区域的站点会全落在同一行上，7 个节点横向挤压重叠。
 * 带状定义把「西侧贴左、东侧贴右、中间居中」的物理关系直接表达出来。
 *
 * ⚠️ 必须与 scripts/derive_camera_map.py 的 REGION_BANDS 保持一致：
 *    后端按同一张表把坐标写进 config/camera_map.json，两处不一致会导致
 *    「有坐标时走真实坐标、无坐标时走降级布局」两套结果对不上。
 */
export const REGION_BANDS = {
  '北侧走廊': [0.35, 0.65, 0.08],
  '西侧走廊': [0.02, 0.38, 0.30],
  '东侧走廊': [0.62, 0.98, 0.30],
  '中间走廊': [0.25, 0.75, 0.50],
  '南侧走廊': [0.33, 0.67, 0.68],
  '高性能机房02通道': [0.02, 0.38, 0.88],
  '高性能机房04通道': [0.62, 0.98, 0.88],
};

/** 未识别区域的兜底带与名称。 */
const UNKNOWN_REGION = '未识别区域';
const UNKNOWN_BAND = [0.35, 0.65, 0.55];

/** 站点节点在画布上的归一化半宽（与后端脚本的 SITE_HALF_WIDTH 对应）。 */
const SITE_HALF_WIDTH = NODE_W / 2 / VIEW_W;

/** 折线圆角半径。 */
const CORNER_RADIUS = 8;

/* ------------------------------------------------------------------ *
 * 纯函数层：图构建
 * ------------------------------------------------------------------ */

/** 从中文 desc 解析区域（按关键词长度倒序匹配，避免「机房02通道」串到「机房04通道」）。 */
export function detectRegion(desc) {
  const text = String(desc || '');
  if (!text) return null;
  const keywords = Object.keys(REGION_BANDS).sort((a, b) => b.length - a.length);
  for (const keyword of keywords) {
    if (text.includes(keyword)) return keyword;
  }
  return null;
}

/**
 * 找出共位反向相机对（拓扑里 expected_seconds <= maxSeconds 的边）。
 * 返回 `[[camA, camB], ...]`，去重但不排序 —— 保留首次出现的顺序。
 */
export function findCoplanarPairs(topology, maxSeconds = COPLANAR_MAX_SECONDS) {
  const pairs = [];
  const seen = new Set();
  for (const camera of Object.keys(topology || {})) {
    for (const hop of topology[camera] || []) {
      const next = hop && hop.next;
      if (!next) continue;
      if (Number(hop.expected_seconds || 0) > maxSeconds) continue;
      const key = [camera, next].sort().join('\u0000');
      if (seen.has(key)) continue;
      seen.add(key);
      pairs.push([camera, next]);
    }
  }
  return pairs;
}

/** 返回 [直接前驱集合, 直接后继集合]。 */
function neighborsOf(topology, camera) {
  const successors = new Set();
  for (const hop of topology[camera] || []) {
    if (hop && hop.next) successors.add(hop.next);
  }
  const predecessors = new Set();
  for (const other of Object.keys(topology || {})) {
    for (const hop of topology[other] || []) {
      if (hop && hop.next === camera) predecessors.add(other);
    }
  }
  return [predecessors, successors];
}

/** 组内相机是否在拓扑上并列（共享直接前驱或后继）。 */
function isParallelGroup(topology, group) {
  if (group.length < 2) return false;
  for (let i = 0; i < group.length; i += 1) {
    const [leftPred, leftSucc] = neighborsOf(topology, group[i]);
    for (let j = i + 1; j < group.length; j += 1) {
      const [rightPred, rightSucc] = neighborsOf(topology, group[j]);
      for (const node of leftPred) if (rightPred.has(node)) return true;
      for (const node of leftSucc) if (rightSucc.has(node)) return true;
    }
  }
  return false;
}

/**
 * desc 完全相同**且在拓扑上并列**的相机视为同一视点（rnd_21 / rnd_22）。
 *
 * 必须同时要求并列：`reg_08` 与 `rnd_10` 的 desc 也完全相同（都是「L2南侧走廊西西向东」），
 * 但分属常规环与随机链两条互不相连的路线 —— 只能说明命名粒度到走廊级，
 * 不能据此认定同一点位。
 */
export function findDuplicateDescGroups(cameraMap, topology, exclude = new Set()) {
  const byDesc = new Map();
  for (const camera of Object.keys(cameraMap || {})) {
    if (exclude.has(camera)) continue;
    const desc = (cameraMap[camera] || {}).desc || '';
    if (!desc) continue;
    if (!byDesc.has(desc)) byDesc.set(desc, []);
    byDesc.get(desc).push(camera);
  }
  const groups = [];
  for (const cameras of [...byDesc.values()]) {
    if (cameras.length < 2) continue;
    const sorted = [...cameras].sort();
    if (sorted.length === 2) {
      if (isParallelGroup(topology, sorted)) groups.push(sorted);
      continue;
    }
    for (let i = 0; i < sorted.length; i += 1) {
      for (let j = i + 1; j < sorted.length; j += 1) {
        if (isParallelGroup(topology, [sorted[i], sorted[j]])) groups.push([sorted[i], sorted[j]]);
      }
    }
  }
  return groups;
}

/**
 * 构建站点图。
 *
 * @param {object} floorplan `/api/floorplan` 响应（或 trajectory 里的 floorplan 包）
 * @param {object} topology  `/api/topology` 响应：`{cam: [{next, expected_seconds}]}`
 * @param {Array}  groups    trajectory 的 `groups`（**不是**原始 segments）
 * @returns {{sites: Array, edges: Array, meta: object}}
 */
export function buildSiteGraph(floorplan, topology, groups) {
  const cameras = (floorplan && floorplan.cameras) || {};
  const topo = topology || {};
  const groupList = Array.isArray(groups) ? groups : [];

  // ---- 1. 相机 → 站点 ----
  const siteOfCamera = new Map();
  const sites = [];

  const addSite = (cameraList) => {
    const id = [...cameraList].sort().join('+');
    const site = {
      id,
      cameras: [...cameraList].sort(),
      region: UNKNOWN_REGION,
      desc: '',
      coord: null,
      visited: false,
      order: null,
      visits: 0,
      dwellS: 0,
      frames: 0,
    };
    site.cameras.forEach((camera) => siteOfCamera.set(camera, site));
    sites.push(site);
    return site;
  };

  const pairs = findCoplanarPairs(topo);
  const paired = new Set();
  pairs.forEach(([a, b]) => {
    paired.add(a);
    paired.add(b);
    addSite([a, b]);
  });

  const dupGroups = findDuplicateDescGroups(cameras, topo, paired);
  dupGroups.forEach((group) => {
    group.forEach((camera) => paired.add(camera));
    addSite(group);
  });

  Object.keys(cameras).forEach((camera) => {
    if (!paired.has(camera)) addSite([camera]);
  });

  // ---- 2. 站点归属区域与坐标 ----
  let mappedCount = 0;
  sites.forEach((site) => {
    const regionVotes = new Map();
    let coord = null;
    let desc = '';
    site.cameras.forEach((camera) => {
      const meta = cameras[camera] || {};
      if (!desc && meta.desc) desc = meta.desc;
      const region = detectRegion(meta.desc) || UNKNOWN_REGION;
      regionVotes.set(region, (regionVotes.get(region) || 0) + 1);
      if (!coord && Array.isArray(meta.map_xy) && meta.map_xy.length === 2) {
        coord = [Number(meta.map_xy[0]), Number(meta.map_xy[1])];
      }
    });
    let best = UNKNOWN_REGION;
    let bestVotes = -1;
    for (const [region, votes] of regionVotes) {
      if (votes > bestVotes) {
        best = region;
        bestVotes = votes;
      }
    }
    site.region = best;
    site.desc = desc;
    site.coord = coord;
    if (coord) mappedCount += 1;
  });

  // ---- 3. 拜访状态：吃 groups（已压好的视图组），不吃原始 segments ----
  let step = 0;
  groupList.forEach((group) => {
    const touched = new Set();
    (group.cameras || []).forEach((camera) => {
      const site = siteOfCamera.get(camera);
      if (site) touched.add(site);
    });
    touched.forEach((site) => {
      if (site.order === null) {
        step += 1;
        site.order = step;
      }
      site.visited = true;
      site.visits += 1;
      site.dwellS += Number(group.duration_s || 0);
      site.frames += Number(group.frames || 0);
    });
  });

  // 最后一步到达的站点标记为「当前所在」
  let lastSiteId = null;
  for (const group of groupList) {
    (group.cameras || []).forEach((camera) => {
      const site = siteOfCamera.get(camera);
      if (site) lastSiteId = site.id;
    });
  }
  sites.forEach((site) => {
    if (!site.visited) site.state = 'unvisited';
    else if (site.id === lastSiteId) site.state = 'current';
    else site.state = 'visited';
  });

  // ---- 4. 边：丢弃共位内部边，其余映射到站点对 ----
  const edgeMap = new Map();
  for (const camera of Object.keys(topo)) {
    for (const hop of topo[camera] || []) {
      const next = hop && hop.next;
      if (!next) continue;
      const seconds = Number(hop.expected_seconds || 0);
      if (seconds <= COPLANAR_MAX_SECONDS) continue;
      const fromSite = siteOfCamera.get(camera);
      const toSite = siteOfCamera.get(next);
      if (!fromSite || !toSite || fromSite === toSite) continue;
      const key = [fromSite.id, toSite.id].sort().join('\u0000');
      const existing = edgeMap.get(key);
      if (existing) {
        existing.seconds = Math.min(existing.seconds, seconds);
        continue;
      }
      edgeMap.set(key, {
        id: key,
        from: fromSite.id,
        to: toSite.id,
        seconds,
        traversed: false,
        flow: null,
      });
    }
  }

  // ---- 5. 动线：把站点访问序列的相邻对标记为「走过」 ----
  const sequence = groupList
    .map((group) => {
      const touched = [];
      (group.cameras || []).forEach((camera) => {
        const site = siteOfCamera.get(camera);
        if (site && !touched.includes(site.id)) touched.push(site.id);
      });
      return touched;
    })
    .flat()
    .filter((id, index, arr) => index === 0 || arr[index - 1] !== id);

  let flowStep = 0;
  for (let i = 1; i < sequence.length; i += 1) {
    const key = [sequence[i - 1], sequence[i]].sort().join('\u0000');
    const edge = edgeMap.get(key);
    // 只标记拓扑里真实存在的边：动线里出现「拓扑上不相邻的跳变」时不能凭空虚连
    if (edge && !edge.traversed) {
      flowStep += 1;
      edge.traversed = true;
      edge.flow = flowStep;
    }
  }

  const siteIndex = new Map(sites.map((site) => [site.id, site]));
  const edges = [...edgeMap.values()].map((edge) => ({
    ...edge,
    fromRegion: (siteIndex.get(edge.from) || {}).region,
    toRegion: (siteIndex.get(edge.to) || {}).region,
  }));

  return {
    sites,
    edges,
    meta: {
      siteCount: sites.length,
      cameraCount: Object.keys(cameras).length,
      mappedCount,
      hasCoordinates: mappedCount > 0,
      visitedCount: sites.filter((site) => site.visited).length,
      traversedEdgeCount: edges.filter((edge) => edge.traversed).length,
      sequence,
    },
  };
}

/**
 * 计算站点像素坐标（viewBox 单位）。
 *
 * 有 `map_xy` 的站点直接用坐标；缺失的降级到「区域锚点 + 簇内水平排布」，
 * 保证 `map_xy` 全为 null 时也能出一张结构正确的图。
 */
export function layoutSites(sites, options = {}) {
  const viewW = options.viewW || VIEW_W;
  const viewH = options.viewH || VIEW_H;
  const fallback = [];

  const positioned = sites.map((site) => {
    if (site.coord) {
      return {
        ...site,
        x: Math.min(viewW - NODE_W / 2, Math.max(NODE_W / 2, site.coord[0] * viewW)),
        y: Math.min(viewH - NODE_H / 2, Math.max(NODE_H / 2, site.coord[1] * viewH)),
        positioned: true,
      };
    }
    fallback.push(site);
    return { ...site, x: 0, y: 0, positioned: false };
  });

  if (fallback.length) {
    const byRegion = new Map();
    fallback.forEach((site) => {
      if (!byRegion.has(site.region)) byRegion.set(site.region, []);
      byRegion.get(site.region).push(site);
    });
    byRegion.forEach((members, region) => {
      members.sort((a, b) => String(a.id).localeCompare(String(b.id)));
      const band = REGION_BANDS[region] || UNKNOWN_BAND;
      const left = band[0] + SITE_HALF_WIDTH;
      const right = band[1] - SITE_HALF_WIDTH;
      const span = Math.max(0, right - left);
      const center = span > 0 ? (left + right) / 2 : (band[0] + band[1]) / 2;
      const count = members.length;
      members.forEach((site, index) => {
        const target = positioned.find((item) => item.id === site.id);
        if (!target) return;
        const offset = count === 1 ? 0 : -span / 2 + index * (span / (count - 1));
        target.x = Math.min(
          viewW - NODE_W / 2,
          Math.max(NODE_W / 2, (center + offset) * viewW),
        );
        target.y = Math.min(
          viewH - NODE_H / 2,
          Math.max(NODE_H / 2, band[2] * viewH),
        );
      });
    });
  }

  return positioned;
}

/* ------------------------------------------------------------------ *
 * 纯函数层：正交路由
 * ------------------------------------------------------------------ */

/** 求从矩形中心朝目标方向射出时，与矩形边界的交点。 */
export function rectBorderPoint(cx, cy, halfW, halfH, tx, ty) {
  const dx = tx - cx;
  const dy = ty - cy;
  if (dx === 0 && dy === 0) return [cx, cy];
  const scaleX = dx === 0 ? Infinity : halfW / Math.abs(dx);
  const scaleY = dy === 0 ? Infinity : halfH / Math.abs(dy);
  const scale = Math.min(scaleX, scaleY);
  return [cx + dx * scale, cy + dy * scale];
}

/**
 * 把折线点序列转成带圆角的 SVG path。
 * 中间每个拐点用二次贝塞尔倒角，避免 90° 硬拐角在细线上显得毛刺。
 */
export function roundedPath(points, radius = CORNER_RADIUS) {
  if (!Array.isArray(points) || points.length < 2) return '';
  if (points.length === 2) {
    return `M ${points[0][0]} ${points[0][1]} L ${points[1][0]} ${points[1][1]}`;
  }
  let d = `M ${points[0][0]} ${points[0][1]}`;
  for (let i = 1; i < points.length - 1; i += 1) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    const [x2, y2] = points[i + 1];
    const len1 = Math.hypot(x1 - x0, y1 - y0);
    const len2 = Math.hypot(x2 - x1, y2 - y1);
    if (!len1 || !len2) continue;
    const r = Math.min(radius, len1 / 2, len2 / 2);
    const ax = x1 + ((x0 - x1) / len1) * r;
    const ay = y1 + ((y0 - y1) / len1) * r;
    const bx = x1 + ((x2 - x1) / len2) * r;
    const by = y1 + ((y2 - y1) / len2) * r;
    d += ` L ${ax} ${ay} Q ${x1} ${y1} ${bx} ${by}`;
  }
  const last = points[points.length - 1];
  d += ` L ${last[0]} ${last[1]}`;
  return d;
}

/**
 * 生成两个站点之间的正交（曼哈顿）路由。
 *
 * 为什么不用直线：15 个节点规模下斜线会大面积交叉。正交路由让边贴着网格走，
 * 拐角数量可控，可读性高得多。
 *
 * 返回 `{d, label: [x, y]}`，label 是时延标注的落点。
 */
export function routeBetween(from, to, options = {}) {
  const halfW = NODE_W / 2;
  const halfH = NODE_H / 2;
  const p1 = rectBorderPoint(from.x, from.y, halfW, halfH, to.x, to.y);
  const p2 = rectBorderPoint(to.x, to.y, halfW, halfH, from.x, from.y);

  const dx = Math.abs(p2[0] - p1[0]);
  const dy = Math.abs(p2[1] - p1[1]);

  // 垂直对齐或水平对齐：直接一条直线（最常见，节点按区域成行成列）
  if (dx < 1 || dy < 1) {
    return {
      d: roundedPath([p1, p2], options.radius),
      label: [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 - 10],
    };
  }

  // 否则走 Z 形：主轴取跨度大的那个方向，中途横切一次
  if (dx >= dy) {
    const midX = (p1[0] + p2[0]) / 2;
    return {
      d: roundedPath([p1, [midX, p1[1]], [midX, p2[1]], p2], options.radius),
      label: [midX, (p1[1] + p2[1]) / 2],
    };
  }
  const midY = (p1[1] + p2[1]) / 2;
  return {
    d: roundedPath([p1, [p1[0], midY], [p2[0], midY], p2], options.radius),
    label: [(p1[0] + p2[0]) / 2, midY - 10],
  };
}

/* ------------------------------------------------------------------ *
 * 渲染层
 * ------------------------------------------------------------------ */

let instanceSeq = 0;

/** 站点副标题：区域 + 序号（已访问时带次序）。 */
function siteCaption(site) {
  const region = site.region === UNKNOWN_REGION ? '' : site.region;
  if (site.order === null) return region || '未访问';
  return region ? `${site.order} · ${region}` : `${site.order}`;
}

/**
 * 站点主标题：站点内并列的相机名。
 *
 * 共位反向相机的编号通常成对（rnd_04 / rnd_05），完整写 `RND_04 + RND_05`
 * 有 15 个字符，会被迫降字号到 10px —— 未访问站点的文字本来就淡，再缩小就难读。
 * 同前缀时压成 `RND_04/05`（10 字符），即可统一用 12px。
 * 精确的完整列表在悬浮提示里给出（见 buildTooltipHtml）。
 */
function siteTitle(site) {
  const names = site.cameras.map((camera) => String(camera).toUpperCase());
  if (names.length === 1) return names[0];
  const parts = names.map((name) => name.match(/^(.*?)(\d+)$/));
  if (parts.every(Boolean) && parts.every((part) => part[1] === parts[0][1])) {
    return `${parts[0][1]}${parts.map((part) => part[2].padStart(2, '0')).join('/')}`;
  }
  return names.join(' + ');
}

/** 驻留时长人类可读格式。 */
function formatDwell(seconds) {
  const value = Number(seconds) || 0;
  if (value < 60) return `${Math.round(value)}s`;
  if (value < 3600) return `${Math.floor(value / 60)}m${Math.round(value % 60)}s`;
  return `${(value / 3600).toFixed(1)}h`;
}

function buildTooltipHtml(site) {
  const rows = [
    ['相机', escapeHtml(site.cameras.map((c) => String(c).toUpperCase()).join(' + '))],
    ['位置', escapeHtml(site.desc || site.region || '—')],
    ['状态', site.visited ? (site.state === 'current' ? '当前所在' : `第 ${site.order} 站`) : '未经过'],
  ];
  if (site.visited) {
    rows.push(['驻留', escapeHtml(formatDwell(site.dwellS))]);
    rows.push(['命中', `${site.visits} 段 / ${site.frames} 帧`]);
  }
  return rows
    .map(([label, value]) => `<div class="tg-tip-row"><span>${label}</span><b>${value}</b></div>`)
    .join('');
}

/**
 * 渲染轨迹路线图。
 *
 * @param {HTMLElement} container 挂载容器
 * @param {{sites: Array, edges: Array, meta: object}} graph buildSiteGraph 的产物
 * @param {object} options
 *   - `loop`: trajectory 的 `loop` 字段（用于循环折叠提示）
 *   - `onNodeClick(site)`: 节点点击回调（时间轴联动）
 *   - `enablePlay`: 是否显示播放控制条
 * @returns {object|null} 控制器（含 destroy / play / pause / reset），容器缺失时返回 null
 */
export function renderTrajGraph(container, graph, options = {}) {
  if (!container) return null;
  if (!graph || !graph.sites || graph.sites.length === 0) {
    container.innerHTML = '<div class="empty-state" style="padding: 10px;">无可绘制的站点数据</div>';
    return null;
  }

  destroyTrajGraph(container);

  const instanceId = (instanceSeq += 1);
  const arrowId = `tg-arrow-${instanceId}`;
  const sites = layoutSites(graph.sites, options);
  const index = new Map(sites.map((site) => [site.id, site]));

  const renderedEdges = graph.edges
    .map((edge) => {
      const from = index.get(edge.from);
      const to = index.get(edge.to);
      if (!from || !to) return null;
      const route = routeBetween(from, to, options);
      return { ...edge, from, to, ...route };
    })
    .filter(Boolean)
    // 未走过的边先画，避免压在动线之上
    .sort((a, b) => Number(a.traversed) - Number(b.traversed));

  const edgeSvg = renderedEdges
    .map((edge) => {
      const cls = ['tg-edge', edge.traversed ? 'tg-traversed' : 'tg-idle'].join(' ');
      // 时延只标注长位移边（>60s）：短边密密麻麻全是数字反而看不清
      const label = edge.seconds > 60
        ? `<text class="tg-edge-label" x="${edge.label[0].toFixed(0)}" y="${edge.label[1].toFixed(0)}" text-anchor="middle">${Math.round(edge.seconds)}s</text>`
        : '';
      return `<path class="${cls}" d="${edge.d}" marker-end="url(#${arrowId})" data-edge="${escapeHtml(edge.id)}"></path>${label}`;
    })
    .join('');

  const nodeSvg = sites
    .map((site) => {
      const x = site.x - NODE_W / 2;
      const y = site.y - NODE_H / 2;
      const cls = ['tg-node', `tg-${site.state}`].join(' ');
      const title = siteTitle(site);
      // 相机名过长时缩字号，避免溢出节点（节点宽 158，12px 等宽约容 13 字符后收窄）
      const camFont = title.length > 13 ? 10 : 12;
      return `<g class="${cls}" data-site="${escapeHtml(site.id)}" data-order="${site.order === null ? '' : site.order}" tabindex="0" role="button" aria-label="${escapeHtml(`站点 ${site.id}`)}">
  <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${NODE_W}" height="${NODE_H}" rx="10" stroke-width="${site.visited ? 2 : 1}"></rect>
  <text class="tg-node-region" x="${(x + 16).toFixed(1)}" y="${(y + 19).toFixed(1)}">${escapeHtml(siteCaption(site))}</text>
  <text class="tg-node-cams" x="${(x + 16).toFixed(1)}" y="${(y + 37).toFixed(1)}" style="font-size:${camFont}px">${escapeHtml(title)}</text>
</g>`;
    })
    .join('');

  const loop = options.loop || null;
  const loopNote = loop && loop.detected
    ? `<div class="tg-loop-note">素材循环播放，已折叠 ${loop.loops && loop.loops > 1 ? loop.loops : ''} 轮（策略：${escapeHtml(loop.method || 'period')}）—— 不折叠会得到「来回跑几百次」的假象</div>`
    : '';

  const stat = `共 ${graph.meta.siteCount} 个站点（${graph.meta.cameraCount} 台相机）· 该身份经过 ${graph.meta.visitedCount} 站 · 走过 ${graph.meta.traversedEdgeCount} 段路径`;

  container.innerHTML = `
    <div class="tg-head">
      <div class="tg-title">轨迹路线图</div>
      <div class="tg-stat">${escapeHtml(stat)}</div>
    </div>
    ${loopNote}
    <div class="tg-stage">
      <svg class="tg-svg" viewBox="0 0 ${VIEW_W} ${VIEW_H}" preserveAspectRatio="xMidYMid meet" role="img"
           aria-label="人员跨站点通行路线图">
        <defs>
          <marker id="${arrowId}" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5"
                  stroke-linecap="round" stroke-linejoin="round"></path>
          </marker>
        </defs>
        ${edgeSvg}
        ${nodeSvg}
      </svg>
      <div class="tg-tip" hidden></div>
    </div>
    ${options.enablePlay === false ? '' : `
    <div class="tg-controls">
      <button class="action-btn tg-btn" data-tg-action="play" type="button">▶ 播放动线</button>
      <button class="action-btn tg-btn" data-tg-action="pause" type="button">⏸ 暂停</button>
      <button class="action-btn tg-btn" data-tg-action="reset" type="button">⏮ 重置</button>
      <span class="tg-progress" data-tg-progress>顺序高亮：全部</span>
    </div>`}
    ${graph.meta.hasCoordinates ? '' : '<div class="tg-hint">节点坐标缺失，已按区域锚点降级排布；补全 config/camera_map.json 的 map_xy 后即为真实点位。</div>'}
  `;

  return bindGraphInteractions(container, { graph, sites, arrowId, options });
}

/* ------------------------------------------------------------------ *
 * 交互与动线
 * ------------------------------------------------------------------ */

function bindGraphInteractions(container, ctx) {
  const { graph, sites, options } = ctx;
  const svg = container.querySelector('.tg-svg');
  const tip = container.querySelector('.tg-tip');
  const progress = container.querySelector('[data-tg-progress]');
  const nodeEls = new Map();
  container.querySelectorAll('.tg-node').forEach((el) => {
    nodeEls.set(el.getAttribute('data-site'), el);
  });
  const edgeEls = [...container.querySelectorAll('.tg-edge.tg-traversed')];

  const ordered = [...sites]
    .filter((site) => site.order !== null)
    .sort((a, b) => a.order - b.order);

  let timer = null;
  let cursor = 0;

  const timers = [];
  const clearTimers = () => {
    while (timers.length) {
      clearTimeout(timers.pop());
    }
  };

  const applyStaticState = () => {
    clearTimers();
    cursor = ordered.length;
    ordered.forEach((site) => nodeEls.get(site.id)?.classList.remove('tg-dim'));
    if (progress) progress.textContent = '顺序高亮：全部';
    // 已走过的边恢复为「动线」样式
    edgeEls.forEach((el) => el.classList.add('tg-flowing'));
  };

  const pause = () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  };

  const reset = () => {
    pause();
    clearTimers();
    cursor = 0;
    nodeEls.forEach((el) => el.classList.add('tg-dim'));
    if (progress) progress.textContent = '顺序高亮：0 / ' + ordered.length;
  };

  const renderCursor = () => {
    ordered.forEach((site, i) => {
      const el = nodeEls.get(site.id);
      if (!el) return;
      el.classList.toggle('tg-dim', i >= cursor);
    });
    if (progress) progress.textContent = `顺序高亮：${cursor} / ${ordered.length}`;
  };

  const play = () => {
    pause();
    reset();
    if (!ordered.length) return;
    let i = 0;
    timer = setInterval(() => {
      i += 1;
      cursor = i;
      renderCursor();
      if (i >= ordered.length) pause();
    }, 450);
    renderCursor();
  };

  const onClick = (event) => {
    const nodeEl = event.target.closest('.tg-node');
    if (!nodeEl || !svg || !svg.contains(nodeEl)) return;
    const site = sites.find((item) => item.id === nodeEl.getAttribute('data-site'));
    if (!site || typeof options.onNodeClick !== 'function') return;
    options.onNodeClick(site);
  };

  const onMove = (event) => {
    if (!tip) return;
    const nodeEl = event.target.closest('.tg-node');
    if (!nodeEl || !svg || !svg.contains(nodeEl)) {
      tip.hidden = true;
      return;
    }
    const site = sites.find((item) => item.id === nodeEl.getAttribute('data-site'));
    if (!site) return;
    const box = container.getBoundingClientRect();
    tip.innerHTML = buildTooltipHtml(site);
    tip.hidden = false;
    const left = Math.min(event.clientX - box.left + 12, box.width - 190);
    const top = Math.max(4, event.clientY - box.top - 8);
    tip.style.left = `${Math.max(4, left)}px`;
    tip.style.top = `${top}px`;
  };

  const onLeave = () => {
    if (tip) tip.hidden = true;
  };

  const onKey = (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const nodeEl = event.target.closest ? event.target.closest('.tg-node') : null;
    if (!nodeEl) return;
    event.preventDefault();
    const site = sites.find((item) => item.id === nodeEl.getAttribute('data-site'));
    if (site && typeof options.onNodeClick === 'function') options.onNodeClick(site);
  };

  container.addEventListener('click', onClick);
  container.addEventListener('mousemove', onMove);
  container.addEventListener('mouseleave', onLeave);
  container.addEventListener('keydown', onKey);
  container.querySelectorAll('[data-tg-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.getAttribute('data-tg-action');
      if (action === 'play') play();
      else if (action === 'pause') pause();
      else if (action === 'reset') reset();
    });
  });

  applyStaticState();

  const controller = {
    sites,
    play,
    pause,
    reset,
    /** 与时间轴联动：外部高亮到第 n 站 */
    highlight(order) {
      pause();
      cursor = Math.max(0, Math.min(ordered.length, Number(order) || 0));
      renderCursor();
    },
    destroy() {
      pause();
      clearTimers();
      container.removeEventListener('click', onClick);
      container.removeEventListener('mousemove', onMove);
      container.removeEventListener('mouseleave', onLeave);
      container.removeEventListener('keydown', onKey);
    },
  };
  container.__tgController = controller;
  return controller;
}

/** 清理某容器上的轨迹图（关闭弹窗前必须调用，否则定时器与监听会残留）。 */
export function destroyTrajGraph(container) {
  if (!container) return;
  const controller = container.__tgController;
  if (controller && typeof controller.destroy === 'function') controller.destroy();
  container.__tgController = null;
  container.innerHTML = '';
}

/** 暴露给测试与调试：当前画布尺寸常量。 */
export const VIEW_SIZE = [VIEW_W, VIEW_H];
