/**
 * 前端轨迹路线图测试（static/js/modules/traj_graph.js）
 *
 * 纯 Node 环境跑：纯函数层不需要 DOM；渲染层用最小 container 桩件驱动，
 * 断言「站点合并、区域识别、访问计数、正交路由、坐标边界、空数据降级」。
 *
 * 数据源直接用仓库里的真实 config/topology.json + config/camera_map.json ——
 * 这样一旦配置改动与代码里的站点判据发生漂移，本用例会立刻失败
 * （判据漂移的后果是「节点画出来了但坐标指向别处」，很难靠肉眼发现）。
 *
 * 运行：node tests/test_traj_graph_frontend.mjs
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  buildSiteGraph,
  detectRegion,
  destroyTrajGraph,
  findCoplanarPairs,
  findDuplicateDescGroups,
  layoutSites,
  rectBorderPoint,
  renderTrajGraph,
  roundedPath,
  routeBetween,
  NODE_H,
  NODE_W,
  VIEW_SIZE,
} from '../static/js/modules/traj_graph.js';

const configDir = new URL('../config/', import.meta.url);
const topology = JSON.parse(readFileSync(new URL('topology.json', configDir), 'utf-8'));
const cameraMap = JSON.parse(readFileSync(new URL('camera_map.json', configDir), 'utf-8'));

/** 把 config/camera_map.json 转成 `/api/floorplan` 响应里 cameras 字段的形状。 */
const floorplan = { cameras: cameraMap };

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`  PASS  ${name}`);
}

/** 渲染层最小桩件：只实现 renderTrajGraph / bindGraphInteractions 用到的那几个方法。 */
class FakeContainer {
  constructor() {
    this.innerHTML = '';
    this.listeners = {};
    this.__tgController = null;
  }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== fn);
  }
  getBoundingClientRect() { return { left: 0, top: 0, width: 1000, height: 720 }; }
}

/* ------------------------------------------------------------------ */
/* 区域识别                                                            */
/* ------------------------------------------------------------------ */

console.log('== 区域识别（desc 解析）==');

check('东侧走廊', () => {
  assert.equal(detectRegion('L2东侧走廊南北向南'), '东侧走廊');
});
check('机房 02 与 04 通道不串区（按关键词长度倒序匹配）', () => {
  assert.equal(detectRegion('L2高性能机房02通道南西向东'), '高性能机房02通道');
  assert.equal(detectRegion('L2高性能机房04通道东南向北'), '高性能机房04通道');
});
check('空 desc 返回 null', () => {
  assert.equal(detectRegion(''), null);
  assert.equal(detectRegion(null), null);
});
check('全部 22 路相机都能识别出区域（不允许出现未识别）', () => {
  const unknown = Object.keys(cameraMap).filter((cam) => !detectRegion(cameraMap[cam].desc));
  assert.deepEqual(unknown, []);
});

/* ------------------------------------------------------------------ */
/* 站点合并                                                            */
/* ------------------------------------------------------------------ */

console.log('\n== 站点合并判据 ==');

check('共位反向相机对 7 组（拓扑 5s 短边）', () => {
  const pairs = findCoplanarPairs(topology);
  assert.equal(pairs.length, 7);
  const flat = pairs.flat().sort();
  assert.deepEqual(flat, [
    'reg_01', 'reg_02', 'reg_05', 'reg_06',
    'rnd_01', 'rnd_02', 'rnd_04', 'rnd_05',
    'rnd_06', 'rnd_07', 'rnd_11', 'rnd_12',
    'rnd_17', 'rnd_18',
  ].sort());
});

check('desc 相同且拓扑并列 → 合并（rnd_21 + rnd_22）', () => {
  const exclude = new Set(findCoplanarPairs(topology).flat());
  const groups = findDuplicateDescGroups(floorplan.cameras, topology, exclude);
  assert.deepEqual(groups, [['rnd_21', 'rnd_22']]);
});

check('desc 相同但拓扑不相邻 → 不合并（reg_08 与 rnd_10 都是「L2南侧走廊西西向东」）', () => {
  assert.equal(cameraMap.reg_08.desc, cameraMap.rnd_10.desc);
  const exclude = new Set(findCoplanarPairs(topology).flat());
  const flattened = findDuplicateDescGroups(floorplan.cameras, topology, exclude).flat();
  assert.ok(!flattened.includes('reg_08'), 'reg_08 不应被合并');
  assert.ok(!flattened.includes('rnd_10'), 'rnd_10 不应被合并');
});

check('22 台相机合并为 14 个站点', () => {
  const graph = buildSiteGraph(floorplan, topology, []);
  assert.equal(graph.meta.cameraCount, 22);
  assert.equal(graph.meta.siteCount, 14);
});

check('每个相机恰好归属一个站点', () => {
  const graph = buildSiteGraph(floorplan, topology, []);
  const seen = graph.sites.flatMap((site) => site.cameras);
  assert.equal(seen.length, 22);
  assert.equal(new Set(seen).size, 22);
});

check('共位内部边（5s）不进入边集合，且不存在自环边', () => {
  const graph = buildSiteGraph(floorplan, topology, []);
  graph.edges.forEach((edge) => {
    assert.notEqual(edge.from, edge.to, `边 ${edge.id} 是站点自环，说明共位边没被丢弃`);
    assert.ok(edge.seconds > 10, `边 ${edge.id} 时延 ${edge.seconds}s 属于共位短边`);
  });
});

/* ------------------------------------------------------------------ */
/* 访问状态与动线                                                      */
/* ------------------------------------------------------------------ */

console.log('\n== 访问状态与动线 ==');

// rnd_11 与 rnd_12 是共位对 → 同属一个站点；两条 group 分别命中它，应累加 visits
const sampleGroups = [
  { cameras: ['rnd_10'], enter: 1000, exit: 1060, duration_s: 60, frames: 150 },
  { cameras: ['rnd_11'], enter: 1080, exit: 1100, duration_s: 20, frames: 50 },
  { cameras: ['rnd_12'], enter: 1102, exit: 1110, duration_s: 8, frames: 20 },
  { cameras: ['rnd_06'], enter: 1120, exit: 1140, duration_s: 20, frames: 50 },
];

check('访问序号按首次到达递增', () => {
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  const byId = new Map(graph.sites.map((site) => [site.id, site]));
  assert.equal(byId.get('rnd_10').order, 1);
  assert.equal(byId.get('rnd_11+rnd_12').order, 2);
  assert.equal(byId.get('rnd_06+rnd_07').order, 3);
});

check('同一站点的多次访问累加 visits / 驻留 / 帧数', () => {
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  const site = graph.sites.find((item) => item.id === 'rnd_11+rnd_12');
  assert.equal(site.visited, true);
  assert.equal(site.visits, 2);
  assert.equal(site.dwellS, 28);
  assert.equal(site.frames, 70);
});

check('最后到达的站点标记为 current，其余为 visited', () => {
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  const current = graph.sites.filter((site) => site.state === 'current');
  assert.equal(current.length, 1);
  assert.equal(current[0].id, 'rnd_06+rnd_07');
});

check('未访问站点保持 unvisited 且 order 为 null', () => {
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  const untouched = graph.sites.find((item) => item.id === 'rnd_19');
  assert.equal(untouched.visited, false);
  assert.equal(untouched.order, null);
  assert.equal(untouched.state, 'unvisited');
});

check('动线只标记拓扑里真实存在的边（不相邻的跳变不虚连）', () => {
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  graph.edges.filter((edge) => edge.traversed).forEach((edge) => {
    const pairs = new Set();
    for (const camera of Object.keys(topology)) {
      for (const hop of topology[camera] || []) {
        pairs.add([camera, hop.next].sort().join('|'));
      }
    }
    const fromCams = graph.sites.find((s) => s.id === edge.from).cameras;
    const toCams = graph.sites.find((s) => s.id === edge.to).cameras;
    const linked = fromCams.some((a) => toCams.some((b) => pairs.has([a, b].sort().join('|'))));
    assert.ok(linked, `动线边 ${edge.id} 在拓扑里不存在`);
  });
});

check('无 groups 时全部站点为未访问，且不抛异常', () => {
  const graph = buildSiteGraph(floorplan, topology, null);
  assert.equal(graph.meta.visitedCount, 0);
  assert.equal(graph.meta.traversedEdgeCount, 0);
  assert.equal(graph.meta.siteCount, 14);
});

/* ------------------------------------------------------------------ */
/* 布局与正交路由                                                      */
/* ------------------------------------------------------------------ */

console.log('\n== 布局与正交路由 ==');

check('有 map_xy 时按坐标定位', () => {
  const graph = buildSiteGraph(floorplan, topology, []);
  const sites = layoutSites(graph.sites);
  const target = sites.find((site) => site.id === 'rnd_10');
  assert.equal(target.positioned, true);
  assert.equal(Math.round(target.x), Math.round(cameraMap.rnd_10.map_xy[0] * VIEW_SIZE[0]));
  assert.equal(Math.round(target.y), Math.round(cameraMap.rnd_10.map_xy[1] * VIEW_SIZE[1]));
});

check('坐标全缺失时降级到区域锚点，且节点不越出画布', () => {
  const bare = { cameras: Object.fromEntries(Object.keys(cameraMap).map((cam) => [cam, { desc: cameraMap[cam].desc, map_xy: null }])) };
  const graph = buildSiteGraph(bare, topology, []);
  const sites = layoutSites(graph.sites);
  assert.equal(graph.meta.hasCoordinates, false);
  sites.forEach((site) => {
    assert.ok(site.positioned === false, `${site.id} 不应被视为已定位`);
    assert.ok(site.x - NODE_W / 2 >= 0 && site.x + NODE_W / 2 <= VIEW_SIZE[0], `${site.id} 横向越界`);
    assert.ok(site.y - NODE_H / 2 >= 0 && site.y + NODE_H / 2 <= VIEW_SIZE[1], `${site.id} 纵向越界`);
  });
});

check('坐标越界（手工标注写错）时被夹回画布内', () => {
  const weird = { cameras: { cam_x: { desc: 'L2北侧走廊东东向西', map_xy: [-3, 9] } } };
  const graph = buildSiteGraph(weird, {}, []);
  const site = layoutSites(graph.sites)[0];
  assert.equal(site.x, NODE_W / 2);
  assert.equal(site.y, VIEW_SIZE[1] - NODE_H / 2);
});

check('rectBorderPoint 在水平方向上取左右边交点', () => {
  const point = rectBorderPoint(100, 100, 75, 27, 300, 100);
  assert.deepEqual(point, [175, 100]);
});

check('rectBorderPoint 在垂直方向上取上下边交点', () => {
  const point = rectBorderPoint(100, 100, 75, 27, 100, 300);
  assert.deepEqual(point, [100, 127]);
});

check('roundedPath：两点出直线，三点出带圆角的折线', () => {
  assert.equal(roundedPath([[0, 0], [10, 0]]), 'M 0 0 L 10 0');
  const d = roundedPath([[0, 0], [50, 0], [50, 50]], 8);
  assert.ok(d.startsWith('M 0 0'));
  assert.ok(d.includes('Q 50 0'), '拐点应使用二次贝塞尔倒角');
  assert.ok(d.endsWith('L 50 50'));
});

check('routeBetween：同行站点出直线，跨行站点出 Z 形', () => {
  const a = { x: 100, y: 100 };
  const b = { x: 400, y: 100 };
  assert.ok(!routeBetween(a, b).d.includes('Q'), '水平对齐不应产生拐角');

  const c = { x: 100, y: 100 };
  const d2 = { x: 600, y: 400 };
  const route = routeBetween(c, d2);
  assert.ok(route.d.includes('Q'), '跨行应产生 Z 形拐角');
  assert.ok(Array.isArray(route.label) && route.label.length === 2);
});

/* ------------------------------------------------------------------ */
/* 渲染层                                                              */
/* ------------------------------------------------------------------ */

console.log('\n== 渲染层 ==');

check('空站点列表渲染为空状态而不抛异常', () => {
  const container = new FakeContainer();
  const controller = renderTrajGraph(container, { sites: [], edges: [], meta: {} });
  assert.equal(controller, null);
  assert.ok(container.innerHTML.includes('empty-state'));
});

check('正常渲染输出站点、边与箭头 marker', () => {
  const container = new FakeContainer();
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  const controller = renderTrajGraph(container, graph, { loop: { detected: false } });
  assert.ok(controller, '应返回控制器');
  assert.ok(container.innerHTML.includes('tg-svg'));
  assert.ok(container.innerHTML.includes('tg-node'));
  assert.ok(container.innerHTML.includes('tg-edge'));
  assert.ok(container.innerHTML.includes('marker'));
  assert.ok(container.innerHTML.includes('轨迹路线图'));
  controller.destroy();
});

check('已访问节点带上 tg-visited 类，未访问带上 tg-unvisited', () => {
  const container = new FakeContainer();
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  renderTrajGraph(container, graph, {});
  assert.ok(container.innerHTML.includes('tg-visited'));
  assert.ok(container.innerHTML.includes('tg-unvisited'));
  assert.ok(container.innerHTML.includes('tg-current'));
});

check('同前缀共位对压缩成 RND_11/12，避免长名被迫降字号', () => {
  const container = new FakeContainer();
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  renderTrajGraph(container, graph, {});
  assert.ok(container.innerHTML.includes('RND_11/12'), '共位对应压成「前缀+编号/编号」');
  assert.ok(!container.innerHTML.includes('RND_11 + RND_12'), '不应再输出完整长名');
  assert.ok(!container.innerHTML.includes('font-size:10px'), '压缩后不应再触发降字号');
});

check('单相机站点的标题保持完整不改写', () => {
  const container = new FakeContainer();
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  renderTrajGraph(container, graph, {});
  assert.ok(container.innerHTML.includes('RND_10'));
  assert.ok(container.innerHTML.includes('RND_16'));
});

check('循环折叠提示只在 loop.detected 时出现', () => {
  const withLoop = new FakeContainer();
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  renderTrajGraph(withLoop, graph, { loop: { detected: true, method: 'period', loops: 12 } });
  assert.ok(withLoop.innerHTML.includes('tg-loop-note'));

  const withoutLoop = new FakeContainer();
  renderTrajGraph(withoutLoop, graph, { loop: { detected: false } });
  assert.ok(!withoutLoop.innerHTML.includes('tg-loop-note'));
});

check('坐标缺失时显示降级提示', () => {
  const bare = { cameras: Object.fromEntries(Object.keys(cameraMap).map((cam) => [cam, { desc: cameraMap[cam].desc, map_xy: null }])) };
  const graph = buildSiteGraph(bare, topology, []);
  const container = new FakeContainer();
  renderTrajGraph(container, graph, {});
  assert.ok(container.innerHTML.includes('tg-hint'));
});

check('重复渲染同一容器会先销毁上一个实例', () => {
  const container = new FakeContainer();
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  renderTrajGraph(container, graph, {});
  const first = container.__tgController;
  renderTrajGraph(container, graph, {});
  const second = container.__tgController;
  assert.ok(first && second && first !== second, '应生成新的控制器');
});

check('destroyTrajGraph 可重复调用（关闭弹窗路径不允许抛异常）', () => {
  const container = new FakeContainer();
  const graph = buildSiteGraph(floorplan, topology, sampleGroups);
  renderTrajGraph(container, graph, {});
  destroyTrajGraph(container);
  destroyTrajGraph(container);
  assert.equal(container.__tgController, null);
  assert.equal(container.innerHTML, '');
});

check('原始 desc 不进入 DOM（区域名取自白名单常量），相机 id 仍会被转义', () => {
  const injected = {
    cameras: {
      '<img onerror=x>': { desc: 'L2东侧走廊<script>alert(1)</script>', map_xy: [0.5, 0.5] },
    },
  };
  const graph = buildSiteGraph(injected, {}, []);
  // 区域名只可能是 REGION_ANCHORS 的键，原始 desc 永远不进模板
  assert.equal(graph.sites[0].region, '东侧走廊');
  const container = new FakeContainer();
  renderTrajGraph(container, graph, {});
  assert.ok(!container.innerHTML.includes('<script>'), 'desc 里的脚本不得出现在输出中');
  assert.ok(!container.innerHTML.includes('<img'), '相机 id 里的标签必须被转义');
  assert.ok(container.innerHTML.includes('&lt;IMG'), '转义后的相机 id 应以实体的形式出现');
});

console.log(`\n全部 ${passed} 项通过`);
