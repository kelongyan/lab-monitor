/**
 * 前端 MJPEG 连接管理器测试（static/js/modules/stream_manager.js）
 *
 * 纯 Node 环境跑：用最小 DOM 桩件（IntersectionObserver / document / img）驱动模块，
 * 断言「并发上限、焦点常驻、离开视口断流、冻结最后一帧、弹窗挂起、轮转、故障冷却」。
 *
 * 运行：node tests/test_stream_manager_frontend.mjs
 */
import assert from 'node:assert/strict';

const FROZEN = 'data:image/jpeg;base64,FROZEN';
const docListeners = {};
let rotateCb = null;
let rotateMs = 0;

// 与 static/js/modules/stream_manager.js 的私有常量保持一致：
//   FILL_INTERVAL_MS(=3000, 仍有卡片一帧未出时的快速填充周期)
//   ROTATE_INTERVAL_MS(=6000, 常规轮转周期，1a5b17b 由 12s 调到 6s)
// 该模块不导出这两个常量，只能在这里镜像一份；改动模块时请同步改这里，
// 否则本用例会像 1a5b17b 之后那样静默失败。
const EXPECTED_FILL_MS = 3000;
const EXPECTED_ROTATE_MS = 6000;

class FakeIntersectionObserver {
  constructor(callback) {
    this.callback = callback;
    this.targets = new Set();
    FakeIntersectionObserver.last = this;
  }
  observe(target) { this.targets.add(target); }
  unobserve(target) { this.targets.delete(target); }
  disconnect() { this.targets.clear(); }
}

class FakeImg {
  constructor(camId, cardClass = 'cam-card') {
    this.id = `stream-img-${camId}`;
    this.camId = camId;
    this.src = 'INIT';
    this.naturalWidth = 640;
    this.naturalHeight = 360;
    this.handlers = {};
    this.attrs = { 'data-stream-src': `/stream/${camId}`, 'data-cam-id': camId };
    this.card = { cls: cardClass, owner: this };
  }
  getAttribute(name) { return this.attrs[name] ?? null; }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  dispatch(type) { (this.handlers[type] || []).forEach(fn => fn()); }
  closest(selector) {
    return selector.includes(this.card.cls) ? this.card : null;
  }
}

globalThis.IntersectionObserver = FakeIntersectionObserver;
globalThis.window = {};
globalThis.document = {
  hidden: false,
  addEventListener(type, fn) { (docListeners[type] ||= []).push(fn); },
  createElement() {
    return {
      width: 0,
      height: 0,
      getContext: () => ({ drawImage() {} }),
      toDataURL: () => FROZEN,
    };
  },
};
// 拦截定时器：轮转回调改为手动触发，故障重试不占用真实时间
globalThis.setInterval = (fn, ms) => { rotateCb = fn; rotateMs = ms; return 1; };
globalThis.clearInterval = () => { rotateCb = null; rotateMs = 0; };
globalThis.setTimeout = () => 0;

const sm = await import('../static/js/modules/stream_manager.js');

// 并发上限直接取模块导出值，避免调整上限后测试因硬编码数字而失效
const MAX = sm.MAX_CONCURRENT_STREAMS;

// camNames(3, 2) → ['cam_03', 'cam_04']
function camNames(start, count) {
  return Array.from(
    { length: count },
    (_, i) => `cam_${String(start + i).padStart(2, '0')}`,
  );
}

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`  PASS  ${name}`);
}

// 把一批卡片标记为进入/离开视口（top 越小越靠近视口顶部）
function setVisible(imgs, isIntersecting) {
  const records = imgs.map((img, idx) => ({
    target: img.card,
    isIntersecting,
    boundingClientRect: { top: idx * 100, left: 0 },
  }));
  FakeIntersectionObserver.last.callback(records);
}

function buildGrid(count, { pinnedFirst = false } = {}) {
  sm.resetStreamRegistry();
  const imgs = [];
  for (let i = 1; i <= count; i++) {
    const camId = `cam_${String(i).padStart(2, '0')}`;
    const pinned = pinnedFirst && i === 1;
    const img = new FakeImg(camId, pinned ? 'theater-focus-card' : 'cam-card');
    // 焦点卡片同时带 cam-card 类，closest 命中 '.cam-card, .carousel-thumb-item'
    if (pinned) img.card.cls = 'cam-card';
    sm.registerStreamImage(img, { camId, pinned });
    imgs.push(img);
  }
  sm.reconcileStreams();
  return imgs;
}

console.log('== 前端 MJPEG 连接管理器 ==');

// 1. 22 路全部登记后，未进入视口时一路都不建流
const imgs = buildGrid(22);
check('未进入视口时不建流', () => {
  assert.equal(sm.getStreamStats().streaming.length, 0);
  assert.equal(imgs[0].src, 'INIT');
});

// 2. 首屏 10 路进入视口：只建 MAX 路，且优先靠近视口顶部的那几路
setVisible(imgs.slice(0, 10), true);
check('可见卡片超上限时只建 MAX_CONCURRENT_STREAMS 路', () => {
  const stats = sm.getStreamStats();
  assert.equal(stats.limit, MAX);
  assert.equal(stats.visible, 10);
  assert.deepEqual(stats.streaming, camNames(1, MAX));
  assert.equal(imgs[0].src, '/stream/cam_01');
  assert.equal(imgs[MAX + 1].src, 'INIT');   // 排在后面的可见卡片暂不建流
});

check('仍有卡片一帧未出时用更短的填充周期', () => {
  assert.equal(rotateMs, EXPECTED_FILL_MS);
});

// 3. 向下滚动：正在推流的那几路离开视口 → 断流并冻结最后一帧，后面的可见卡片接管连接
setVisible(imgs.slice(0, MAX), false);
check('离开视口断流并保留最后一帧', () => {
  const stats = sm.getStreamStats();
  assert.deepEqual(stats.streaming, camNames(MAX + 1, MAX));
  assert.equal(imgs[0].src, FROZEN);   // 不是空白，画面冻结在最后一帧
  assert.equal(imgs[MAX].src, `/stream/cam_${String(MAX + 1).padStart(2, '0')}`);
});

// 4. 更下方的卡片进入视口：按「最近进入视口优先」抢到连接
setVisible(imgs.slice(10, 15), true);
check('最近进入视口的卡片优先建流', () => {
  const stats = sm.getStreamStats();
  // 首屏 10 路里前 MAX 路已滚出视口，又新进来 5 路
  assert.equal(stats.visible, 10 - MAX + 5);
  assert.deepEqual(stats.streaming, camNames(11, MAX));
  assert.equal(imgs[MAX].src, FROZEN); // 被挤下去的那路冻结画面而不是空白
});

check('全部可见卡片都出过画面后回到常规轮转周期', () => {
  // 可见卡片数超过并发上限时要轮几轮才能让每一路都出过一帧；
  // 只要还有「一帧未出」的卡片，模块就该维持更短的填充周期。
  for (let i = 0; i < 20 && rotateMs === EXPECTED_FILL_MS && typeof rotateCb === 'function'; i++) {
    rotateCb();
  }
  assert.equal(rotateMs, EXPECTED_ROTATE_MS);
});

// 5. 轮转：手动触发轮转回调，被挤下去的可见卡片应拿到连接
check('轮转让被挤下的可见卡片重新拿到连接', () => {
  assert.equal(typeof rotateCb, 'function');   // 存在饥饿卡片时才开启轮转定时器
  const before = sm.getStreamStats().streaming;
  rotateCb();
  const after = sm.getStreamStats().streaming;
  assert.notDeepEqual(after, before);
  assert.equal(after.length, MAX);
  // 整批轮转：上一轮持有连接的那批全部让位
  before.forEach(cam => assert.ok(!after.includes(cam)));
});

// 6. 焦点大屏常驻建流，且总建流数仍不超上限
const focusImgs = buildGrid(22, { pinnedFirst: true });
check('焦点大屏未进入视口也保持建流', () => {
  const stats = sm.getStreamStats();
  assert.deepEqual(stats.streaming, ['cam_01']);
  assert.equal(focusImgs[0].src, '/stream/cam_01');
});

setVisible(focusImgs.slice(1, 12), true);
check('焦点 + 可见卡片合计不超上限', () => {
  const stats = sm.getStreamStats();
  assert.equal(stats.streaming.length, MAX);
  assert.ok(stats.streaming.includes('cam_01'));
});

// 7. 弹窗自带视频流时挂起网格建流，把连接槽位让给弹窗与 REST 轮询
sm.setStreamsSuspended(true);
check('弹窗打开时网格全部断流', () => {
  const stats = sm.getStreamStats();
  assert.equal(stats.suspended, true);
  assert.equal(stats.streaming.length, 0);
  assert.equal(focusImgs[0].src, FROZEN);
});

sm.setStreamsSuspended(false);
check('弹窗关闭后恢复建流', () => {
  const stats = sm.getStreamStats();
  assert.equal(stats.suspended, false);
  assert.equal(stats.streaming.length, MAX);
  assert.ok(stats.streaming.includes('cam_01'));
});

// 8. 标签页切到后台：全部断流；回到前台恢复
globalThis.document.hidden = true;
docListeners.visibilitychange.forEach(fn => fn());
check('标签页后台时全部断流', () => {
  const stats = sm.getStreamStats();
  assert.equal(stats.pageHidden, true);
  assert.equal(stats.streaming.length, 0);
});

globalThis.document.hidden = false;
docListeners.visibilitychange.forEach(fn => fn());
check('回到前台恢复建流', () => {
  assert.equal(sm.getStreamStats().streaming.length, MAX);
});

// 9. 流中断（服务重启 / 404）：让出槽位，且冷却期内不立刻重连
check('流报错后让出槽位且进入冷却', () => {
  const before = sm.getStreamStats().streaming;
  const brokenId = before.find(id => id !== 'cam_01');
  const brokenImg = focusImgs.find(img => img.camId === brokenId);
  brokenImg.dispatch('error');
  const after = sm.getStreamStats().streaming;
  assert.ok(!after.includes(brokenId));
  assert.equal(after.length, MAX);   // 空出的槽位被其他可见卡片接管
});

// 10. 回退路径：不支持 IntersectionObserver 时维持全量建流
delete globalThis.IntersectionObserver;
const fallback = await import('../static/js/modules/stream_manager.js?fallback=1');
check('无 IntersectionObserver 时回退为全量建流', () => {
  for (let i = 1; i <= 22; i++) {
    const camId = `cam_${String(i).padStart(2, '0')}`;
    fallback.registerStreamImage(new FakeImg(camId), { camId });
  }
  const stats = fallback.getStreamStats();
  assert.equal(stats.supported, false);
  assert.equal(stats.streaming.length, 22);
});

// ── 写请求守卫头（static/js/utils/api.js） ─────────────────────────────────
globalThis.location = { protocol: 'http:', host: '127.0.0.1:8000' };
const api = await import('../static/js/utils/api.js');
const fetchCalls = [];
globalThis.fetch = async (url, options) => {
  fetchCalls.push({ url, options });
  return { ok: true, status: 200, text: async () => '{"status":"success"}' };
};

console.log('== 写请求守卫头 ==');

await api.fetchJson('/api/roi', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: '{}',
});
check('POST 自动补 X-Lab-Monitor-Request 且保留原有头', () => {
  const headers = fetchCalls.at(-1).options.headers;
  assert.equal(headers[api.WRITE_GUARD_HEADER], '1');
  assert.equal(headers['Content-Type'], 'application/json');
});

await api.fetchJson('/api/status');
check('GET 不补守卫头', () => {
  const headers = fetchCalls.at(-1).options.headers;
  assert.ok(!headers || headers[api.WRITE_GUARD_HEADER] === undefined);
});

await api.fetchJson('/api/topology', { method: 'delete' });
check('小写方法名同样识别为写请求', () => {
  assert.equal(fetchCalls.at(-1).options.headers[api.WRITE_GUARD_HEADER], '1');
});

await api.fetchJson('/api/roi', {
  method: 'PUT',
  headers: new Headers({ 'Content-Type': 'application/json' }),
});
check('headers 为 Headers 实例时也能注入', () => {
  assert.equal(fetchCalls.at(-1).options.headers.get(api.WRITE_GUARD_HEADER), '1');
});

console.log(`\n全部 ${passed} 项通过`);
