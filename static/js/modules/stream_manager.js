/**
 * MJPEG 长连接管理器（按可见性建流）
 *
 * 浏览器对同一域名的并发 HTTP/1.1 连接上限为 6，而每路 MJPEG（<img src="/stream/xxx">）
 * 都是一条永不结束的长连接。22 路全量建流会把连接池占满：只有约 6 路能出图，
 * 其余无限排队，连 /api/status、/api/metrics/reid 等 REST 轮询也抢不到槽位而饿死。
 *
 * 本模块按「卡片是否在视口内 + 全局并发上限」动态建流/断流：
 *   - 进入视口（含预留一屏）才赋 src 建流，离开视口立即断流；
 *   - 同时建流数不超过 MAX_CONCURRENT_STREAMS，给 REST 轮询与弹窗流留槽位；
 *   - 被挤下去的可见卡片保留最后一帧画面（冻结帧），不会变成空白或闪烁；
 *   - 不支持 IntersectionObserver 时回退为全量建流（与改造前行为一致）。
 */

// 同时建流上限。浏览器上限 6，这里留 2 个槽位：pollStatus 会并发发两个 REST
// 请求（/api/status 与 /api/identities），只留 1 个槽会让它们互相排队。
export const MAX_CONCURRENT_STREAMS = 4;
// 预留视口高度的 20%：只把「真正快要看到」的卡片算作可见。留太多（如整屏）
// 会把大量离屏卡片也塞进轮转队列，拉长每一路轮到连接的间隔。
const OBSERVER_ROOT_MARGIN = '20% 0px';
// 可见卡片数超过上限时的轮转周期（毫秒）；置 0 可关闭轮转（被挤下的卡片将长期冻结）
const ROTATE_INTERVAL_MS = 12000;
// 首轮填充周期：仍有可见卡片一帧未出过时用更短的间隔，避免大批卡片长时间全黑
const FILL_INTERVAL_MS = 3000;
// 冻结帧最大宽度，控制 dataURL 体积（多路卡片同时冻结时的内存占用）
const FREEZE_MAX_WIDTH = 480;
// 流中断（服务重启 / 404）后的重试冷却，避免占着槽位又死循环重连
const RETRY_DELAY_MS = 5000;
// 1x1 透明 GIF：断流占位图
export const BLANK_IMAGE =
  'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

const SUPPORTED = typeof IntersectionObserver !== 'undefined';

const entries = new Map();   // img 元素 → 流条目
const targets = new Map();   // 被观察的卡片元素 → 流条目
const rotation = [];         // 可见的非焦点 img，队首优先获得连接（最近进入视口者插到队首）

let observer = null;
let rotateTimer = null;
let rotateInterval = 0;      // 当前轮转定时器的周期，用于判断是否需要重建
let suspended = false;       // 弹窗内有独立视频流时挂起网格建流
let pageHidden = false;      // 标签页切到后台时全部断流，省服务端算力

function getObserver() {
  if (observer || !SUPPORTED) return observer;
  observer = new IntersectionObserver(onIntersect, {
    root: null,
    rootMargin: OBSERVER_ROOT_MARGIN,
    threshold: 0,
  });
  return observer;
}

function onIntersect(observerEntries) {
  const entered = [];
  observerEntries.forEach(record => {
    const entry = targets.get(record.target);
    if (!entry) return;
    if (record.isIntersecting) {
      if (!entry.visible) entered.push({ entry, rect: record.boundingClientRect });
      entry.visible = true;
    } else {
      entry.visible = false;
      removeFromRotation(entry.img);
    }
  });
  // 同一批进入视口的卡片（如首屏加载）按离视口顶部更近者优先，
  // 整批插到队首 → 满足「最近进入视口优先」，滚动时新露出的卡片抢得连接。
  entered.sort((a, b) => (a.rect.top - b.rect.top) || (a.rect.left - b.rect.left));
  for (let i = entered.length - 1; i >= 0; i--) {
    const img = entered[i].entry.img;
    if (entered[i].entry.pinned) continue;
    removeFromRotation(img);
    rotation.unshift(img);
  }
  reconcileStreams();
}

function removeFromRotation(img) {
  const idx = rotation.indexOf(img);
  if (idx >= 0) rotation.splice(idx, 1);
}

// 把当前显示的帧画到离屏 canvas 并转成 dataURL，用作断流后的静态占位画面
function captureLastFrame(img) {
  try {
    const width = img.naturalWidth;
    const height = img.naturalHeight;
    if (!width || !height) return null;
    const scale = Math.min(1, FREEZE_MAX_WIDTH / width);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.7);
  } catch {
    return null;   // 画布被污染或尚无解码帧时降级为空白占位
  }
}

function startStream(entry) {
  if (entry.streaming) return;
  entry.streaming = true;
  entry.everStreamed = true;
  entry.retryAt = 0;
  // 断流时 src 已被换成 data URI，这里赋回流地址即触发一次新的 MJPEG 连接
  entry.img.src = entry.streamSrc;
}

function stopStream(entry) {
  if (!entry.streaming) return;
  entry.streaming = false;
  // 断流手法说明：
  //   1) 仅 removeAttribute('src') 在部分浏览器不会立即中止 multipart/x-mixed-replace 长连接；
  //   2) img.src = '' 会被解析成当前页面 URL，反而多发一次无意义的 HTML 请求。
  // 因此改赋一个静态 data URI（优先用冻结的最后一帧），浏览器会立刻 abort 旧请求并关闭连接。
  entry.img.src = captureLastFrame(entry.img) || BLANK_IMAGE;
}

// 流被中断（服务端重启、404 等）时释放槽位，冷却后自动重试，避免整屏画面永久卡死
function onStreamError(entry) {
  if (!entry.streaming) return;   // 主动断流触发的 abort 不算故障
  entry.streaming = false;
  entry.retryAt = Date.now() + RETRY_DELAY_MS;
  entry.img.src = BLANK_IMAGE;
  reconcileStreams();
  setTimeout(reconcileStreams, RETRY_DELAY_MS + 50);
}

/**
 * 依据「焦点常驻 + 可见性 + 并发上限」重新计算该建流与该断流的卡片。
 * 幂等：可以随时调用（进出视口、布局切换、弹窗开合、标签页前后台）。
 */
export function reconcileStreams() {
  if (!SUPPORTED) return;   // 回退模式下全部常开，不做增减
  if (suspended || pageHidden) {
    entries.forEach(stopStream);
    updateRotateTimer();
    return;
  }

  const active = new Set();
  const now = Date.now();
  let budget = MAX_CONCURRENT_STREAMS;
  // 焦点大屏（1x1 / 1+N 主视角）必须始终保持建流
  entries.forEach(entry => {
    if (entry.pinned && !(entry.retryAt > now)) {
      active.add(entry);
      budget -= 1;
    }
  });
  if (budget < 0) budget = 0;

  for (const img of rotation) {
    if (budget <= 0) break;
    const entry = entries.get(img);
    if (!entry || entry.pinned || !entry.visible) continue;
    if (entry.retryAt > now) continue;   // 故障冷却中，槽位让给其他卡片
    active.add(entry);
    budget -= 1;
  }

  // 先断后建：避免瞬时连接数超过浏览器上限而把新请求排进队列
  entries.forEach(entry => {
    if (!active.has(entry)) stopStream(entry);
  });
  entries.forEach(entry => {
    if (active.has(entry)) startStream(entry);
  });
  updateRotateTimer();
}

function countStarving() {
  const now = Date.now();
  let starving = 0;
  entries.forEach(entry => {
    if (entry.visible && !entry.pinned && !entry.streaming && !(entry.retryAt > now)) {
      starving += 1;
    }
  });
  return starving;
}

// 是否还有可见卡片一帧都没出过（首轮填充阶段用更短的轮转周期）
function needsFastFill() {
  const now = Date.now();
  let fast = false;
  entries.forEach(entry => {
    if (entry.visible && !entry.pinned && !entry.everStreamed && !(entry.retryAt > now)) {
      fast = true;
    }
  });
  return fast;
}

function updateRotateTimer() {
  const needed = ROTATE_INTERVAL_MS > 0 && !suspended && !pageHidden && countStarving() > 0;
  if (!needed) {
    if (rotateTimer !== null) clearInterval(rotateTimer);
    rotateTimer = null;
    rotateInterval = 0;
    return;
  }
  const interval = needsFastFill() ? FILL_INTERVAL_MS : ROTATE_INTERVAL_MS;
  if (rotateTimer !== null && rotateInterval === interval) return;
  if (rotateTimer !== null) clearInterval(rotateTimer);
  rotateInterval = interval;
  rotateTimer = setInterval(rotateOnce, interval);
}

// 把当前持有连接的一整批挪到队尾，让被挤下去的可见卡片轮到连接。
// 逐路轮转时 n 路走完一圈需要 n 个周期（22 路 × 12s ≈ 4 分钟），
// 整批轮转只需 ceil(n / 上限) 个周期（9 路可见时约 36s）。
function rotateOnce() {
  if (rotation.length <= 1) return;
  const step = Math.min(MAX_CONCURRENT_STREAMS, rotation.length - 1);
  rotation.push(...rotation.splice(0, step));
  reconcileStreams();
}

/**
 * 登记一路视频流图片。调用前 img 必须带 data-stream-src（真实流地址），
 * 其 src 应为占位图，由本模块决定何时赋上真实地址。
 * @param {HTMLImageElement} img
 * @param {{camId?: string, pinned?: boolean}} options pinned=true 表示焦点大屏，始终保持建流
 */
export function registerStreamImage(img, { camId = '', pinned = false } = {}) {
  if (!img) return;
  const streamSrc = img.getAttribute('data-stream-src') || '';
  if (!streamSrc) return;
  const entry = {
    img,
    camId,
    pinned,
    streamSrc,
    visible: pinned,
    streaming: false,
    everStreamed: false,
    retryAt: 0,
    target: null,
  };
  entries.set(img, entry);
  img.addEventListener('error', () => onStreamError(entry));

  if (!SUPPORTED) {
    // 回退：不支持 IntersectionObserver 时维持改造前的全量建流行为
    startStream(entry);
    return;
  }

  let target = img.closest('.cam-card, .carousel-thumb-item') || img;
  if (targets.has(target)) target = img;   // 同一卡片内多个流图时退化为观察 img 自身
  entry.target = target;
  targets.set(target, entry);
  getObserver().observe(target);
}

/** 网格重建前调用：断开全部旧连接并清空登记表 */
export function resetStreamRegistry() {
  if (observer) observer.disconnect();
  // 旧卡片马上会被 innerHTML 覆盖，但「仅从 DOM 移除 <img>」不保证浏览器立即中止
  // multipart 响应，所以这里显式把 src 换成占位图来关闭长连接。
  entries.forEach(entry => {
    if (entry.streaming) {
      entry.streaming = false;
      entry.img.src = BLANK_IMAGE;
    }
  });
  entries.clear();
  targets.clear();
  rotation.length = 0;
  if (rotateTimer !== null) {
    clearInterval(rotateTimer);
    rotateTimer = null;
  }
  rotateInterval = 0;
}

/** 弹窗内有独立视频流（焦点视窗/放大监视器/ROI 画板）时挂起网格建流，把连接槽位让出来 */
export function setStreamsSuspended(value) {
  const next = Boolean(value);
  if (next === suspended) return;
  suspended = next;
  reconcileStreams();
}

/** 调试用：DevTools 控制台执行 __labStreamStats() 可核查实际建流数 */
export function getStreamStats() {
  const streaming = [];
  let visible = 0;
  entries.forEach(entry => {
    if (entry.streaming) streaming.push(entry.camId || entry.img.id || '?');
    if (entry.visible) visible += 1;
  });
  return {
    supported: SUPPORTED,
    limit: MAX_CONCURRENT_STREAMS,
    registered: entries.size,
    visible,
    suspended,
    pageHidden,
    streaming,
  };
}

if (typeof document !== 'undefined') {
  pageHidden = document.hidden === true;
  // 标签页切到后台时全部断流，避免 22 路继续白烧服务端编码算力
  document.addEventListener('visibilitychange', () => {
    pageHidden = document.hidden === true;
    reconcileStreams();
  });
  window.__labStreamStats = getStreamStats;
}
