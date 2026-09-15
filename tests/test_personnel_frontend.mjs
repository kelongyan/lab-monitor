/**
 * 人员档案库前端测试（static/js/modules/personnel.js + modals.js 身份档案弹窗）
 *
 * 纯 Node 环境跑：最小 DOM 桩件驱动真模块，断言
 * 「列表渲染、SIM 徽标只给 synthetic、头像 URL、分页、检索、详情、删除两步确认、
 *   注册照 multipart 上传、以图搜人 Top-K 渲染」。
 *
 * 重点防的是两个"静默失败"：
 *   1. 头像 URL 自己拼 → 与实际 mount 拼法不符 → 全部 404 且无声降级成首字块
 *   2. 前端做本地过滤 → 只筛当前页 → 用户搜不到人却不知道该怪谁
 *
 * 运行：node tests/test_personnel_frontend.mjs
 */
import assert from 'node:assert/strict';

globalThis.location = { protocol: 'http:', host: '127.0.0.1:8931' };

console.log('== 人员档案库前端 ==');

const failures = [];
let passed = 0;
async function check(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failures.push([name, e]);
    console.log(`  FAIL  ${name}\n        ${e.message.split('\n')[0]}`);
  }
}

// --------------------------------------------------------------------------- //
// 最小 DOM 桩件                                                                 //
// --------------------------------------------------------------------------- //
class FakeClassList {
  constructor() { this.set = new Set(); }
  add(...c) { c.forEach(x => this.set.add(x)); }
  remove(...c) { c.forEach(x => this.set.delete(x)); }
  contains(c) { return this.set.has(c); }
  toggle(c, on) { on ? this.set.add(c) : this.set.delete(c); }
}

class FakeEl {
  constructor(tag = 'div', id = '') {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.classList = new FakeClassList();
    this._innerHTML = '';
    this.value = '';
    this.disabled = false;
    this.dataset = {};
    this.style = {};
    this.attrs = {};
    this.handlers = {};
    this.children = [];
    this.isConnected = true;
    // <select> 必须有 options：syncControls 会 Array.from(select.options)，
    // 缺了会抛 "undefined is not iterable" 并被 catch 成"加载失败"（桩件坑）
    this.options = [];
  this._cards = [];
  this._sourceBtns = [];
  this.files = [];          // <input type=file>：上传用例靠它喂"已选文件"
  this._readyState = 0;
}
  get innerHTML() { return this._innerHTML; }
  set innerHTML(v) {
    this._innerHTML = String(v);
    // <select> 的 options 要跟着 innerHTML 一起变：syncControls 用
    // Array.from(select.options).map(o => o.value) 判断"下拉是否需要重建"，
    // 桩件不更新的话每次都比对失败、反复重建（这正是先前 "undefined is not
    // iterable" 之外的第二层桩件坑）。
    if (this.tagName === 'SELECT') {
      this.options = [...this._innerHTML.matchAll(/<option value="([^"]*)"/g)]
        .map(m => ({ value: m[1] }));
    }
    // 真浏览器里 innerHTML 赋值会丢弃旧子树（连带旧 handler）。桩件不换元素，
    // 所以要手动清掉被重绘按钮上的监听，否则 renderPager 每画一次就多绑一层。
    if (this.id === 'personnel-pager') {
      ['personnel-prev', 'personnel-next'].forEach(k => {
        const t = els.get(k); if (t) t.handlers = {};
      });
    }
    if (this.id === 'personnel-detail') {
      ['personnel-detail-back', 'personnel-detail-delete',
       'personnel-photo-upload', 'personnel-photo-input'].forEach(k => {
        const t = els.get(k); if (t) t.handlers = {};
      });
    }
  }
  getAttribute(n) { return this.attrs[n] ?? null; }
  setAttribute(n, v) { this.attrs[n] = String(v); }
  removeAttribute(n) { delete this.attrs[n]; }
  addEventListener(t, fn) { (this.handlers[t] ||= []).push(fn); }
  removeEventListener() {}
  dispatch(t, ev = {}) { (this.handlers[t] || []).forEach(fn => fn({ target: this, ...ev })); }
  appendChild(c) { this.children.push(c); return c; }
  replaceWith() {}
  remove() { this.isConnected = false; }
  focus() {}
  click() {}               // file input 的程序化打开；桩件里是 no-op
  closest(sel) {
    // 只支持 [data-*] 这一种：来源筛选靠 event.target.closest('[data-source]')
    const m = /^\[data-([a-z-]+)\]$/.exec(sel);
    if (m) {
      const key = m[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      if (Object.prototype.hasOwnProperty.call(this.dataset, key)) return this;
    }
    return null;
  }
  querySelector(sel) {
    // 详情面板里 bindClipPlayback 用 #id 取播放器子树；桩件按 id 查全局表即可
    const m = /^#([\w-]+)$/.exec(sel);
    if (m) return els.get(m[1]) || null;
    if (sel === '.person-clip') return this._clips?.[0] || null;
    return null;
  }
  querySelectorAll(sel) {
    if (sel === '.person-card') return this._cards;
    if (sel === '[data-source]') return this._sourceBtns;
    if (sel === '.person-clip') return this._clips || [];
    if (sel === '[data-gid]') return this._gidRows || [];
    if (sel === '.person-clip-play') return (this._clips || []).map(c => c._playBtn);
    return [];
  }
  // --- <video> 行为（bindClipPlayback 依赖）---
  get readyState() { return this._readyState ?? 0; }
  set readyState(v) { this._readyState = v; }
  get currentTime() { return this._currentTime ?? 0; }
  set currentTime(v) { this._currentTime = v; }
  get duration() { return this._duration ?? 0; }
  set duration(v) { this._duration = v; }
  load() {
    // 真浏览器 load() 后会异步补上 metadata；桩件同步模拟，readyState 至少为 1
    this._readyState = Math.max(this._readyState ?? 0, 1);
  }
  pause() { this._paused = true; }
  play() { this._played = true; return Promise.resolve(); }
}

const els = new Map();
function mk(id, tag = 'div') {
  const el = new FakeEl(tag, id);
  els.set(id, el);
  return el;
}

const STATIC_IDS = ['personnel-modal', 'personnel-grid', 'personnel-pager',
  'personnel-list-pane', 'personnel-detail', 'personnel-search-input',
  'personnel-dept-filter', 'personnel-source-filter', 'personnel-reload',
  'personnel-create', 'toast-container',
  // 下面这些由 personDetailHtml() / renderPager() 生成；真浏览器里 getElementById
  // 能找到，桩件里先建好，模块才能把 handler 绑上（否则绑定时静默失败）
  'personnel-detail-back', 'personnel-detail-delete', 'personnel-prev', 'personnel-next',
  // P1-1 片段播放器（player / video / label / close 都由 clipsSection 渲染）
  'personnel-player', 'personnel-video', 'personnel-player-label',
  'personnel-player-close',
  // 注册照上传入口（personDetailHtml 渲染的按钮 + 隐藏 file input）
  'personnel-photo-upload', 'personnel-photo-input',
  // ReID 身份档案弹窗（modals.js openIdentitySearchModal 渲染）。
  // trajectory-modal 必须存在：openModal 靠它把 classList 置为 active，
  // beginModalRequest 的 isCurrent() 再依赖 active 判定 —— 缺了它
  // openIdentitySearchModal 会在 fetch 后静默 return，body 永远不渲染。
  'modal-traj-title', 'modal-traj-body', 'trajectory-modal',
  'btn-by-image-search', 'by-image-input', 'by-image-result',
  // 跨相机抓拍证据区块（modals.js loadCrossCameraEvidence 渲染的目标容器）。
  // 桩件里 innerHTML 只是字符串，真实 DOM 里这个 div 由 showTrajectoryModal
  // 写入；不预先注册的话 getElementById 返回 null，函数会静默 return，
  // 面板内容永远测不到。
  'cross-camera-panel'];

const TAG_BY_ID = {
  'personnel-search-input': 'input',
  'personnel-dept-filter': 'select',
  'personnel-video': 'video',
};

function resetDom() {
  els.clear();
  STATIC_IDS.forEach(id => mk(id, TAG_BY_ID[id] || 'div'));
  const dept = els.get('personnel-dept-filter');
  dept.innerHTML = '<option value="">全部部门</option>';
  els.get('personnel-list-pane').classList.add('active');
  els.get('personnel-source-filter')._sourceBtns = [];
}
resetDom();

globalThis.document = {
  hidden: false,
  activeElement: null,
  body: new FakeEl('body'),
  getElementById: (id) => els.get(id) || null,
  createElement: (tag) => new FakeEl(tag),
  addEventListener() {},
  querySelector: () => null,
  querySelectorAll: () => [],
};
globalThis.window = { prompt: () => null };
globalThis.HTMLImageElement = class {};

// fetch 打桩：记录后端收到的参数（"筛选必须交后端"的证据）
const REQUESTS = [];
let scenario = () => ({ personnel: [], count: 0, departments: [] });
let failNext = null;

globalThis.fetch = async (url, options = {}) => {
  REQUESTS.push({ url: String(url), options });
  if (failNext) { const payload = failNext; failNext = null;
    return { ok: false, status: 500, text: async () => JSON.stringify(payload) }; }
  return { ok: true, status: 200, text: async () => JSON.stringify(scenario(String(url))) };
};

const tick = (ms = 5) => new Promise(r => setTimeout(r, ms));
const lastUrl = () => REQUESTS.at(-1)?.url || '';

function person(over = {}) {
  return {
    person_id: 'p1', name: '张伟', employee_no: 'QLU-26-0001', department: '网络运维',
    source: 'synthetic', identity_count: 3, total_appearances: 166, photo_count: 1,
    last_seen: Date.now() / 1000 - 120, last_camera: 'rnd_04',
    thumb_path: 'outputs/personnel_crops/p1.jpg', thumb_url: '/personnel-crops/p1.jpg',
    ...over,
  };
}
const grid = () => els.get('personnel-grid').innerHTML;

const mod = await import('../static/js/modules/personnel.js');
// 工具条 handler 在 initPersonnelPanel 里绑；必须先调，否则 reload/搜索点击全是空操作
mod.initPersonnelPanel();

/**
 * 渲染一次列表，并把渲染出的卡片交给调用方（供点击测试）。
 *
 * 模块用 grid.querySelectorAll('.person-card') 取卡片再绑 handler。桩件里
 * innerHTML 只是一个字符串，所以这里把渲染结果解析成假卡片、替换掉
 * querySelectorAll —— 必须发生在 renderGrid **之前**（即触发 load 前），
 * 否则模块拿到的是上一次的卡片集合。
 */
async function reloadList(payload) {
  scenario = (url) => (typeof payload === 'function' ? payload(url) : payload);
  const target = els.get('personnel-grid');
  target._cards = [];
  target._htmlAtCardCollect = null;
  target.querySelectorAll = function (sel) {
    if (sel !== '.person-card') return [];
    // renderGrid 已把新 HTML 写进 _innerHTML，据此重建卡片（带 data-person）
    const ids = [...this._innerHTML.matchAll(/data-person="([^"]*)"/g)].map(m => m[1]);
    this._cards = ids.map(pid => {
      const c = new FakeEl('div');
      c.dataset = { person: pid };
      return c;
    });
    return this._cards;
  };
  els.get('personnel-reload').dispatch('click');
  await tick(10);
  return target._cards;
}

/**
 * 把检索与分页重置回"第一页 + 无筛选"。
 *
 * 模块的 state 是跨调用保留的（这是有意的：用户关掉弹窗再打开不该丢筛选），
 * 所以用例之间必须显式复位，否则前一个用例留下的 q/offset 会污染后一个
 * —— 这类"测试自己互相干扰"最容易伪装成产品 bug。
 */
async function resetFilters() {
  const input = els.get('personnel-search-input');
  if (input.value) { input.value = ''; input.dispatch('input'); await tick(320); }
  els.get('personnel-dept-filter').dispatch('change', { target: { value: '' } });
  await tick(10);
}

// --------------------------------------------------------------------------- //

await check('打开弹窗即拉列表，请求带分页参数', async () => {
  REQUESTS.length = 0;
  scenario = () => ({ personnel: [person()], count: 1, departments: ['网络运维'] });
  mod.openPersonnelModal();
  await tick(10);
  assert.equal(REQUESTS.length, 1, `应只发 1 个请求，实际 ${REQUESTS.length}`);
  assert.match(lastUrl(), /^\/api\/personnel\?/);
  assert.match(lastUrl(), /limit=24/);
  assert.match(lastUrl(), /offset=0/);
  assert.ok(els.get('personnel-modal').classList.contains('active'));
});

await check('列表渲染姓名/工号/部门与聚合数字', async () => {
  await reloadList({ personnel: [person()], count: 1, departments: ['网络运维'] });
  const html = grid();
  for (const [what, needle] of [['姓名', '张伟'], ['工号', 'QLU-26-0001'],
                                ['部门', '网络运维'], ['身份数', '3 身份'],
                                ['出现次数', '166 次'], ['相机', 'RND_04']]) {
    assert.ok(html.includes(needle), `缺${what}: ${needle}`);
  }
});

await check('synthetic 打「模拟」徽标，real 不打', async () => {
  await reloadList({ personnel: [person()], count: 1, departments: [] });
  assert.ok(grid().includes('person-badge sim'), 'synthetic 缺 SIM 徽标');
  await reloadList({ personnel: [person({ source: 'real', person_id: 'p2' })],
                     count: 1, departments: [] });
  assert.ok(!grid().includes('person-badge sim'), 'real 不该有 SIM 徽标');
  assert.ok(grid().includes('data-person="p2"'), 'real 卡片未渲染');
});

await check('头像 URL 只用后端 thumb_url，不自己拼磁盘路径', async () => {
  await reloadList({ personnel: [person()], count: 1, departments: [] });
  const html = grid();
  assert.ok(html.includes('src="/personnel-crops/p1.jpg"'), `实际: ${html.slice(0, 160)}`);
  assert.ok(!html.includes('outputs/personnel_crops'), '不该把磁盘路径当 URL');
});

await check('后端未给 thumb_url 时用文件名兜底', async () => {
  // 有 thumb_path 无 thumb_url：兜底应取 thumb_path 的**文件名**（这里是 p1.jpg，
  // 因为 person() 的默认 thumb_path 指向 p1），而不是把整串磁盘路径当 URL。
  await reloadList({ personnel: [person({ person_id: 'p3', thumb_url: undefined })],
                     count: 1, departments: [] });
  const html = grid();
  assert.ok(html.includes('src="/personnel-crops/p1.jpg"'), `实际: ${html.slice(0, 200)}`);
  assert.ok(!html.includes('outputs/'), '不该把 outputs/ 带进 URL');
  assert.ok(!html.includes('personnel_crops'), '不该把下划线目录名当 URL 段');
});

await check('无头像时降级为首字块（而不是坏图）', async () => {
  await reloadList({ personnel: [person({ thumb_path: null, thumb_url: null })],
                     count: 1, departments: [] });
  const html = grid();
  assert.ok(html.includes('person-avatar fallback'), '缺首字降级块');
  assert.ok(!html.includes('<img'), '无图不该输出 img');
  assert.ok(html.includes('>张<'), '首字块应显示姓氏');
});

await check('搜索走后端且 250ms 防抖', async () => {
  await reloadList({ personnel: [], count: 0, departments: [] });
  REQUESTS.length = 0;
  const input = els.get('personnel-search-input');
  input.value = '张';
  input.dispatch('input');
  await tick(60);
  assert.equal(REQUESTS.length, 0, '防抖期内不该发请求');
  await tick(320);
  assert.equal(REQUESTS.length, 1, `防抖后应发 1 个，实际 ${REQUESTS.length}`);
  assert.ok(lastUrl().includes('q=%E5%BC%A0'), `q 未编码: ${lastUrl()}`);
});

await check('空结果区分「没匹配」与「还没有档案」', async () => {
  await resetFilters();
  // 无筛选 + 空库 → "还没有档案" + 造数据指引
  await reloadList({ personnel: [], count: 0, departments: [] });
  assert.ok(grid().includes('还没有人员档案'), `实际: ${grid()}`);
  assert.ok(grid().includes('seed_personnel_mock.py'), '空库应提示如何造数据');

  // 有筛选 + 空结果 → "没有匹配"
  const input = els.get('personnel-search-input');
  input.value = '不存在的人';
  input.dispatch('input');
  await tick(320);
  assert.ok(grid().includes('没有匹配的人员档案'), `实际: ${grid()}`);
  await resetFilters();
});

await check('分页文案与按钮禁用态正确', async () => {
  await reloadList({ personnel: [person()], count: 60, departments: [] });
  const pager = els.get('personnel-pager').innerHTML;
  assert.ok(pager.includes('1–24 / 共 60 人 · 第 1/3 页'), `实际: ${pager}`);
  assert.ok(/id="personnel-prev" disabled/.test(pager), '第 1 页上一页应禁用');
  assert.ok(!/id="personnel-next" disabled/.test(pager), '还有下一页不该禁用');
});

await check('翻页带 offset', async () => {
  REQUESTS.length = 0;
  await reloadList({ personnel: [person()], count: 60, departments: [] });
  els.get('personnel-next').dispatch('click');
  await tick(10);
  assert.ok(lastUrl().includes('offset=24'), `翻页 offset 错: ${lastUrl()}`);
});

await check('换部门筛选回到第一页并带参数', async () => {
  await resetFilters();
  const cards = await reloadList({ personnel: [person()], count: 60,
                                   departments: ['网络运维'] });
  assert.ok(cards.length >= 1, '前置渲染未拿到卡片');
  REQUESTS.length = 0;
  els.get('personnel-next').dispatch('click');
  await tick(10);
  assert.ok(lastUrl().includes('offset=24'), `前置翻页未生效: ${lastUrl()}`);
  els.get('personnel-dept-filter').dispatch('change', { target: { value: '网络运维' } });
  await tick(10);
  assert.ok(lastUrl().includes('offset=0'), `换筛选应回第一页: ${lastUrl()}`);
  assert.ok(lastUrl().includes('department=%E7%BD%91'), `缺部门参数: ${lastUrl()}`);
  await resetFilters();
});

await check('来源筛选按钮带 source 参数', async () => {
  REQUESTS.length = 0;
  await reloadList({ personnel: [person()], count: 1, departments: [] });
  const btn = new FakeEl('button');
  btn.dataset = { source: 'synthetic' };
  els.get('personnel-source-filter')._sourceBtns = [btn];
  els.get('personnel-source-filter').dispatch('click', { target: btn });
  await tick(10);
  assert.ok(lastUrl().includes('source=synthetic'), `实际: ${lastUrl()}`);
});

await check('部门下拉由后端 departments 填充', async () => {
  await reloadList({ personnel: [person()], count: 1,
                     departments: ['网络运维', '系统软件'] });
  const html = els.get('personnel-dept-filter').innerHTML;
  assert.ok(html.includes('全部部门'), '缺默认项');
  assert.ok(html.includes('网络运维') && html.includes('系统软件'), `实际: ${html}`);
});

await check('点卡片进详情：拉 activity、显示身份与分相机活动量', async () => {
  const cards = await reloadList((url) => {
    if (url.includes('/activity')) {
      return { person_id: 'p1', camera_count: 2, raw_rows: 166,
               per_camera: [{ camera_id: 'rnd_04', raw_rows: 30, gid_count: 2,
                              first_ts: Date.now() / 1000 - 3600,
                              last_ts: Date.now() / 1000 - 60 }] };
    }
    if (url.includes('/api/personnel?')) {
      return { personnel: [person()], count: 1, departments: [] };
    }
    return { ...person(), identities: [{ global_id: 'abcd1234', name_confidence: 0.91,
                                         last_camera: 'rnd_04', total_appearances: 40 }] };
  });
  assert.ok(cards.length >= 1, '没有卡片被渲染');
  REQUESTS.length = 0;
  cards[0].dispatch('click');
  await tick(30);
  const detail = els.get('personnel-detail');
  const html = detail.innerHTML;
  assert.ok(detail.classList.contains('active'), '详情面板未激活');
  assert.ok(!els.get('personnel-list-pane').classList.contains('active'), '列表应被隐藏');
  for (const [what, needle] of [['姓名', '张伟'], ['匿名身份', 'abcd1234'],
                                ['相机', 'RND_04'], ['SIM 徽标', '模拟数据'],
                                ['身份数', '>1<'], ['轨迹条数', '166']]) {
    assert.ok(html.includes(needle), `详情缺${what}: ${needle}`);
  }
  assert.ok(REQUESTS.some(r => r.url.includes('/activity')), '未拉 activity');
  assert.ok(!/undefined|NaN/.test(html), `详情含 undefined/NaN: ${html.slice(0, 300)}`);
});

await check('详情里的未知字段不显示 undefined', async () => {
  await reloadList((url) => (url.includes('/activity')
    ? { camera_count: 0, raw_rows: 0, per_camera: [] }
    : (url.includes('/api/personnel?')
      ? { personnel: [person()], count: 1, departments: [] }
      : { ...person(), identities: [] })));
  const cards = els.get('personnel-grid')._cards;
  cards[0].dispatch('click');
  await tick(30);
  const html = els.get('personnel-detail').innerHTML;
  assert.ok(!html.includes('undefined'), `含 undefined: ${html.slice(0, 300)}`);
  assert.ok(html.includes('名下暂无匿名身份'), '空身份应有解释文案');
});

await check('返回列表恢复列表面板', async () => {
  els.get('personnel-detail-back').dispatch('click');
  assert.ok(els.get('personnel-list-pane').classList.contains('active'), '未回到列表');
  assert.ok(!els.get('personnel-detail').classList.contains('active'), '详情未隐藏');
});

// --------------------------------------------------------------------------- //
// P1-1c 视频片段列表 + 播放定位                                                  //
// --------------------------------------------------------------------------- //

/** 一个带视频内坐标的片段（后端 /api/search/person?person_id= 的形状）。 */
function clipAsset(over = {}) {
  return {
    camera_id: 'rnd_08', asset_id: 7, file: 'videos_low/rnd_08.mp4',
    hit_count: 42, loop_factor: 3, position_known: true,
    video_first_ts: 12.5, video_last_ts: 30.25, ...over,
  };
}

/**
 * 进详情页，并按需喂片段/活动量响应。返回详情面板的 innerHTML。
 *
 * 两个可选请求分别可传 Error 表示"该接口失败"（模块会 catch 成 null），
 * 传普通对象表示"成功了但内容为空" —— 这两种状态在界面上必须长得不一样。
 */
async function openDetailWithClips(searchPayload, activityPayload) {
  await reloadList((url) => {
    if (url.includes('/activity')) {
      if (activityPayload instanceof Error) throw activityPayload;
      return activityPayload ?? { camera_count: 0, raw_rows: 0, per_camera: [] };
    }
    if (url.includes('/api/personnel?')) {
      return { personnel: [person()], count: 1, departments: [] };
    }
    if (url.includes('/api/search/person')) {
      if (searchPayload instanceof Error) throw searchPayload;
      return searchPayload;
    }
    return { ...person(), identities: [] };
  });
  const cards = els.get('personnel-grid')._cards;
  cards[0].dispatch('click');
  await tick(30);
  return els.get('personnel-detail').innerHTML;
}

await check('详情按 person_id 检索片段（合并名下全部身份，不是只看第一个 gid）', async () => {
  REQUESTS.length = 0;
  await openDetailWithClips({ assets: [clipAsset()], asset_count: 1, global_ids: ['a', 'b'] });
  const searchReq = REQUESTS.find(r => r.url.includes('/api/search/person'));
  assert.ok(searchReq, '未发起片段检索');
  assert.ok(searchReq.url.includes('person_id=p1'),
    `必须按 person_id 检索（合并全部身份），实际: ${searchReq.url}`);
  assert.ok(!searchReq.url.includes('global_id='), '不该退化成单个 global_id');
});

await check('片段列表渲染相机/时间区间/命中帧数/循环倍数', async () => {
  const html = await openDetailWithClips({ assets: [clipAsset()] });
  for (const [what, needle] of [['相机', 'RND_08'], ['起止秒', '12.5s ~ 30.3s'],
                                ['命中帧数', '42 帧'], ['循环倍数', '循环 ×3']]) {
    assert.ok(html.includes(needle), `缺${what}: ${needle}`);
  }
  assert.ok(!/undefined|NaN/.test(html), `含 undefined/NaN: ${html.slice(0, 300)}`);
});

await check('播放 URL 必须带 asset_id（原片与低清片时长不同，不带就 seek 错帧）', async () => {
  const html = await openDetailWithClips({ assets: [clipAsset()] });
  assert.ok(html.includes('data-clip="/media/rnd_08?asset_id=7"'),
    `片段 URL 应带 asset_id，实际: ${html.slice(0, 400)}`);
});

await check('无视频内坐标时不给播放按钮（老数据不能瞎跳）', async () => {
  const html = await openDetailWithClips({
    assets: [clipAsset({ position_known: false, video_first_ts: null, video_last_ts: null })],
  });
  assert.ok(html.includes('无视频内坐标'), '应标注缺坐标');
  assert.ok(!html.includes('data-clip='), '缺坐标不该给可播放 URL');
});

await check('点击播放：定位到 video_first_ts 并起播', async () => {
  await openDetailWithClips({ assets: [clipAsset()] });
  const detail = els.get('personnel-detail');
  const video = els.get('personnel-video');
  const player = els.get('personnel-player');
  assert.ok(player.hidden === false || player.hidden === undefined || player.hidden === true,
    '播放器元素应存在');

  // 桩件里 innerHTML 只是字符串，模块绑定时拿不到真按钮；直接喂一个到 querySelectorAll
  const btn = new FakeEl('button');
  btn.dataset = { clip: '/media/rnd_08?asset_id=7', start: '12.5', end: '30.25' };
  btn.closest = () => ({ dataset: { camera: 'rnd_08' } });
  detail.querySelectorAll = (sel) => (sel === '.person-clip-play' ? [btn] : []);

  // 重新进详情以触发 bindClipPlayback（它会读 detail.querySelectorAll）
  els.get('personnel-detail-back').dispatch('click');
  els.get('personnel-grid')._cards[0].dispatch('click');
  await tick(30);

  video._readyState = 1;
  video._duration = 120;
  video._currentTime = 0;
  video._played = false;
  player.hidden = true;
  btn.dispatch('click');
  await tick(10);

  assert.equal(video.getAttribute('src'), '/media/rnd_08?asset_id=7', '未设置播放源');
  assert.ok(Math.abs(video.currentTime - 12.5) < 0.01,
    `未定位到 video_first_ts，实际 ${video.currentTime}`);
  assert.equal(player.hidden, false, '播放器应展开');
  assert.ok(video._played, '未被要求起播');
});

await check('seek 目标超过视频时长时夹到末尾（原片/低清片时长不同）', async () => {
  await openDetailWithClips({ assets: [clipAsset({ video_first_ts: 999, video_last_ts: 1000 })] });
  const detail = els.get('personnel-detail');
  const video = els.get('personnel-video');
  const label = els.get('personnel-player-label');
  const btn = new FakeEl('button');
  btn.dataset = { clip: '/media/rnd_08?asset_id=7', start: '999', end: '1000' };
  btn.closest = () => ({ dataset: { camera: 'rnd_08' } });
  detail.querySelectorAll = (sel) => (sel === '.person-clip-play' ? [btn] : []);

  els.get('personnel-detail-back').dispatch('click');
  els.get('personnel-grid')._cards[0].dispatch('click');
  await tick(30);

  video._readyState = 1;
  video._duration = 120;      // 源视频只有 120s，却要求 seek 到 999s
  video._currentTime = 0;
  btn.dispatch('click');
  await tick(10);

  assert.ok(video.currentTime <= 120, `越界 seek 未夹住: ${video.currentTime}`);
  assert.ok(!Number.isNaN(video.currentTime), 'currentTime 变成 NaN');
  assert.ok((label.textContent || '').includes('夹到末尾'),
    `越界时应在标签上说明，实际: ${label.textContent}`);
});

await check('片段检索失败时说明失败，不谎报「没有轨迹」', async () => {
  // 活动量**成功且为空**，只有片段接口失败 —— 这样"没有轨迹"若出现，
  // 只可能来自片段那一节，断言才精确。
  const html = await openDetailWithClips(new Error('boom'),
    { camera_count: 0, raw_rows: 0, per_camera: [] });
  assert.ok(html.includes('片段检索失败'), `应说明检索失败: ${html.slice(0, 400)}`);
  assert.ok(html.includes('活动量读取失败') === false,
    '活动量请求成功，不该说它失败');
  // 档案其余信息仍要正常显示
  assert.ok(html.includes('张伟'), '片段失败不该拖垮整个详情');
});

await check('确实没有片段时才说「没有轨迹」', async () => {
  const html = await openDetailWithClips({ assets: [], asset_count: 0 });
  assert.ok(html.includes('名下身份没有轨迹记录'), `实际: ${html.slice(0, 400)}`);
  assert.ok(!html.includes('片段检索失败'), '有响应就不该说失败');
});

await check('活动量接口失败时也不谎报「没有轨迹」，且统计显示 — 而非 0', async () => {
  const html = await openDetailWithClips({ assets: [] }, new Error('activity down'));
  assert.ok(html.includes('活动量读取失败'), `活动量失败要明说: ${html.slice(0, 500)}`);
  assert.ok(!html.includes('该档案名下身份没有轨迹记录'),
    '接口失败不能被说成"确实没有轨迹"');
  // 0 会让用户以为"查过了，是零"，— 才表示"没拿到"
  const stats = html.slice(html.indexOf('person-stats'), html.indexOf('匿名身份'));
  assert.ok(!stats.includes('>0<') || stats.includes('—'),
    `失败时统计不该显示 0: ${stats}`);
});

await check('删除需两步：第一次只武器化，第二次才发 DELETE', async () => {
  const cards = await reloadList((url) => (url.includes('/activity')
    ? { camera_count: 0, raw_rows: 0, per_camera: [] }
    : (url.includes('/api/personnel?')
      ? { personnel: [person()], count: 1, departments: [] }
      : { ...person(), identities: [] })));
  cards[0].dispatch('click');
  await tick(30);
  const btn = els.get('personnel-detail-delete');
  REQUESTS.length = 0;
  btn.dispatch('click');
  await tick(10);
  assert.equal(REQUESTS.length, 0, '第一次点击不该发请求');
  assert.equal(btn.dataset.armed, '1', '未进入待确认状态');
  assert.ok(btn.textContent.includes('再点一次确认'), `实际文案: ${btn.textContent}`);
  btn.dispatch('click');
  await tick(20);
  const del = REQUESTS.find(r => r.options?.method === 'DELETE');
  assert.ok(del, `未发出 DELETE，实际: ${JSON.stringify(REQUESTS.map(r => [r.url, r.options?.method]))}`);
  assert.ok(del.url.includes('/api/personnel/p1'), del.url);
});

await check('请求失败显示后端错误信息', async () => {
  await reloadList({ personnel: [], count: 0, departments: [] });
  failNext = { error: '数据库连不上' };
  els.get('personnel-reload').dispatch('click');
  await tick(15);
  const html = grid();
  assert.ok(html.includes('加载失败'), `实际: ${html}`);
  assert.ok(html.includes('数据库连不上'), '应显示后端错误信息');
});

await check('列表数据里的 HTML 被转义（防 XSS）', async () => {
  await reloadList({ personnel: [person({ name: '<img src=x onerror=alert(1)>',
                                          department: '</div><script>alert(1)</script>' })],
                     count: 1, departments: [] });
  const html = grid();
  assert.ok(!html.includes('<img src=x onerror'), '姓名未转义 → XSS');
  assert.ok(!html.includes('<script>'), '部门未转义 → XSS');
  assert.ok(html.includes('&lt;img'), '应输出转义后的实体');
});

// --------------------------------------------------------------------------- //
// 注册照上传 + 以图搜人（批次六 P1 交互入口）                                    //
// --------------------------------------------------------------------------- //

const modals = await import('../static/js/modules/modals.js');

await check('详情提供上传注册照入口，选文件后 POST multipart 且带守卫头', async () => {
  await openDetailWithClips({ assets: [] });
  const input = els.get('personnel-photo-input');
  assert.ok(els.get('personnel-photo-upload'), '详情缺上传按钮');
  input.files = [new File([new Uint8Array(8)], 'zhangwei.jpg', { type: 'image/jpeg' })];
  REQUESTS.length = 0;
  input.dispatch('change');
  await tick(40);
  const req = REQUESTS.find(r => r.url.includes('/photos'));
  assert.ok(req, `未发出注册照上传请求: ${JSON.stringify(REQUESTS.map(r => r.url))}`);
  assert.equal(req.options.method, 'POST');
  assert.ok(req.options.body instanceof FormData, '注册照必须走 multipart');
  assert.ok(req.options.body.has('image'), '缺 image 字段');
  assert.equal(req.options.headers['X-Lab-Monitor-Request'], '1', '写请求必须带守卫头');
});

await check('注册照上传失败：按钮恢复可用、选择框清空以便重试', async () => {
  await openDetailWithClips({ assets: [] });
  const input = els.get('personnel-photo-input');
  const btn = els.get('personnel-photo-upload');
  // 桩件的 innerHTML 不反映到 textContent，先按真浏览器的渲染结果初始化
  btn.textContent = '＋ 上传注册照';
  input.files = [new File([new Uint8Array(8)], 'nobody.jpg', { type: 'image/jpeg' })];
  failNext = { error: '图中未检测到人员' };
  try {
    input.dispatch('change');
    await tick(40);
    assert.equal(btn.disabled, false, '失败后按钮必须恢复可用');
    assert.ok((btn.textContent || '').includes('上传注册照'), `按钮未恢复文案: ${btn.textContent}`);
    assert.equal(input.value, '', '选择框应被清空以便重试');
  } finally {
    failNext = null;   // 断言失败也不能把 500 泄漏给后续用例
  }
});

await check('以图搜人：入口常驻、multipart 上传、Top-K 渲染分数与命中态', async () => {
  // 空库也要有入口：身份网格为空时入口消失，用户会以为功能下线了
  scenario = () => ({ ids: [] });
  await modals.openIdentitySearchModal();
  await tick(20);
  let body = els.get('modal-traj-body');
  assert.ok(body.innerHTML.includes('btn-by-image-search'), '空库时以图搜人入口必须还在');

  // 第二次打开会重复 bindByImageSearch：先清掉桩件元素上的旧监听，
  // 否则 change 会触发两次、请求翻倍
  ['btn-by-image-search', 'by-image-input', 'by-image-result'].forEach(k => {
    const t = els.get(k); if (t) t.handlers = {};
  });
  scenario = (url) => {
    if (url.includes('/api/search/by-image')) {
      return { person_count: 1, galleries_compared: 6, threshold: 0.68,
               matches: [
                 { global_id: 'a1b2c3d4', score: 0.812, matched: true,
                   name: '张伟', asset_count: 3 },
                 { global_id: 'e5f6a7b8', score: 0.42, matched: false }] };
    }
    return { ids: ['a1b2c3d4', 'e5f6a7b8'] };
  };
  await modals.openIdentitySearchModal();
  await tick(20);
  body = els.get('modal-traj-body');
  assert.ok(body.innerHTML.includes('a1b2c3d4'), '身份网格应渲染');

  const input = els.get('by-image-input');
  input.files = [new File([new Uint8Array(8)], 'suspect.jpg', { type: 'image/jpeg' })];
  REQUESTS.length = 0;
  input.dispatch('change');
  await tick(30);
  const req = REQUESTS.find(r => r.url.includes('/api/search/by-image'));
  assert.ok(req, '未发出以图搜人请求');
  assert.equal(req.options.method, 'POST');
  assert.ok(req.options.body instanceof FormData && req.options.body.has('image'),
    '必须 multipart 携带 image');
  assert.equal(req.options.headers['X-Lab-Monitor-Request'], '1', '写请求必须带守卫头');
  const box = els.get('by-image-result').innerHTML;
  assert.ok(box.includes('Top1'), '缺 Top1 排名');
  assert.ok(box.includes('81.2%'), `相似度未按百分比显示: ${box.slice(0, 200)}`);
  assert.ok(box.includes('命中') && box.includes('未达阈值'),
    '命中/未达阈值两种状态都要展示');
  assert.ok(box.includes('张伟'), '底库命中的姓名应显示');
  assert.ok(box.includes('a1b2c3d4') && box.includes('e5f6a7b8'),
    'matched=false 的候选也要展示（离线检索不是二值判定）');
});

await check('以图搜人失败：422 后端人话透传到界面', async () => {
  failNext = null;
  // 先清掉上一用例绑定的监听再打开弹窗：重复绑定的第二个 handler 会在
  // failNext 被消耗后也发一次请求，用成功结果覆盖掉失败提示
  ['btn-by-image-search', 'by-image-input', 'by-image-result'].forEach(k => {
    const t = els.get(k); if (t) t.handlers = {};
  });
  scenario = () => ({ ids: ['a1b2c3d4'] });
  await modals.openIdentitySearchModal();
  await tick(20);
  const input = els.get('by-image-input');
  input.files = [new File([new Uint8Array(8)], 'empty.jpg', { type: 'image/jpeg' })];
  failNext = { error: '图中未检测到人员' };
  try {
    input.dispatch('change');
    await tick(30);
    assert.ok(els.get('by-image-result').innerHTML.includes('图中未检测到人员'),
      `应透传后端错误: ${els.get('by-image-result').innerHTML.slice(0, 200)}`);
  } finally {
    failNext = null;
  }
});

/**
 * 跨相机抓拍证据（modals.js loadCrossCameraEvidence）。
 *
 * 这块是把「同一个人出现在多路相机」用真实画面摆出来，两个静默失败要防住：
 *   1. 拼图 URL 拼错 → <img> 404，面板看起来"空空的"却没有任何报错
 *   2. 描述相同的相机（rnd_21/rnd_22 都叫「L2高性能机房04通道西南向北」）不提示
 *      → 用户会把"跨 4 路相机"误读成 4 个不同位置
 */
function snapshotPayload(over = {}) {
  return {
    global_id: 'a1b2c3d4', camera_count: 2, cached: true, skipped: {},
    strip_url: '/identity-snapshots/a1b2c3d4_strip.jpg',
    cameras: [
      { camera: 'rnd_04', desc: 'L2中间走廊东东向西', video_ts: 3.8,
        wall_time: 1789305409.9, bbox: [1, 2, 3, 4], frames: 246,
        url: '/identity-snapshots/a1b2c3d4_rnd_04.jpg' },
      { camera: 'rnd_22', desc: 'L2高性能机房04通道西南向北', video_ts: 21.3,
        wall_time: 1789305410.5, bbox: [5, 6, 7, 8], frames: 58,
        url: '/identity-snapshots/a1b2c3d4_rnd_22.jpg' },
    ],
    ...over,
  };
}

function trajectoryDetailPayload(over = {}) {
  return {
    global_id: 'a1b2c3d4', total_appearances: 1506, last_camera: 'rnd_06',
    trajectory: [{ camera: 'rnd_04', time_str: '17:24:53',
                   end_time_str: '17:27:05', bbox: [1, 2, 3, 4] }],
    ...over,
  };
}

function trajectoryGroupsPayload() {
  return {
    groups: [
      { cameras: ['rnd_04'], enter: 1789305409, exit: 1789305542,
        duration_s: 133, frames: 269, segment_count: 1, multi_view: false },
      { cameras: ['rnd_04', 'rnd_22'], enter: 1789305542, exit: 1789305551,
        duration_s: 9, frames: 6, segment_count: 4, multi_view: true },
    ],
    per_camera: {},
    flicker: { raw_segments: 21, groups: 2, absorbed_segments: 19,
               multi_view_groups: 1, rounds: 3 },
  };
}

await check('跨相机抓拍证据：拼图 + 每相机卡片 + 通行链视图组', async () => {
  scenario = (url) => {
    if (url.includes('/snapshots')) return snapshotPayload();
    if (url.includes('/trajectory')) return trajectoryGroupsPayload();
    return trajectoryDetailPayload();
  };
  await modals.showTrajectoryModal('a1b2c3d4');
  await tick(40);
  const html = els.get('cross-camera-panel').innerHTML;
  assert.ok(html.includes('/identity-snapshots/a1b2c3d4_strip.jpg'),
    `必须渲染服务端拼图: ${html.slice(0, 200)}`);
  assert.ok(html.includes('RND_04') && html.includes('RND_22'), '每路相机都要列出');
  assert.ok(html.includes('L2中间走廊东东向西'), '相机中文描述要展示');
  assert.ok(html.includes('3.8') && html.includes('21.3'), '视频内秒数要展示');
  assert.ok(html.includes('246') && html.includes('58'), '按相机的命中帧数要展示');
  assert.ok(html.includes('墙钟'), '墙钟时间要与视频内时间区分开');
  assert.ok(html.includes('2 路相机'), '要说明覆盖了几路相机');
  // 视图组：多视角必须显式标注，否则两条相机交替会被当成"来回跑"
  assert.ok(html.includes('多视角'), '多视角视图组必须标注');
  assert.ok(html.includes('21 段') && html.includes('2 个视图组'),
    `抖动合并说明缺失: ${html.slice(-300)}`);
  assert.ok(html.includes('已合并'), '合并掉的段数要说明');
});

await check('跨相机抓拍：描述相同的相机要提示同一视点', async () => {
  scenario = (url) => {
    if (url.includes('/snapshots')) {
      const base = snapshotPayload();
      return { ...base, cameras: base.cameras.map(c => ({ ...c, desc: 'L2高性能机房04通道西南向北' })) };
    }
    if (url.includes('/trajectory')) return trajectoryGroupsPayload();
    return trajectoryDetailPayload();
  };
  await modals.showTrajectoryModal('a1b2c3d4');
  await tick(40);
  const html = els.get('cross-camera-panel').innerHTML;
  assert.ok(html.includes('同一视点'),
    `描述重复时必须提示，否则跨相机数量会被误读: ${html.slice(0, 300)}`);
});

await check('跨相机抓拍：无抓拍时给空状态而非静默留白', async () => {
  scenario = (url) => {
    if (url.includes('/snapshots')) return snapshotPayload({ camera_count: 0, cameras: [], strip_url: null });
    if (url.includes('/trajectory')) return trajectoryGroupsPayload();
    return trajectoryDetailPayload();
  };
  await modals.showTrajectoryModal('a1b2c3d4');
  await tick(40);
  const html = els.get('cross-camera-panel').innerHTML;
  assert.ok(html.includes('未生成跨相机抓拍'),
    `无抓拍时必须说明原因: ${html.slice(0, 200)}`);
});

await check('跨相机抓拍：接口失败只影响该区块，不拖垮弹窗', async () => {
  scenario = (url) => {
    if (url.includes('/snapshots')) throw new Error('boom');
    if (url.includes('/trajectory')) return trajectoryGroupsPayload();
    return trajectoryDetailPayload();
  };
  await modals.showTrajectoryModal('a1b2c3d4');
  await tick(40);
  const html = els.get('cross-camera-panel').innerHTML;
  assert.ok(html.includes('加载失败'), `失败要显式提示: ${html.slice(0, 200)}`);
  // 主弹窗内容必须已经渲染（说明抓拍失败没有中断主流程）
  assert.ok(els.get('modal-traj-body').innerHTML.includes('移动路线时序链'),
    '抓拍失败不应中断主弹窗渲染');
});

console.log('');
if (failures.length) {
  console.log(`✗ ${failures.length} 项失败 / ${passed + failures.length} 项`);
  for (const [n, e] of failures) console.log(`  - ${n}: ${e.message.split('\n')[0]}`);
  process.exit(1);
}
console.log(`全部 ${passed} 项通过`);
