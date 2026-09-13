/**
 * 人员档案库模块（批次五 P0-3）
 *
 * 与 openIdentitySearchModal() 的分工：那个看"匿名 gid"，这个看"实名档案"。
 * 之前 header 上「人员档案数据库」按钮打开的其实是匿名 gid 网格 —— 名不副实，
 * 现在把实名档案做成独立弹窗，按钮语义才对得上。
 *
 * 四条实现约束：
 * 1. 筛选与分页**全部交给后端**（GET /api/personnel?q=&department=&source=&limit=&offset=）。
 *    在前端做本地过滤的话，翻页后只会筛当前那一页，用户搜不到人却不知道该怪谁。
 * 2. 工具条写在 index.html 里（不在 JS 里重建）。每次搜索都重画工具条会让输入框丢焦点，
 *    中文输入法尤其明显 —— 只重画 #personnel-grid。
 * 3. 头像走 /personnel-crops/（静态文件挂载），**绝不能用 /stream/**：
 *    并发 MJPEG 上限 5 路（stream_manager.js），且 closeModal 会把任何
 *    src 含 /stream/ 的 img 换成占位图。
 * 4. 不使用内联 onerror：图片加载失败统一用 error 事件委托处理。
 */
import { fetchJson, showToast } from '../utils/api.js';
import { escapeHtml, escapeAttr } from '../utils/formatter.js';
import { openModal, showTrajectoryModal } from './modals.js';

const MODAL_ID = 'personnel-modal';
const PAGE_SIZE = 24;

let state = { q: '', department: '', source: '', offset: 0, total: 0 };
let searchTimer = null;
// 自管代际：modals.js 的 beginModalRequest 没有导出，且它的取消表按弹窗 id 记账，
// 跨模块共用会把两边的请求互相 abort。
let generation = 0;
let activeController = null;

const isSynthetic = (row) => row.source === 'synthetic';

function el(id) {
  return document.getElementById(id);
}

function formatWhen(ts) {
  if (!ts) return '—';
  const diff = Date.now() / 1000 - Number(ts);
  if (diff < 90) return '刚刚';
  if (diff < 3600) return `${Math.round(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.round(diff / 3600)} 小时前`;
  if (diff < 30 * 86400) return `${Math.round(diff / 86400)} 天前`;
  const d = new Date(Number(ts) * 1000);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

/**
 * 头像 URL。**只用后端给的 thumb_url**，不要自己拼 thumb_path：
 * 库里存的是磁盘路径（outputs/personnel_crops/x.jpg），而对外提供文件的是
 * StaticFiles mount /personnel-crops —— 两者拼法不同，前端一旦自己拼就会
 * 全部 404 且静默降级成首字块（这个 bug 已发生过一次）。
 * 兼容 thumb_path：万一连的是旧后端或手写的行，用文件名兜底。
 */
function thumbUrl(row) {
  if (row.thumb_url) return String(row.thumb_url);
  const raw = row.thumb_path;
  if (!raw) return '';
  const name = String(raw).replace(/\\/g, '/').split('/').pop();
  return name ? `/personnel-crops/${encodeURIComponent(name)}` : '';
}

/**
 * 头像 HTML。data-initial 供加载失败时降级用（见 bindAvatarFallback）。
 * 头像文件只有模拟数据会产生，真档案常常没有 —— 缺图必须看起来是设计如此，
 * 而不是像界面坏了。
 */
function avatarHtml(row, sizeClass) {
  const initial = escapeHtml(String((row.name || '?').slice(0, 1)));
  const src = thumbUrl(row);
  if (!src) return `<div class="person-avatar${sizeClass} fallback">${initial}</div>`;
  return `<img class="person-avatar${sizeClass}" src="${escapeAttr(src)}"
    alt="${escapeHtml(row.name || '')}" loading="lazy" data-initial="${escapeAttr(initial)}">`;
}

/** 图片 404/断链时换成首字块。用委托，一次绑好，翻页与重渲染都不用重绑。 */
function bindAvatarFallback() {
  el(MODAL_ID)?.addEventListener('error', (event) => {
    const img = event.target;
    if (!(img instanceof HTMLImageElement) || !img.classList.contains('person-avatar')) return;
    const box = document.createElement('div');
    const isLarge = img.classList.contains('lg');
    box.className = `person-avatar${isLarge ? ' lg' : ''} fallback`;
    box.textContent = img.dataset.initial || '?';
    img.replaceWith(box);
  }, true);   // 图片 error 不冒泡，只能捕获阶段收
}

function syncControls(departments) {
  const select = el('personnel-dept-filter');
  const search = el('personnel-search-input');
  const sourceBox = el('personnel-source-filter');
  if (search && search.value !== state.q) search.value = state.q;
  sourceBox?.querySelectorAll('[data-source]').forEach(btn => {
    btn.classList.toggle('active', (btn.dataset.source || '') === state.source);
  });
  if (!select) return;
  const wanted = departments || [];
  const existing = Array.from(select.options).map(o => o.value).filter(Boolean);
  if (existing.join('|') !== wanted.join('|')) {
    select.innerHTML = '<option value="">全部部门</option>'
      + wanted.map(d => `<option value="${escapeAttr(d)}">${escapeHtml(d)}</option>`).join('');
  }
  if (state.department && !wanted.includes(state.department)) {
    showToast(`部门「${state.department}」下已无人员，筛选已清除`, 'info');
    state.department = '';
  }
  select.value = state.department;
}

function renderPager() {
  const pager = el('personnel-pager');
  if (!pager) return;
  const from = state.total ? state.offset + 1 : 0;
  const to = Math.min(state.total, state.offset + PAGE_SIZE);
  const page = Math.floor(state.offset / PAGE_SIZE) + 1;
  const pages = Math.max(1, Math.ceil(state.total / PAGE_SIZE));
  const prevDisabled = state.offset <= 0;
  const nextDisabled = to >= state.total;
  pager.innerHTML = `
    <button class="action-btn" id="personnel-prev" ${prevDisabled ? 'disabled' : ''}>‹ 上一页</button>
    <span class="person-pager-info">${from}–${to} / 共 ${state.total} 人 · 第 ${page}/${pages} 页</span>
    <button class="action-btn" id="personnel-next" ${nextDisabled ? 'disabled' : ''}>下一页 ›</button>
  `;
  el('personnel-prev')?.addEventListener('click', () => {
    if (!prevDisabled) load({ offset: Math.max(0, state.offset - PAGE_SIZE) });
  });
  el('personnel-next')?.addEventListener('click', () => {
    if (!nextDisabled) load({ offset: state.offset + PAGE_SIZE });
  });
}

function personCard(row) {
  const pid = escapeAttr(row.person_id);
  const nameHtml = escapeHtml(row.name || '未命名');
  const sim = isSynthetic(row)
    ? '<span class="person-badge sim" title="随机生成的演示数据，不代表识别能力">模拟</span>'
    : '';
  return `
    <div class="person-card${sim ? ' is-sim' : ''}" data-person="${pid}"
         role="button" tabindex="0" aria-label="查看 ${nameHtml} 的档案">
      ${avatarHtml(row, '')}
      <div class="person-card-main">
        <div class="person-name-row">
          <span class="person-name">${nameHtml}</span>${sim}
        </div>
        <div class="person-empno">${escapeHtml(row.employee_no || '无工号')}</div>
        <div class="person-dept">${escapeHtml(row.department || '未分配部门')}</div>
        <div class="person-meta">
          <span title="名下匿名身份数">${Number(row.identity_count) || 0} 身份</span>
          <span title="名下身份累计出现记录条数">${Number(row.total_appearances) || 0} 次</span>
          <span title="注册照张数">${Number(row.photo_count) || 0} 照</span>
        </div>
      </div>
      <div class="person-card-side">
        <div class="person-when">${escapeHtml(formatWhen(row.last_seen))}</div>
        <div class="person-cam">${escapeHtml(String(row.last_camera || '').toUpperCase() || '—')}</div>
      </div>
    </div>`;
}

function bindCardOpenings(scope) {
  scope.querySelectorAll('.person-card').forEach(card => {
    const open = () => showPersonDetail(card.dataset.person);
    card.addEventListener('click', open);
    card.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        open();
      }
    });
  });
}

function renderGrid(rows) {
  const grid = el('personnel-grid');
  if (!grid) return;
  if (!rows.length) {
    const filtered = state.q || state.department || state.source;
    grid.innerHTML = `<div class="person-empty">
      <div class="person-empty-title">${filtered ? '没有匹配的人员档案' : '还没有人员档案'}</div>
      <div class="person-empty-sub">${filtered
        ? '换个姓名 / 工号 / 部门再试，或把部门与来源切回"全部"'
        : '运行 <code>scripts/seed_personnel_mock.py --count 40</code> 生成模拟档案，'
          + '或用右上角「新建档案」手工录入实名'}</div>
    </div>`;
    return;
  }
  grid.innerHTML = rows.map(personCard).join('');
  bindCardOpenings(grid);
}

function showListPane() {
  stopPlayback();               // 详情里可能挂着 Range 播放，回列表必须断开
  const detail = el('personnel-detail');
  const listPane = el('personnel-list-pane');
  if (detail) detail.classList.remove('active');
  if (listPane) listPane.classList.add('active');
}

/**
 * 彻底断开回放。只把播放器 hidden 起来是不够的：<video> 带着 src
 * 会继续持有 /media 的 Range 连接（后台持续下载、有声卡时还会出声）。
 * pause + removeAttribute('src') + load() 才是浏览器认可的"断流"写法。
 */
function stopPlayback() {
  const video = el('personnel-video');
  if (video) {
    video.pause();
    video.removeAttribute('src');
    video.load();
  }
  const player = el('personnel-player');
  if (player) player.hidden = true;
}

async function load(patch = {}) {
  state = { ...state, ...patch };
  generation += 1;
  const mine = generation;
  if (activeController) activeController.abort();
  activeController = new AbortController();

  showListPane();
  const grid = el('personnel-grid');
  if (grid) grid.innerHTML = '<div class="person-empty">读取人员档案中…</div>';

  const params = new URLSearchParams({ limit: String(PAGE_SIZE), offset: String(state.offset) });
  if (state.q) params.set('q', state.q);
  if (state.department) params.set('department', state.department);
  if (state.source) params.set('source', state.source);

  try {
    const res = await fetchJson(`/api/personnel?${params}`, { signal: activeController.signal });
    if (mine !== generation || !el(MODAL_ID)?.classList.contains('active')) return;
    state.total = Number(res.count) || 0;
    syncControls(res.departments || []);
    renderGrid(res.personnel || []);
    renderPager();
  } catch (err) {
    if (mine !== generation) return;                 // 已被更新的输入取代，不报错
    if (grid) grid.innerHTML = `<div class="person-empty">加载失败：${escapeHtml(err.message)}</div>`;
  }
}

function gidRow(identity) {
  const gid = escapeAttr(identity.global_id);
  return `
    <div class="person-gid-row" data-gid="${gid}" role="button" tabindex="0">
      <span class="person-gid">#${escapeHtml(identity.global_id)}</span>
      <span class="person-gid-conf">置信 ${escapeHtml(
        typeof identity.name_confidence === 'number'
          ? Math.round(identity.name_confidence * 100) + '%' : '—')}</span>
      <span class="person-gid-cam">${escapeHtml(
        String(identity.last_camera || '').toUpperCase() || '—')}</span>
      <span class="person-gid-count">${Number(identity.total_appearances) || 0} 次</span>
    </div>`;
}

function camRow(camera) {
  return `
    <div class="person-cam-row">
      <span class="person-gid-cam">${escapeHtml(String(camera.camera_id).toUpperCase())}</span>
      <span>${Number(camera.raw_rows) || 0} 条</span>
      <span class="person-cam-span">${escapeHtml(formatWhen(camera.first_ts))}
        → ${escapeHtml(formatWhen(camera.last_ts))}</span>
    </div>`;
}

function personDetailHtml(person, activity, search) {
  const identities = person.identities || [];
  const sim = isSynthetic(person)
    ? '<span class="person-badge sim">模拟数据</span>' : '';
  const rawRows = Number(activity?.raw_rows) || 0;

  const camRows = activity?.per_camera || [];
  // 与片段列表同一条口径：**请求失败**和**确实没有轨迹**必须分开说。
  // 混为一谈时用户会以为"系统查过了，这人没来过"，而其实只是接口挂了。
  const activityEmpty = activity == null
    ? '活动量读取失败（不影响档案其他信息，可点「刷新」重试）。'
    : '该档案名下身份没有轨迹记录。';

  return `
    <div class="person-detail-head">
      <button class="action-btn" id="personnel-detail-back">‹ 返回列表</button>
      <div class="person-detail-id">
        ${avatarHtml(person, ' lg')}
        <div>
          <div class="person-name-row">
            <span class="person-name lg">${escapeHtml(person.name || '未命名')}</span>${sim}
          </div>
          <div class="person-empno">${escapeHtml(person.employee_no || '无工号')}
            · ${escapeHtml(person.department || '未分配部门')}</div>
          ${person.note ? `<div class="person-dept">${escapeHtml(person.note)}</div>` : ''}
        </div>
      </div>
    </div>

    <div class="meta-grid person-stats">
      <div class="meta-card"><div class="label">名下身份</div>
        <div class="val">${identities.length}</div></div>
      <div class="meta-card"><div class="label">轨迹条数</div>
        <div class="val">${activity == null ? '—' : rawRows}</div></div>
      <div class="meta-card"><div class="label">覆盖相机</div>
        <div class="val">${activity == null ? '—' : Number(activity.camera_count) || 0}</div></div>
      <div class="meta-card"><div class="label">注册照</div>
        <div class="val">${Number(person.photo_count) || 0}</div></div>
    </div>

    <div class="person-section-title">匿名身份（点击看跨镜头轨迹）</div>
    ${identities.length
      ? identities.map(gidRow).join('')
      : '<div class="person-empty-sub">名下暂无匿名身份。在「ReID 身份库」里给某个 gid 命名即可归到本档案。</div>'}

    ${clipsSection(search)}

    <div class="person-section-title">分相机活动量</div>
    <div class="person-note">
      条数是 appearance 原始行数。素材循环播放会让同一段像素被反复记录，
      所以它不等于"独立出现次数" —— 轨迹弹窗里会标出 loop 倍数。
    </div>
    ${camRows.length
      ? camRows.map(camRow).join('')
      : `<div class="person-empty-sub">${activityEmpty}</div>`}

    <div class="person-danger-zone">
      <button class="action-btn" id="personnel-detail-delete"
              style="color: var(--danger); border-color: rgba(220, 38, 38, 0.3);">
        删除此档案（解除身份绑定）
      </button>
    </div>
  `;
}

/**
 * 一个资产渲染成多行 —— **每段到访一行**。
 * 后端把同一视频里的不连续到访保留在 segments[]（每次进入/离开各一段），
 * 之前只画一行并用全资产最早~最晚时间当区间，中间没出现的时段也被当成"在场"，
 * 播放也只能跳到最早一段。旧行为对"一天来过三次"的人会显示成一次连续长驻。
 * 旧数据没有 segments 或段缺视频内坐标时，退化为"无视频内坐标"。
 */
function assetClipRows(asset) {
  const cam = escapeHtml(String(asset.camera_id || '').toUpperCase());
  const segments = Array.isArray(asset.segments) && asset.segments.length
    ? asset.segments
    : [{ video_enter_ts: asset.video_first_ts, video_exit_ts: asset.video_last_ts }];
  const loop = Number(asset.loop_factor) > 1
    ? `<span class="person-badge loop" title="素材循环播放，同一段像素被记录多遍">循环 ×${escapeHtml(String(asset.loop_factor))}</span>`
    : '';
  const frames = `<span class="person-clip-frames">${Number(asset.hit_count) || 0} 帧</span>`;
  // asset_id 必须带上：同一相机在库里有"原片 + 低清转码"两行，时长不同
  // （实测 239.12 vs 239.64），而 video_ts 是按低清片标定的。不带 asset_id
  // 就可能拿到原片，seek 会差几十毫秒 —— 在循环素材里就是错帧。
  const url = `/media/${encodeURIComponent(asset.camera_id)}${asset.asset_id != null
    ? `?asset_id=${encodeURIComponent(asset.asset_id)}` : ''}`;
  return segments.map((seg, i) => {
    const start = seg.video_enter_ts;
    const end = seg.video_exit_ts;
    const known = start != null && start !== undefined;
    // 循环倍数与帧命中是资产级指标，只标在第一段，避免每行都刷一遍
    const side = i === 0 ? `${loop}${frames}` : '';
    const revisitTag = i === 0 ? '' :
      `<span class="person-clip-note">第 ${i + 1} 段</span>`;
    const action = known
      ? `<button class="action-btn person-clip-play" data-clip="${escapeAttr(url)}"
           data-start="${escapeAttr(String(Number(start)))}"
           data-end="${escapeAttr(String(Number(end)))}">播放</button>`
      : '<span class="person-clip-note">无视频内坐标</span>';
    return `
      <div class="person-clip" data-camera="${escapeAttr(asset.camera_id || '')}">
        <div class="person-clip-main">
          <span class="person-clip-cam">${cam}</span>
          <span class="person-clip-time">${known
            ? `${escapeHtml(Number(start).toFixed(1))}s ~ ${escapeHtml(Number(end ?? start).toFixed(1))}s`
            : '墙钟时间见活动量'}</span>
          ${revisitTag}
        </div>
        <div class="person-clip-side">
          ${side}
          ${action}
        </div>
      </div>`;
  }).join('');
}

function clipsSection(search) {
  const assets = search?.assets || [];
  const total = assets.reduce((sum, a) => sum + (Number(a.hit_count) || 0), 0);
  // 两种"空"必须区分，否则用户会把故障当成事实：
  //   search == null → 请求失败（说明检索坏了，可重试）
  //   search 有值且无资产 → 确实没有轨迹（相机下线 / 名下身份无记录）
  const emptyText = search == null
    ? '片段检索失败（档案其余信息不受影响，可点「刷新」重试）。'
    : '名下身份没有轨迹记录，或相机已下线。';
  return `
    <div class="person-section-title">视频片段（点击播放定位）</div>
    <div class="person-note">
      坐标是该身份在<b>源视频内</b>的秒数（已按循环折叠）。同一视频里每段
      不连续到访各占一行。播放走 <code>/media</code> 的低清转码片 ——
      原片体积约 30 倍，且时间轴与轨迹坐标不是同一条，用原片 seek 会错帧。
    </div>
    <div class="person-player" id="personnel-player" hidden>
      <video id="personnel-video" controls preload="metadata"></video>
      <div class="person-player-bar">
        <span id="personnel-player-label"></span>
        <button class="action-btn" id="personnel-player-close">收起播放器</button>
      </div>
    </div>
    ${assets.length
      ? assets.map(assetClipRows).join('')
      : `<div class="person-empty-sub">${emptyText}</div>`}
    ${assets.length
      ? `<div class="person-note">共 ${assets.length} 个视频文件 · ${total} 帧命中${
          search?.truncated ? '（轨迹超上限被截断，建议缩小时间窗）' : ''}</div>`
      : ''}
  `;
}

/**
 * 播放定位。两个要点：
 *  1. seek 必须**等 loadedmetadata** —— metadata 未就绪时设 currentTime 会被忽略，
 *     表现是"点了播放却从头开始"，且不报错。
 *  2. 短视频循环素材里 start 可能大于视频实际时长（原片/低清片时长不同），
 *     越界 seek 会被浏览器夹到末尾；这里显式夹一次并给出提示，避免用户以为坏了。
 */
function bindClipPlayback(scope) {
  const player = scope.querySelector('#personnel-player');
  const video = scope.querySelector('#personnel-video');
  const label = scope.querySelector('#personnel-player-label');
  const close = scope.querySelector('#personnel-player-close');
  if (!player || !video) return;

  scope.querySelectorAll('.person-clip-play').forEach(btn => {
    btn.addEventListener('click', () => {
      const url = btn.dataset.clip;
      const start = Number(btn.dataset.start) || 0;
      const cam = btn.closest('.person-clip')?.dataset.camera || '';
      if (video.getAttribute('src') !== url) video.setAttribute('src', url);
      player.hidden = false;
      const seek = () => {
        const duration = video.duration;
        const target = Number.isFinite(duration) && duration > 0
          ? Math.min(start, Math.max(0, duration - 0.5))
          : start;
        try { video.currentTime = target; } catch { /* metadata 未就绪，忽略 */ }
        if (label) {
          label.textContent = `${cam.toUpperCase()} · 定位 ${target.toFixed(1)}s`
            + (target < start ? `（源视频仅 ${duration.toFixed(1)}s，已夹到末尾）` : '');
        }
        video.play().catch(() => { /* 自动播放被拦，用户可手点播放键 */ });
      };
      if (video.readyState >= 1) seek();
      else {
        video.addEventListener('loadedmetadata', seek, { once: true });
        video.load();
      }
    });
  });

  close?.addEventListener('click', () => {
    video.pause();
    video.removeAttribute('src');
    video.load();               // 断开分片连接，别把 Range 请求挂在那
    player.hidden = true;
  });
}

/**
 * 档案详情：档案字段 + 名下匿名身份 + 分相机活动量 + 视频片段。
 * 片段列表按 person_id 聚合（合并名下全部 gid），不是只看第一个身份 ——
 * 一个人常被拆成多个匿名身份，只看一个会漏掉大半轨迹。
 */
async function showPersonDetail(personId) {
  if (!personId) return;
  const listPane = el('personnel-list-pane');
  const detail = el('personnel-detail');
  if (!detail || !listPane) return;
  generation += 1;                                  // 作废在飞的列表请求
  if (activeController) activeController.abort();
  detail.innerHTML = '<div class="person-empty">读取档案中…</div>';
  detail.classList.add('active');
  listPane.classList.remove('active');

  const pid = encodeURIComponent(personId);
  try {
    // 三个请求并发：档案字段 / 分相机活动量 / 视频片段。
    // 片段检索按 person_id（合并名下全部 gid），失败不阻断详情 ——
    // 它只是详情的一节，不该让整页打不开。
    const [person, activity, search] = await Promise.all([
      fetchJson(`/api/personnel/${pid}`),
      fetchJson(`/api/personnel/${pid}/activity`).catch(() => null),
      // 片段检索失败不阻断详情：它只是其中一节，传 null 让该节显示"检索失败"
      fetchJson(`/api/search/person?person_id=${pid}`).catch(() => null),
    ]);
    if (!detail.classList.contains('active')) return;
    detail.innerHTML = personDetailHtml(person, activity, search);
  } catch (err) {
    detail.innerHTML = `<div class="person-empty">读取档案失败：${escapeHtml(err.message)}</div>`;
    return;
  }

  bindClipPlayback(detail);

  detail.querySelectorAll('[data-gid]').forEach(row => {
    const go = () => showTrajectoryModal(row.dataset.gid);
    row.addEventListener('click', go);
    row.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); go(); }
    });
  });
  el('personnel-detail-back')?.addEventListener('click', showListPane);
  el('personnel-detail-delete')?.addEventListener('click', (event) => {
    const name = detail.querySelector('.person-name')?.textContent || '';
    deletePerson(personId, name, event.target);
  });
}

/** 两步确认：删除会解除名下身份绑定，误删要重新命名，代价不小。 */
async function deletePerson(personId, name, btn) {
  if (!btn) return;
  if (btn.dataset.armed !== '1') {
    btn.dataset.armed = '1';
    btn.textContent = `再点一次确认删除「${name}」`;
    setTimeout(() => {
      if (btn.isConnected && btn.dataset.armed === '1') {
        delete btn.dataset.armed;
        btn.textContent = '删除此档案（解除身份绑定）';
      }
    }, 4000);
    return;
  }
  btn.disabled = true;
  try {
    await fetchJson(`/api/personnel/${encodeURIComponent(personId)}`, { method: 'DELETE' });
    showToast(`已删除档案「${name}」，名下身份回到匿名状态`, 'success');
    showListPane();
    await load({ offset: 0 });
  } catch (err) {
    showToast(`删除失败：${err.message}`, 'error');
    btn.disabled = false;
    delete btn.dataset.armed;
    btn.textContent = '删除此档案（解除身份绑定）';
  }
}

/** 新建档案：复用人已有的后端 POST /api/personnel，不引新端点。 */
async function createPerson() {
  const name = window.prompt('人员姓名（必填）');
  if (name === null) return;
  const trimmed = name.trim();
  if (!trimmed) { showToast('姓名不能为空', 'warning'); return; }
  const employeeNo = (window.prompt('工号（可留空）') || '').trim();
  const department = (window.prompt('部门（可留空）') || '').trim();
  try {
    const res = await fetchJson('/api/personnel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: trimmed,
        employee_no: employeeNo || null,
        department: department || null,
      }),
      timeoutMs: 8000,
    });
    showToast(`已创建档案「${res.name || trimmed}」`, 'success');
    await load({ offset: 0, q: trimmed });
  } catch (err) {
    showToast(`创建失败：${err.message}`, 'error');
  }
}

export function openPersonnelModal() {
  openModal(MODAL_ID);
  showListPane();
  load({ offset: 0 });
}

/** 在 app.js 的 bindEvents() 里调用一次。 */
export function initPersonnelPanel() {
  bindAvatarFallback();
  const search = el('personnel-search-input');
  search?.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => load({ q: search.value.trim(), offset: 0 }), 250);
  });
  el('personnel-dept-filter')?.addEventListener('change', (event) => {
    load({ department: event.target.value, offset: 0 });
  });
  el('personnel-source-filter')?.addEventListener('click', (event) => {
    const btn = event.target.closest('[data-source]');
    if (btn) load({ source: btn.dataset.source || '', offset: 0 });
  });
  el('personnel-reload')?.addEventListener('click', () => load({}));
  el('personnel-create')?.addEventListener('click', () => { createPerson(); });
  // 关闭时作废在飞请求并断开回放：旧响应会覆盖下次打开的列表，
  // 不断流的话 <video> 会带着 src 在弹窗背后继续拉 /media 分片
  el(MODAL_ID)?.addEventListener('modal:closed', () => {
    generation += 1;
    if (activeController) activeController.abort();
    clearTimeout(searchTimer);
    stopPlayback();
  });
}
