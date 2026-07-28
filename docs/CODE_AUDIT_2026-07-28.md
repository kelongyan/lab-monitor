# Lab-Monitor 全栈代码审计与缺陷修复清单

> - 审计日期：2026-07-28
> - 审计范围：Python 后端、视频处理流水线、ReID、告警状态机、持久化、FastAPI、WebSocket、原生 JavaScript 前端、CSS、PowerShell 启停脚本与运行配置
> - 当前状态：41 组确定性缺陷与 7 组条件性风险已完成代码修复和回归验证
> - 说明：本文档是后续修复工作的唯一编号台账。修复提交、测试记录和评审说明应引用对应问题编号。

---

## 1. 审计结论

本轮共确认 41 组确定性缺陷，并记录 7 组需要在生产环境持续关注的条件性风险。

最严重的问题不是单纯的页面显示异常，而是以下核心闭环在审计基线中无法可靠成立。当前工作区已按本文编号全部修复，并完成单元、API、浏览器和真实数据迁移验证：

1. 跨摄像头通行校准没有获得真正的跨摄样本，现有统计已被同摄像头自循环数据污染。
2. ReID 歧义匹配可被注册兜底逻辑绕过，可能把不同人员合并成同一个 Global ID。
3. 人员出现在错误摄像头时也会删除告警监听，可能漏掉真正的失踪告警。
4. 摄像头离线状态和视频输出不一致，断流后可能继续展示最后一帧。
5. 拓扑配置同时存在持久化 XSS、缺少鉴权、先写文件后校验等问题。
6. 前端历史回填和 WebSocket 重放没有去重，刷新或重连会重复告警、计数和响铃。

### 1.1 优先级定义

| 优先级 | 定义 | 处理要求 |
| --- | --- | --- |
| P0 | 安全漏洞、核心识别/告警结果错误、可能造成监控失真或配置损坏 | 阻断其他功能开发，优先修复并做专项回归 |
| P1 | 会造成单路监控失效、数据错误、资源持续泄漏或重要功能不可用 | P0 完成后立即处理，必须补最小自动化测试 |
| P2 | 边界条件、性能、可维护性、交互一致性和部署可靠性问题 | 分批修复，纳入常规回归 |

### 1.2 审计基线与修复后运行数据

- 审计基线：`outputs/transit_stats.json` 只有同摄像头自循环；修复后加载阶段会清理自循环、非拓扑边和异常样本，新样本仅能由合法跨摄边产生。
- 审计基线：JSONL 2775 行、SQLite 1732 条。迁移前已备份为 `outputs/lab_monitor.pre-alert-migration.2026-07-28.db`；幂等迁移新增 1042 条，修复后 SQLite 为 2774 条唯一事件。
- 682 条缺少 `alert_id` 的旧 JSONL 事件已按规范化内容生成稳定 `legacy_<sha256>` ID；第二次迁移新增 0 条；1 条完全重复记录按同一事件合并；原 JSONL 未改写。
- 审计基线：SQLite 有 106 条身份元数据但无法恢复特征。修复后身份主特征、Feature Bank、`feature_space` 和完整 appearance 轨迹均可跨重启恢复；旧元数据继续保留，但不会伪造特征进入 gallery。
- 审计基线：`outputs/screenshots/` 为空。修复后新告警在可取得帧时以 `alert_id` 原子保存 JPEG 并返回 `screenshot_url`；无帧时明确显示无快照，不再用当前直播冒充历史画面。
- 最终真实数据核对：SQLite `COUNT(*)=2774`、`COUNT(DISTINCT alert_id)=2774`；历史 API `total=2774`；CSV 解析后 2774 行；offset 分页无重叠；`cam_03` 过滤共 812 条且样本 camera 全部正确。
- 最终 Playwright 回归覆盖 800x600、1280x720、1920x1080、焦点视频带左侧日志、200 条 DOM 上限、告警去重、无快照详情、ROI/弹窗竞态、轮询不重入和键盘可访问性。

---

## 2. P0 缺陷

### P0-01 拓扑配置可形成持久化 DOM XSS

- **状态**：已完成
- **涉及文件**：
  - `static/js/modules/modals.js:299-308`
  - `static/js/modules/modals.js:462-487`
  - `server.py:172-186`
- **根因**：拓扑节点 ID 通过模板字符串直接插入 SVG，并最终赋值给 `innerHTML`。服务端只检查请求顶层是 JSON 对象，没有校验 camera ID 白名单和字符范围，随后将内容持久化到 `config/topology.json`。
- **攻击链**：未授权客户端 POST 恶意拓扑 -> 服务端持久化 -> 操作员打开拓扑弹窗 -> 恶意 SVG/事件属性进入 DOM -> 同源脚本执行。
- **影响**：攻击者可篡改大屏、读取同源告警和身份数据、调用同源写接口，甚至伪造监控状态。
- **复现思路**：向拓扑起点或终点字段写入能够闭合 `<text>` 节点的 SVG 片段，再打开拓扑弹窗，检查是否生成非预期 DOM 节点或执行事件处理器。
- **修复建议**：
  1. 前端禁止将外部值拼入 SVG `innerHTML`，使用 `createElementNS()` 和 `textContent` 创建节点。
  2. 服务端 camera ID 只能来自 `sources.json` 或运行中 pipeline 白名单。
  3. 对拓扑请求建立明确 schema，拒绝控制字符、HTML、空白 ID 和过长值。
  4. 为视频、身份、ROI、拓扑写接口增加认证与授权。
- **验收标准**：
  - 恶意 camera ID 请求返回 400，磁盘和运行时拓扑不发生变化。
  - 前端所有拓扑标签均通过文本节点渲染。
  - 加入服务端 schema 测试和前端 XSS 回归用例。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `src/topology.py`
  - `server.py`
  - `static/js/modules/modals.js`
  - `tests/test_topology.py`
- 根因修复摘要：拓扑节点和边标签改用 `createElementNS()` 与 `textContent` 创建，外部值不再进入 SVG `innerHTML`；服务端同时以 `sources.json` 摄像头集合为白名单，拒绝控制字符、HTML 字符、未知 ID、自循环和非法数值。
- 测试命令：
  - `python -m unittest -v tests.test_topology`
  - `node --check static/js/modules/modals.js`
- Playwright/API 验证：Playwright 拦截 `/api/topology` 注入 `</text><script>...` 后，恶意内容仅作为文本显示，弹窗内 `script` 节点数为 0，执行探针为 `false`；API 同类输入返回 400。
- 剩余风险：远程部署仍必须使用 HTTPS 保护 Basic 凭据，相关要求已写入 RISK-01 修复记录。
- 状态：已完成

### P0-02 跨摄像头通行校准完全失效并污染统计

- **状态**：已完成
- **涉及文件**：`src/pipeline.py:93-95`、`src/pipeline.py:358-367`、`src/pipeline.py:396-398`
- **根因**：`_leave_times` 属于每个 `CameraPipeline` 实例。人员从 A 离开时只写入 A 的字典，到达 B 后却从 B 的私有字典读取，因此无法取得 A 的离开时间。
- **附加问题**：同一人员在同一视频循环或重新出现时，会从当前 pipeline 的 `_leave_times` 读取记录并写成 `cam_A→cam_A`，造成现有校准文件污染。
- **影响**：
  - 真正的 `cam_01→cam_02` 等跨摄时间永远无法积累。
  - `TransitCalibrator` 可能使用错误自循环样本覆盖静态时间窗口。
  - 前端“AI 自适应校准”展示与业务事实不符。
- **复现步骤**：
  1. 在 A pipeline 调用 `_on_person_leave(gid)`。
  2. 在 B pipeline 对同一 gid 调用 `_record_arrival(gid)`。
  3. B 的 `_leave_times` 为空，不会产生 A→B 数据。
- **修复建议**：
  1. 将离开记录移到共享、线程安全的 `TransitCalibrator` 或独立 `TransitStateStore`。
  2. 记录键至少包含 `global_id`、来源摄像头和离开时间。
  3. 到达时只接受拓扑中真实存在且来源与目标不同的边。
  4. 修复后清理现有自循环统计，避免错误样本继续生效。
- **验收标准**：
  - A 离开、B 到达可稳定产生 A→B 样本。
  - 同摄像头重新出现不会记录 transit。
  - 并发多摄像头测试下无重复消费或串人。
  - `transit_stats.json` 只包含合法拓扑边。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `src/calibrator.py`
  - `src/pipeline.py`
  - `src/topology.py`
  - `main.py`
  - `tests/test_core_logic.py`
- 根因修复摘要：离开状态迁移到所有 pipeline 共享的 `TransitCalibrator`，记录包含 Global ID、来源 camera、允许到达 camera 集合和时间戳；到达操作在同一把锁内校验合法拓扑边并原子消费，因此同一离开事件只产生一条样本。
- 测试命令：
  - `python -m unittest -v tests.test_core_logic`
  - `python -m unittest -v`
- Playwright/API 验证：A 离开/B 到达产生 A→B 样本；同摄和非预期 camera 不消费；8 线程同时到达只有一次成功。加载时自动过滤自循环、非拓扑边和异常样本。
- 数据清理：`outputs/transit_stats.json` 中 4 组自循环污染统计已清空为 `{}`；原始数据备份为 `outputs/transit_stats.invalid-self-loops.2026-07-28.json`，可恢复。
- 剩余风险：当前真实拓扑的 10000/30000 秒配置仍需业务确认，本修复未擅自调整。
- 状态：已完成

### P0-03 Ratio Test 被注册兜底绕过

- **状态**：已完成
- **涉及文件**：`src/reid_validator.py:80-99`、`src/pipeline.py:271-283`、`src/identity_store.py:139-184`
- **根因**：`ReIDValidator` 使用 Ratio Test 拒绝 Top-1 和 Top-2 过近的歧义匹配，但缓冲区满后，pipeline 仍调用 `register_if_new()`。后者只检查最高相似度是否超过阈值，不执行 Ratio Test。
- **运行复现**：构造两个高度相似身份和位于两者中间的 query，Validator 返回 `None`，`register_if_new()` 却返回其中一个已有 ID。
- **影响**：不同人员可能被永久合并到同一个 Global ID，后续轨迹、告警、通行统计和身份档案全部串人。
- **修复建议**：
  1. `register_if_new()` 与 Validator 共用同一匹配实现和 Ratio Test 规则。
  2. 区分“完全无匹配”和“存在歧义”两种结果。
  3. 歧义时继续积累特征、延迟决策或注册新身份，不能静默选择 Top-1。
  4. 返回结构化匹配结果，而不是只返回 `(gid, is_new)`。
- **验收标准**：
  - Top-1/Top-2 比值超过歧义阈值时不得归并已有身份。
  - 单一清晰匹配仍可正常复用已有 ID。
  - 并发注册测试不产生重复身份或错误合并。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `src/identity_store.py`
  - `src/reid_validator.py`
  - `src/pipeline.py`
  - `tests/test_core_logic.py`
- 根因修复摘要：`register_if_new()` 复用 `match_feature_detailed()` 的阈值与 Ratio Test，并返回 `matched`、`created`、`ambiguous`、`invalid` 结构化结果；pipeline 遇到歧义时不确认、不归并、不注册，继续使用滚动缓冲积累特征。
- 测试命令：
  - `python -m unittest -v tests.test_core_logic.IdentityResolutionTests`
- Playwright/API 验证：两个高度相似身份的中间向量返回 `ambiguous` 且身份数不变；清晰 Top-1 正常复用；8 线程注册同一新特征只创建一个身份。
- 剩余风险：长期无法消歧的轨迹在离开前可能没有 Global ID，这是避免串人的明确保守策略，后续可增加人工复核队列。
- 状态：已完成

### P0-04 非预期摄像头出现会提前取消告警监听

- **状态**：已完成
- **涉及文件**：`src/alerter.py:163-176`
- **根因**：`resolve()` 在检查 `seen_camera` 是否属于 `expected_cameras` 之前就执行 `_watches.pop(global_id)`。
- **运行复现**：对期待 `cam_02` 的 gid 调用 `resolve(gid, "cam_99")`，返回 `False`，但 watch 已被删除。
- **影响**：追踪抖动、同摄像头重新识别、错误 ReID 或异常路径出现一次，就可能使真正的超时告警永远不再触发。
- **修复建议**：
  1. 在锁内先读取 entry 并判断命中，再决定是否删除。
  2. 非预期摄像头应保留原 watch，或明确进入“异常跳转”状态。
  3. 如需支持多条候选路线，应记录各路线状态，避免单个字典项覆盖全部路径。
- **验收标准**：
  - 非预期摄像头返回 `False` 且 watch 继续存在。
  - 正确摄像头出现后 watch 被删除并返回 `True`。
  - 非预期路径是否产生独立告警由明确测试覆盖。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `src/alerter.py`
  - `tests/test_core_logic.py`
- 根因修复摘要：`resolve()` 在锁内先读取并判断 camera，只有命中 `expected_cameras` 才删除 watch；非预期 camera 只记录警告并保留原监听。
- 测试命令：
  - `python -m unittest -v tests.test_core_logic.AlertStateTests`
- Playwright/API 验证：`cam_c` 对期待 `cam_b` 的 watch 返回 `False` 且状态仍存在，随后 `cam_b` 返回 `True` 并移除状态。
- 剩余风险：非预期路径当前不额外产生独立告警，保持既有产品行为，仅确保预期路径监听不会丢失。
- 状态：已完成

### P0-05 拓扑保存失败会破坏磁盘配置和运行时图

- **状态**：已完成
- **涉及文件**：`server.py:172-189`、`src/topology.py:68-86`
- **根因**：接口先将原始请求写入配置文件，然后才调用 `update_config()` 做字段访问和数字转换；`update_config()` 又先清空 `_graph` 再逐项构造。
- **运行复现**：POST `{"cam_01":[{}]}`，接口因缺少 `next` 返回 500，但错误 JSON 已写入磁盘，运行时图可能已被清空或部分更新。
- **影响**：一次错误输入即可让实时告警拓扑失效，并导致下次启动加载失败。
- **修复建议**：
  1. 使用 Pydantic 模型完整校验请求。
  2. 在临时对象中构建完整拓扑，成功后一次性替换 `_graph`。
  3. 使用临时文件加原子替换持久化，不能直接覆盖目标文件。
  4. 移除 server 和 `CameraTopology.update_config()` 的重复写盘。
- **验收标准**：
  - 任意非法请求均返回 400。
  - 非法请求后磁盘文件哈希和运行时拓扑保持不变。
  - 合法请求只执行一次原子写入。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `src/topology.py`
  - `server.py`
  - `tests/test_topology.py`
- 根因修复摘要：由 `CameraTopology.update_config()` 统一执行完整 schema 校验、临时图构建、同目录临时文件写入、`fsync`、`os.replace` 和运行时图替换；API 移除重复直写。整个提交过程受同一把 `RLock` 串行化，避免并发更新造成磁盘和内存版本错位。
- 测试命令：
  - `python -m unittest -v tests.test_topology`
- Playwright/API 验证：缺字段、未知 ID、XSS、自循环、零耗时和错误数值类型均返回 400；逐例校验请求前后文件 SHA-256 与 `to_dict()` 完全不变。8 线程并发更新后磁盘 JSON 与运行时图一致，且无临时文件残留。
- 剩余风险：磁盘原子替换依赖目标目录与临时文件位于同一文件系统，当前实现已固定为同目录创建。
- 状态：已完成

### P0-06 摄像头离线状态和视频输出不一致

- **状态**：已完成
- **涉及文件**：`src/frame_hub.py:56-65`、`src/frame_hub.py:86-95`、`src/frame_hub.py:101-135`
- **根因**：
  - `mark_offline()` 对尚未创建状态的 camera 直接返回。
  - `get_jpeg()` 只检查是否存在历史帧，不检查 `is_online`。
- **运行复现**：
  - 从未推帧的 camera 调用 `mark_offline()` 后，`get_status()` 返回空列表。
  - camera 推一帧后标记离线，`get_jpeg()` 仍返回 JPEG。
- **影响**：首次 RTSP 连接失败的通道从大屏消失；断流通道继续显示旧画面，容易被误认为实时视频。
- **修复建议**：
  1. 初始化 pipeline 时显式注册所有 camera。
  2. `mark_offline()` 使用 `_ensure_camera()`。
  3. 离线时清除视频缓存，或让 `get_jpeg()` 返回 `None` 以展示离线占位帧。
  4. 在状态中增加 `frame_age_seconds`，前端对超时帧做二次判断。
- **验收标准**：
  - 首次连接失败的 camera 仍显示为离线。
  - 断流后不能继续输出历史画面。
  - 重连成功后可恢复实时流和状态。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：`main.py`、`src/frame_hub.py`、`src/pipeline.py`、`static/js/modules/grid.js`、`tests/test_frame_hub.py`
- 根因修复摘要：启动 pipeline 前显式注册所有 camera；`mark_offline()` 会创建缺失状态并同步清空帧/JPEG；`get_jpeg()` 拒绝为离线 camera 返回历史帧；状态接口增加 `frame_age_seconds` 供前端识别过期画面。
- 测试命令：`python -m unittest -v tests.test_frame_hub`
- Playwright/API 验证：camera 状态数组变空时，旧通道和旧 MJPEG 节点被移除并显示空态；单元测试覆盖首次失败可见、离线无旧 JPEG，以及后续推帧恢复。
- 剩余风险：RTSP 底层连接超时仍依赖 OpenCV/FFmpeg 配置，另由 RISK-05 跟踪。
- 状态：已完成

---

## 3. P1 缺陷

### P1-01 流水线异常会永久杀死单路线程

- **状态**：已完成
- **涉及文件**：`src/pipeline.py:120-173`
- **根因**：`run()`、`_run_file()` 和 `_run_rtsp()` 缺少统一异常边界，Capture 释放也不在 `finally` 中。
- **影响**：ROI 数据异常、模型异常、OpenCV 错误等均可能让单路监控永久停止，同时保留最后的 ONLINE 状态。
- **修复建议**：用 `try/except/finally` 包住单次连接和读帧循环；标记 `PIPELINE_ERROR`；RTSP 按策略重连；本地文件错误应明确退出并上报。
- **验收标准**：注入处理异常后 Capture 被释放、状态变为错误、错误日志包含堆栈，RTSP 可恢复。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/pipeline.py`、`tests/test_frame_hub.py`
- 根因修复摘要：为 pipeline 顶层、本地文件和每次 RTSP 连接建立异常边界，Capture 在 `finally` 中释放；本地文件异常标记 `PIPELINE_ERROR`，RTSP 读取异常记录堆栈后进入重连循环。
- 测试证据：`test_file_processing_exception_releases_capture_and_marks_error` 与 `test_rtsp_read_exception_reconnects_and_releases_each_capture` 均通过。
- 剩余风险：底层解码器若在 C 扩展内永久阻塞，Python 异常边界无法立即抢占；RTSP 超时和停止兜底由 RISK-05/P2-15 控制。
- 状态：已完成

### P1-02 WebSocket 首连和重连重复告警

- **涉及文件**：`static/js/app.js:125-135`、`server.py:472-475`、`static/js/modules/websocket.js:19-40`、`static/js/modules/websocket.js:101-133`
- **根因**：页面先加载 REST 历史，随后 WebSocket 又重放 broadcaster 最近记录；前端没有按 `alert_id` 去重。
- **运行复现**：Playwright 页面已有历史时连接含三条 recent 的 WebSocket，列表前六条呈三条正序加三条逆序重复，累计计数增加 3。
- **影响**：刷新、临时断网或服务重连都会重复显示、重复计数和重复播放声音。
- **修复建议**：前端维护有界 `alert_id Set`；服务端增加事件游标或 `since` 参数；重连只补发缺失事件。
- **验收标准**：同一 `alert_id` 无论通过 REST、首连重放或重连重放，页面只出现一次且只计数一次。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/alerter.py`、`server.py`、`static/js/modules/websocket.js`、`static/js/app.js`、`static/js/modules/modals.js`、`tests/test_persistence_alerts.py`
- 根因修复摘要：前端统一在 `addAlert()` 入口按 `alert_id` 去重，使用容量 1000 的有界 Set 和顺序队列；REST 回填、首连 recent、重连补发和实时消息全部经过同一入口。WebSocket 连接携带最后接收 ID 的 `since` 游标，服务端只返回游标后的 recent；游标不在窗口内时由前端 tombstone 继续兜底。
- 测试证据：`test_recent_after_returns_only_missing_events` 通过；Playwright 将同一 ID 通过历史和实时入口重复注入，第一次接收、第二次拒绝，列表、焦点日志、累计统计和 badge 均只变化一次。
- 剩余风险：recent 仍是内存窗口，客户端离线超过窗口容量时应通过 REST 历史补齐；当前去重集合有界，超过 1000 个事件后最旧 tombstone 会淘汰。
- 状态：已完成

### P1-03 ROI 异步请求可把 A 的围栏保存到 B

- **涉及文件**：`static/js/modules/roi.js:46-50`、`static/js/modules/roi.js:133-156`、`static/js/modules/roi.js:258-313`
- **根因**：camera ID 和点集保存在全局变量中，加载请求没有 AbortController 或会话 token。旧请求返回后可覆盖新 camera 的全局 `roiPoints`。
- **影响**：慢网络下快速切换 camera，可能将 A 的点保存到 B 并实时影响入侵检测。
- **修复建议**：每次打开生成 generation token；取消旧请求；响应前检查 camera；保存时对 camera ID、点集和名称做不可变快照；请求期间禁用重复提交。
- **验收标准**：乱序返回、快速切换和重复点击保存均不会串 camera。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/modules/roi.js`、`static/js/modules/modals.js`、`static/js/utils/api.js`
- 根因修复摘要：每次打开 ROI 编辑器递增 generation 并创建 AbortController；关闭、切换 camera 或开始新请求会中止旧会话。GET 响应只有在 generation、camera ID 和 modal 活动状态同时匹配时才能写 DOM；保存使用 camera、名称和点集的深拷贝快照，并在请求期间禁用保存/删除按钮。
- 测试证据：Playwright 构造 A 请求慢、B 请求快，最终名称为 `B-ZONE`，保存 POST 的 camera 仅为 `Cam_B`、点集为 B 的 3 点；关闭或切换后旧响应不能覆盖当前会话。
- 剩余风险：AbortController 只能阻止客户端继续消费响应，服务端已收到的写请求不能撤销；因此服务端仍必须保留 schema 和 camera 白名单校验。
- 状态：已完成

### P1-04 普通弹窗关闭后 MJPEG 仍持续传输

- **状态**：已完成
- **涉及文件**：`static/js/modules/modals.js:13-16`、`static/js/modules/roi.js:142`、`static/js/modules/modals.js:559-578`
- **根因**：`closeModal()` 只移除 `active`，没有清理 ROI、拓扑放大或快照回退 `<img>` 的 `src`。
- **影响**：隐藏弹窗继续占用连接、带宽、浏览器解码资源和服务端生成器循环；反复打开会累积多条流。
- **修复建议**：建立 modal 生命周期钩子；关闭时只清理该弹窗拥有的 MJPEG 节点；保存、删除、Esc 和遮罩关闭必须走同一清理路径。
- **验收标准**：关闭后 Network 中对应 `/stream/...` 请求终止，重复打开关闭无连接增长。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/modules/modals.js`、`static/js/modules/roi.js`
- 根因修复摘要：普通弹窗的按钮、遮罩和 Esc 关闭统一走 `closeModal()`；关闭时只清空该弹窗内部指向 `/stream/` 的图片源，隐藏视图不再维持 MJPEG 请求。
- Playwright 验证：关闭焦点弹窗后图片 `src` 为空且 overlay 不再 active；重复打开/关闭没有遗留活动流节点。
- 剩余风险：浏览器终止 `<img>` 请求到服务端检测断连存在短暂延迟，但连接不会随反复开关累积。
- 状态：已完成

### P1-05 拓扑节点放大窗口被原弹窗遮挡

- **状态**：已完成
- **涉及文件**：`static/js/modules/modals.js:393-396`、`static/index.html:172-212`、`static/css/components/modals.css:1-15`
- **根因**：点击节点后不关闭拓扑弹窗，两个普通弹窗使用相同 z-index，DOM 中拓扑弹窗位于后方定义位置，因此仍覆盖放大窗口。
- **影响**：用户看不到放大结果，但隐藏窗口已创建视频流，形成额外资源泄漏。
- **修复建议**：打开放大视图前关闭拓扑弹窗，或实现统一 modal stack 和层级管理。
- **验收标准**：点击节点后只存在一个顶层可交互弹窗；关闭后所有相关流终止。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/modules/modals.js`
- 根因修复摘要：点击拓扑节点时先关闭拓扑 modal，再创建 camera 放大视图，避免同层 overlay 按 DOM 顺序互相遮挡。
- Playwright 验证：点击 `cam_04` 后拓扑 modal 已关闭，`.modal-overlay.active` 数量为 1，唯一焦点图片 URL 为 `/stream/cam_04`。
- 剩余风险：当前产品交互仍是单顶层 modal；若未来需要并排或嵌套弹窗，应升级为显式 modal stack，而不是依赖 DOM 顺序。
- 状态：已完成

### P1-06 前端把 API 500 当正常数据

- **涉及文件**：`static/js/modules/modals.js:194-228` 及其他直接调用 `r.json()` 的位置
- **根因**：多数 fetch 只解析 JSON，不检查 `response.ok` 和业务 `error` 字段。
- **影响**：拓扑读取 500 时 `{error: ...}` 会被当作拓扑节点，编辑表为空；用户点击保存可能 POST `{}` 并清空配置。其他页面也会把错误伪装为“暂无数据”。
- **修复建议**：封装统一 `fetchJson()`；非 2xx 抛出结构化错误；加载失败时禁用保存；轮询失败时保留旧状态并明确显示过期。
- **验收标准**：所有主要 API 的 400/500/超时均有明确错误态，不会触发写入或清空数据。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/utils/api.js`、`static/js/app.js`、`static/js/modules/websocket.js`、`static/js/modules/modals.js`、`static/js/modules/roi.js`
- 根因修复摘要：新增全局 `fetchJson()`，统一处理超时取消、非 2xx、非 JSON 响应和业务 `error` 字段，并抛出包含 HTTP status 的结构化错误。状态轮询失败保留旧 camera 和指标；拓扑本体加载失败只显示错误态，不生成编辑区和保存按钮；POST 错误统一展示服务端消息。
- 测试证据：Playwright 注入 `/api/status` 500 后旧 `Cam_A` 保留；`/api/topology` 500 时显示明确错误且无保存按钮；错误 JSON、业务错误和慢请求均不会写入错误数据。
- 剩余风险：轮询失败当前通过控制台 warning 和弹窗错误提示表达，主屏尚未增加全局“数据已过期时间”标记。
- 状态：已完成

### P1-07 前端可能保留旧通道和虚假遥测

- **状态**：已完成
- **涉及文件**：`static/js/modules/grid.js:58-72`、`static/js/modules/grid.js:150-157`、`static/index.html:273-283`
- **根因**：空 camera 数组直接 return；重连 badge 只在首次渲染创建；焦点状态点和 FPS 是固定初始值且从不更新。
- **影响**：后端返回空列表时旧卡片仍显示；焦点窗口可把离线或 CPU 15 FPS camera 显示成绿色、30 FPS。
- **修复建议**：空数组渲染明确空态并清理旧流；统一按 camera ID 更新全部遥测；焦点弹窗订阅同一状态源。
- **验收标准**：状态在线、离线、重连、空列表和 FPS 变化均能在 1 个轮询周期内正确反映。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/frame_hub.py`、`static/js/modules/grid.js`、`static/js/modules/modals.js`、`static/index.html`
- 根因修复摘要：空 camera 数组会销毁旧网格及流并渲染明确空态；焦点视图复用状态轮询数据，动态展示在线状态、真实 FPS 和输出分辨率；重连状态随每轮数据更新。
- Playwright 验证：模拟空 camera 后显示“暂无已注册摄像头通道”；模拟 `Cam_A` 的 15 FPS/640x360 状态后，焦点显示 `15 FPS` 和 `LIVE | 640x360 MJPEG`。
- 剩余风险：状态接口暂时失败时会保留最后一次可信画面，避免误清空；主屏尚未显示最后成功更新时间，相关可观测性记录在 P1-06。
- 状态：已完成

### P1-08 告警截图功能没有生成端

- **涉及文件**：`main.py:120-136`、`src/pipeline.py:50-68`、`src/alerter.py:231`、`static/js/modules/modals.js:67-70`
- **根因**：截图目录只被创建和传递，没有任何帧保存逻辑；`tick()` 的 `screenshot_dir` 参数也没有实际使用。
- **影响**：详情页请求 `{alert_id}.jpg` 永远 404，并回退到当前直播，用户无法查看告警发生瞬间。
- **修复建议**：告警创建时从 FrameHub 或 pipeline 获取帧快照，使用 alert ID 原子保存；告警对象返回明确 `screenshot_url`；失败时展示“无快照”，不能伪装为历史图。
- **验收标准**：每种正式告警都有匹配的静态图片；历史详情打开时不依赖当前 camera 在线。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`main.py`、`src/frame_hub.py`、`src/alerter.py`、`src/pipeline.py`、`static/js/modules/modals.js`、`tests/test_persistence_alerts.py`
- 根因修复摘要：FrameHub 增加锁内复制的最新帧读取；AlertManager 统一构建和发布事件，在写库、JSONL 和广播前以 alert ID 编码 JPEG，先写临时文件再 `os.replace()` 原子替换，成功后写入 `screenshot_url`。pipeline 可直接传当前帧，ticker 类事件从 FrameHub 获取最新帧。
- 测试证据：`test_tick_uses_one_id_for_return_log_database_broadcast_and_snapshot` 和人流密度截图测试通过；快照文件名、SQLite、JSONL、广播和返回值使用同一 ID。Playwright 验证无截图事件显示“该事件无可用现场快照”，500ms 内 `/stream/` 请求数为 0。
- 剩余风险：相机离线、编码失败或无最新帧时事件仍会可靠落库但没有图片；截图磁盘增长由 RISK-02 的保留策略控制。
- 状态：已完成

### P1-09 身份和轨迹没有真正持久化恢复

- **涉及文件**：`main.py:106`、`src/db.py:61-71`、`src/identity_store.py`
- **根因**：SQLite 只保存身份元数据，不保存主特征、Feature Bank 和完整轨迹；启动时总是创建空 `IdentityStore`。
- **影响**：服务重启后人员重新编号，历史告警中的 Global ID 无法在身份接口查到，跨重启 ReID 不成立。
- **修复建议**：明确产品语义。若要求稳定身份，应版本化持久化归一化特征和 Feature Bank 并在启动时恢复；若不要求，应停止宣称状态恢复并将数据库表命名为历史元数据。
- **验收标准**：重启前后同一已登记身份保持一致，旧告警轨迹可查询；模型特征维度变化有迁移或隔离策略。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`main.py`、`src/db.py`、`src/identity_store.py`、`src/reid.py`、`server.py`、`tests/test_persistence_alerts.py`
- 根因修复摘要：身份表版本化保存主特征、Feature Bank、dtype、维度和 `feature_space`；新增 `identity_appearances` 表逐条追加完整轨迹。启动时只恢复与当前模型特征空间兼容且数据可解码的身份；内存 deque 仍只保留最近 200 条，API 可从数据库查询完整历史。
- 测试证据：`test_identity_feature_bank_and_full_trajectory_survive_restart` 验证注册、更新、超过 200 次 appearance、销毁 Store、重建后仍匹配原 ID，数据库轨迹保持完整；`test_incompatible_feature_space_is_not_loaded` 验证同维不同模型不会混库。
- 剩余风险：审计前的 106 条元数据没有可恢复特征，只能作为历史资料保留；生物特征数据库仍需依赖部署权限、备份加密和保留期管理。
- 状态：已完成

### P1-10 `/healthz` 必定返回 500

- **状态**：已完成
- **涉及文件**：`server.py:226-237`
- **根因**：调用 `time.time()` 但模块没有 `import time`。
- **运行复现**：直接调用 `health_check()` 得到 `NameError: name 'time' is not defined`。
- **影响**：健康探针、进程守护和部署检查无法使用。
- **修复建议**：补充导入并增加 API 测试；健康状态应区分服务存活和 camera 是否就绪。
- **验收标准**：未初始化、部分 camera 离线和正常运行三种状态均返回稳定 schema。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `server.py`
  - `tests/test_topology.py`
- 根因修复摘要：在模块级补充 `time` 导入，健康接口不再触发 `NameError`。
- 测试命令：
  - `python -m unittest -v tests.test_topology`
- Playwright/API 验证：未注入 FrameHub 时返回 `cameras_online: 0`；一在线一离线时返回 `cameras_online: 1`；真实 HTTP 请求 `/healthz` 返回 200 和稳定字段。
- 剩余风险：当前端点表达服务存活与在线相机数，尚未将“无在线相机”升级为 readiness 失败；保留现有兼容语义。
- 状态：已完成

### P1-11 人流密度告警是不可达功能

- **涉及文件**：`src/alerter.py:379-406`、`src/pipeline.py:296-356`
- **根因**：`trigger_crowd_warning()` 已实现，但 pipeline 和其他模块从未调用。
- **影响**：文档和前端支持 `CROWD_DENSITY`，实际运行永远不会产生该事件。
- **修复建议**：在检测帧基于有效轨迹数调用；阈值和冷却时间进入配置；明确 WARNING 是否持久化和通知。
- **验收标准**：人数低于、达到、持续超过和恢复四种场景均有测试覆盖且不会刷屏。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/pipeline.py`、`src/alerter.py`、`tests/test_persistence_alerts.py`
- 根因修复摘要：pipeline 在每帧完成有效人员轨迹汇总后调用密度告警入口；AlertManager 按 camera 维护阈值和冷却状态，达到阈值时生成 `CROWD_DENSITY` WARNING，并复用统一发布链生成快照、持久化和广播。
- 测试证据：`test_crowd_density_warning_is_emitted_with_snapshot_and_cooldown` 验证低于阈值不触发、达到阈值触发一次、冷却期持续超限不刷屏，并校验截图与事件 ID 对应。
- 剩余风险：阈值目前是系统级默认值，复杂区域若需要不同承载量，应后续将 camera/ROI 级阈值纳入业务配置。
- 状态：已完成

### P1-12 ReID 匹配率把失败检索计为成功

- **状态**：已完成
- **涉及文件**：`src/reid_validator.py:84-91`、`src/identity_store.py:34-51`
- **根因**：每次检索都调用 `record_match()`；只要不是 Ratio Block 就增加 `successful_matches`，不检查 `matched_id`。
- **运行复现**：正交向量低于阈值，匹配结果为 `None`，指标却得到 `successful_matches=1`、`match_rate=1.0`。
- **影响**：大屏 ReID 成功率严重虚高，无法用于模型质量判断。
- **修复建议**：指标接口显式接收 `matched`；区分成功、低于阈值、Ratio Block、空 gallery 和推理失败。
- **验收标准**：构造上述所有结果时，各计数器和分母定义符合文档。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `src/identity_store.py`
  - `src/reid_validator.py`
  - `tests/test_core_logic.py`
- 根因修复摘要：指标记录接口新增明确 `matched` 参数，只有存在 `matched_id` 才增加 `successful_matches`；Ratio Block 继续独立计数，失败检索只进入总检索分母和延迟统计。
- 测试命令：
  - `python -m unittest -v tests.test_core_logic.IdentityResolutionTests.test_failed_search_is_not_counted_as_success`
- Playwright/API 验证：正交向量检索返回 `None`，`total_searches=1`、`successful_matches=0`、`match_rate=0.0`。
- 剩余风险：当前指标未单列“低于阈值”和“空 gallery”，二者都属于未匹配；不影响成功率正确性。
- 状态：已完成

### P1-13 历史告警查询结果不完整且过滤错误

- **涉及文件**：`server.py:267-310`、`src/db.py:130-156`
- **问题拆分**：
  1. DB 返回空数组时被误判为数据库不可用，并回退 JSONL。
  2. JSONL 路径漏掉 `global_id` 过滤。
  3. `risk_level` 在 SQL LIMIT 之后才过滤，可能返回不足或空结果。
  4. SQLite 有数据时不会合并较旧 JSONL，当前两套数据数量不一致。
  5. `total` 只是当前返回数量，不是过滤条件下的真实总数。
- **运行复现**：查询不存在的 global ID，JSONL 回退仍返回其他身份告警。
- **修复建议**：确定单一权威数据源；风险字段进入 SQL WHERE；实现正确 count；如需迁移 JSONL，提供一次性导入和去重工具，而不是查询时模糊回退。
- **验收标准**：所有过滤组合、空结果、分页、总数和旧数据迁移均有数据库测试。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`main.py`、`src/db.py`、`server.py`、`tests/test_persistence_alerts.py`
- 根因修复摘要：SQLite 成为唯一在线历史权威源；`query_alert_page()` 在 SQL `WHERE` 阶段组合 risk、camera 和 global ID 过滤，再执行 count、limit 和 offset。数据库错误返回 500，不再悄悄回退 JSONL。旧 JSONL 通过稳定 legacy ID 进行一次性幂等导入。
- 数据迁移：迁移前 SQLite 1732 条、JSONL 2775 行；备份后新增导入 1042 条，得到 2774 条唯一事件；682 条无 ID 记录获得稳定 ID；重复执行新增 0 条；原 JSONL 保持不变。
- 测试证据：`test_history_filters_before_pagination_and_reports_true_total`、`test_legacy_jsonl_import_is_complete_and_idempotent` 通过；真实 HTTP 验证 total=2774、offset 页无重叠、`cam_03` 过滤 812 条均正确、不存在 Global ID 返回 total=0。
- 剩余风险：SQLite 是单机权威源，未来多实例部署需要外部数据库或明确的单写者架构；JSONL 仅保留为追加审计副本，不参与在线查询。
- 状态：已完成

### P1-14 IdentityStore 在锁外暴露可变记录

- **涉及文件**：`src/identity_store.py:109-112`、`server.py:373-406`、`src/alerter.py:290-324`
- **根因**：`get()` 直接返回内部 `PersonRecord`，调用方在锁外读取正在被其他 pipeline 修改的 `deque` 和字段。
- **影响**：轨迹响应可能字段不一致，并发 append 时可能出现 deque 迭代异常。
- **修复建议**：提供锁内 `get_snapshot()`，返回复制后的不可变 DTO；外部代码不得持有内部 record 引用。
- **验收标准**：并发更新和轨迹查询压力测试无异常，单个响应内部字段来自同一快照。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/identity_store.py`、`tests/test_persistence_alerts.py`
- 根因修复摘要：`get()` 及查询接口在锁内构造深拷贝快照，特征数组、Feature Bank、appearance 列表和嵌套 bbox 不再引用 Store 内部可变对象；写操作继续在同一锁内完成内存与持久化更新。
- 测试证据：身份持久化重启测试同时覆盖读取快照、Feature Bank 和完整轨迹；调用方修改返回对象不会回写 Store，47 项全量测试通过。
- 剩余风险：深拷贝会增加大 Feature Bank 查询的瞬时内存；当前 bank 和内存 appearance 均有界，尚未构成实际瓶颈。
- 状态：已完成

### P1-15 JPEG 缓存可能用旧帧覆盖新帧

- **状态**：已完成
- **涉及文件**：`src/frame_hub.py:119-135`
- **根因**：帧复制后在锁外编码，写回时只检查 `latest_jpeg is None`，没有确认当前帧仍是被编码的版本。
- **竞态序列**：读取 A -> 编码 A -> pipeline 推入 B 并清空 cache -> A 编码完成并写回 -> B 被错误缓存为 A。
- **影响**：画面可能回退；若随后断流，错误旧帧可能长期保留。
- **修复建议**：为每路帧增加 generation；编码时记录版本，写回时版本必须一致。
- **验收标准**：在编码期间高频推帧，返回缓存永远不倒退到更旧版本。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/frame_hub.py`、`tests/test_frame_hub.py`
- 根因修复摘要：每路帧增加 generation；锁外编码完成后仅在 generation 未变化且 camera 仍在线时写缓存，发生重叠推帧时重试最新版本，避免 A 帧覆盖 B 帧缓存。
- 测试证据：`test_new_generation_wins_when_encoding_overlaps_push` 强制制造编码与推帧重叠，断言最终返回新 generation。
- 剩余风险：已经发送给客户端的旧帧无法撤回；generation 保证的是缓存不会倒退，下一次请求会返回当前最新版本。
- 状态：已完成

### P1-16 MISSING_PERSON 和 SCENE_EXIT 可能重复触发

- **状态**：已完成
- **涉及文件**：`src/alerter.py:249-255`、`src/alerter.py:282-342`
- **根因**：到期 MISSING watch 被删除后，同一轮继续执行 Scene Exit 检查；Scene Exit 只通过当前是否仍在 watch 去重。
- **影响**：同一消失事件在一个 ticker 周期内产生两条告警、两次日志和两次通知。
- **修复建议**：保存消失事件状态或已触发类型；定义两种告警的优先级和互斥规则。
- **验收标准**：同一 ticker 不重复触发；拓扑 MISSING 后仅在继续失联满 5 分钟时升级 SCENE_EXIT；人员重新出现后状态正确重置。

#### 修复记录

- 完成日期：2026-07-29
- 修复提交：工作区改动
- 修改文件：
  - `src/alerter.py`
  - `src/pipeline.py`
  - `tests/test_core_logic.py`
- 根因修复摘要：状态机用 `_missing_person_alerted` 记录 MISSING 触发时间，同一 tick 和后续 300 秒升级窗口内不重复发 SCENE_EXIT；继续失联满 300 秒后允许升级一次。人员重新出现或开始新 watch 时重置状态。候选集合只包含本次运行实际出现或进入 watch 的身份，避免持久化身份在重启后批量误报；observation generation 阻止 ticker 使用重现前的旧快照报警。
- 测试命令：
  - `python -m unittest -v tests.test_core_logic.AlertStateTests`
  - `python -m unittest -v tests.test_persistence_alerts.AlertPersistenceTests.test_scene_exit_uses_one_id_across_all_delivery_channels`
- Playwright/API 验证：45 秒路径先发 MISSING，同轮不发 SCENE_EXIT，继续失联至 MISSING 后 300 秒才升级；末端 camera 在 299.9/300.0 秒边界正确；重新出现可开启下一轮；重启陈旧身份不误报；并发重现 generation 测试通过。Playwright 显示“目标全域场景消失失联”卡片，同 ID 只接收一次，详情字段正确。
- 剩余风险：为避免重启告警风暴，服务停机期间发生且在重启前结束的消失事件不会补报；这是当前单机实时监控语义，若要求离线追溯需持久化运行状态和事件时间线。
- 状态：已完成

---

## 4. P2 缺陷

### P2-01 焦点弹窗强制将 camera ID 转小写

- **状态**：已完成
- **涉及文件**：`static/js/modules/modals.js:671-688`
- **问题**：服务端按原始 ID 精确查找，普通流使用 `Cam_A`，焦点流却请求 `cam_a`。
- **影响**：混合大小写 ID 的普通画面正常，焦点画面显示 OFFLINE。
- **修复建议**：全链路保留原始 ID；URL 段使用 `encodeURIComponent()`；服务端明确 ID 规范。
- **验收标准**：大小写、空格和安全特殊字符 ID 均能打开同一路流。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/modules/modals.js`
- 根因修复摘要：焦点链路保留原始 camera ID，不再调用 `toLowerCase()`；URL 路径段统一通过 `encodeURIComponent()`。
- Playwright 验证：混合大小写 `Cam_A` 的普通流和焦点流均请求 `/stream/Cam_A`，未降格为 `/stream/cam_a`。
- 剩余风险：camera ID 仍区分大小写，这是既有协议语义；配置层应避免仅大小写不同的两个 ID 造成操作混淆。
- 状态：已完成

### P2-02 “清空日志”没有清理焦点缓存

- **涉及文件**：`static/js/modules/websocket.js:211-235`、`static/js/modules/modals.js:581-587`
- **问题**：按钮只清 DOM 和计数，不清 `alertHistoryCache`、`allAlertsHistory`。
- **运行复现**：清空主列表后打开对应 camera 焦点，旧告警重新出现。
- **修复建议**：先确定语义。若只清当前视图，应改名；若清客户端日志，应统一清理所有前端缓存，但不要误删服务端审计历史。
- **验收标准**：按钮文案、主列表、焦点列表和缓存行为一致。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/index.html`、`static/js/modules/websocket.js`、`static/js/modules/modals.js`
- 根因修复摘要：产品语义明确为“清空当前视图”，按钮同步改名；操作会清主列表、主缓存、焦点缓存、焦点 DOM 和当前视图 badge，但保留已见 ID tombstone、服务端审计历史及累计统计。
- 测试证据：Playwright 清空后主列表、焦点列表和 badge 均为 0；同 ID 经 WebSocket 重放不会复活；历史审计 API 数据仍可查询，新 ID 可以正常进入。
- 剩余风险：该操作不是数据删除；若业务需要真正删除审计记录，必须另建带权限、确认和审计的服务端流程。
- 状态：已完成

### P2-03 拓扑弹窗永远匹配不到 AI 校准值

- **状态**：已完成
- **涉及文件**：`static/js/modules/modals.js:272`、`src/calibrator.py:53`
- **问题**：前端拼接 `cam_A->cam_B`，后端统计键是 `cam_A→cam_B`。
- **影响**：即使后端存在有效数据，拓扑图仍只显示静态预期值。
- **修复建议**：API 返回结构化 `from/to` 字段，避免用显示字符串作为协议键。
- **验收标准**：存在至少 5 条合法样本时，拓扑图显示对应校准均值。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `static/js/modules/modals.js`
- 根因修复摘要：前端拓扑边查询键改为与校准器一致的 `from→to` 格式。
- 测试命令：
  - `node --check static/js/modules/modals.js`
- Playwright/API 验证：拦截 `/api/stats` 返回 `cam_04→cam_01` 的 5 条样本、均值 42.5 秒后，拓扑边显示 `AI校准: 42.5s`，其他边继续显示静态预期值。
- 剩余风险：协议仍以组合字符串为键，未来 API 版本可改为结构化边数组；当前前后端格式已一致。
- 状态：已完成

### P2-04 异步轮询无防重入、超时和响应顺序控制

- **涉及文件**：`static/js/app.js:140-142`、`static/js/modules/websocket.js:136-209`
- **问题**：1 秒固定 `setInterval` 可在旧请求未完成时启动新请求，旧响应后到会覆盖新状态。
- **影响**：慢网络下请求堆积、状态回退、服务恢复后仍被旧失败结果干扰。
- **修复建议**：请求完成后再 `setTimeout`；设置 AbortController 超时；用 generation 丢弃过期响应。
- **验收标准**：接口延迟超过轮询周期时，同类请求最多一个在途且状态不倒退。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/app.js`、`static/js/modules/websocket.js`、`static/js/utils/api.js`
- 根因修复摘要：固定 `setInterval` 改为任务完成后再调度下一次的 `setTimeout` runner；每类任务天然只有一个在途请求，`fetchJson()` 为请求提供超时中止，页面卸载时停止后续 timer。
- 测试证据：Playwright 将状态接口延迟 2.5 秒，在 7.2 秒观察窗内共请求 3 次，记录到的最大并发为 1；失败轮次不会清空已有 camera，后续轮次可恢复。
- 剩余风险：浏览器后台标签会主动节流 timer，恢复前台后的首次数据可能较旧，但不会产生并发积压或倒序覆盖。
- 状态：已完成

### P2-05 共享弹窗存在异步响应覆盖

- **涉及文件**：`static/js/modules/modals.js:85-185`
- **问题**：轨迹和身份库复用同一 title/body，没有请求 token；快速切换身份时旧响应可覆盖新标题和内容。
- **修复建议**：打开新内容或关闭弹窗时取消旧请求；响应写 DOM 前验证当前 generation 和实体 ID。
- **验收标准**：乱序响应不会覆盖当前用户选择。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/modules/modals.js`、`static/js/utils/api.js`
- 根因修复摘要：建立按 modal ID 管理的请求 session；打开新实体或关闭弹窗会中止旧 controller 并递增 generation，DOM 写入前校验 modal、generation 和实体 ID；路径参数统一编码。
- 测试证据：Playwright 构造轨迹 A 慢、B 快，最终标题和内容只属于 B；关闭 modal 后 A 的晚响应不能写回；身份库请求不会覆盖已打开的轨迹详情。
- 剩余风险：同一 modal 内未来新增异步分支时必须接入相同 session 管理器，否则可能重新引入覆盖竞态。
- 状态：已完成

### P2-06 焦点实时日志 DOM 无上限增长

- **状态**：已完成
- **涉及文件**：`static/js/modules/modals.js:650-669`
- **问题**：缓存限制 200 条，但焦点窗口打开期间每条告警直接插入 DOM，不删除旧节点。
- **影响**：高频入侵告警会持续增加 DOM、闭包和事件监听器。
- **修复建议**：焦点展示限制 100/200 条或使用虚拟列表，计数区分累计数和展示数。
- **验收标准**：推送数千条告警后 DOM 节点数量保持上限，交互无明显退化。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/modules/modals.js`
- 根因修复摘要：焦点日志每次插入后裁剪到 200 个节点，与前端告警缓存上限一致；累计计数不依赖节点无限保留。
- 测试证据：Playwright 合成 250 条同 camera 告警，内存缓存、计数文本和 `#focus-modal-log-list .alert-card` 均稳定为 200；1280x720 与 1920x1080 下滚动和视频区同时可用。
- 剩余风险：200 是展示上限而非历史删除；更高频率或更长列表需求应使用虚拟滚动，不能简单提高 DOM 上限。
- 状态：已完成

### P2-07 1x1 模式为主 camera 建立重复 MJPEG 流

- **状态**：已完成
- **涉及文件**：`static/js/modules/grid.js:79-103`
- **问题**：主画面创建一条流，轮播缩略图又为所有 camera 创建流，包括当前主 camera。
- **影响**：N 路 camera 产生 N+1 条高帧率连接，多客户端时放大带宽和编码压力。
- **修复建议**：缩略图使用低 FPS 专用流、周期快照或前端复用；当前主 camera 至少不要重复连接。
- **验收标准**：1x1 模式下每个 camera 的连接数符合设计上限。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/modules/grid.js`
- 根因修复摘要：1x1 模式生成缩略图时排除当前主 camera，主画面不再同时出现在缩略图区。
- Playwright 验证：两路 camera 场景仅存在 `/stream/Cam_A` 和 `/stream/cam_b` 各一条，无重复主流。
- 剩余风险：非主 camera 缩略图仍使用标准 MJPEG 流；摄像头数量很大时可进一步增加低 FPS 缩略图端点。
- 状态：已完成

### P2-08 每条告警创建一个 AudioContext 且提示音受自动播放限制

- **涉及文件**：`static/js/utils/formatter.js:24-37`
- **问题**：每次告警新建 AudioContext，从不关闭或复用；页面未经过用户手势时浏览器会阻止启动。
- **运行证据**：Playwright 控制台出现 `AudioContext was not allowed to start`。
- **修复建议**：首次用户交互时初始化单例 AudioContext；复用 context；只创建短生命周期 oscillator/gain 节点。
- **验收标准**：首次授权后连续告警可稳定发声，AudioContext 数量保持 1。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/js/utils/formatter.js`、`static/js/app.js`
- 根因修复摘要：AudioContext 改为模块级单例；首次 `pointerdown` 或 `keydown` 时创建并 resume，之后每次告警只创建短生命周期 oscillator/gain，结束时断开节点。用户尚未授权时静默跳过，不反复创建失败 context。
- 测试证据：全部 JS 语法检查通过；Playwright 多次告警回归未出现自动播放警告，去重事件也不会重复触发声音路径。
- 剩余风险：浏览器或操作系统静音策略仍可能阻止声音，值班端不能只依赖音频，应同时保留视觉告警。
- 状态：已完成

### P2-09 ROI 点数和画面坐标存在边界问题

- **涉及文件**：`server.py:107-124`、`static/js/modules/roi.js:269-288`、`static/css/components/roi-canvas.css`
- **问题**：
  - 服务端允许 1 或 2 个顶点的非多边形配置。
  - 自由画线超过 64 点后直接截取前 64 点，闭合边会横穿原图形。
  - 非 16:9 视频预览使用裁切展示时，归一化坐标与后台完整帧不一致。
- **修复建议**：服务端要求 0 或至少 3 点；调整简化 epsilon 直到不超过限制，不能直接截断；ROI 预览与后端使用一致的 contain/letterbox 映射。
- **验收标准**：4:3、16:9、竖屏视频和复杂自由绘制均能准确命中同一区域。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`server.py`、`static/js/modules/roi.js`、`static/css/components/roi-canvas.css`、`tests/test_roi_api.py`
- 根因修复摘要：服务端只允许空 polygon 或 3-64 个二维归一化点，并校验有限数值和 `[0,1]` 范围；客户端用逐步提高容差的 Douglas-Peucker 简化替代直接截断。预览使用 `object-fit: contain` 和稳定宽高比，指针与绘制都通过实际 image content rect 在 letterbox 与归一化坐标间转换。
- 测试证据：`test_polygon_count_and_coordinate_validation` 覆盖 0/1/2/3/64/65 点及非法坐标；Playwright 验证 A/B ROI 乱序、3 点保存和容器坐标映射，窄屏下预览比例稳定。
- 剩余风险：当前未拒绝自交多边形；如果业务要求严格几何区域，后续可增加自交检测与可视化提示。
- 状态：已完成

### P2-10 CSV 字段转义不完整

- **涉及文件**：`server.py:313-356`
- **问题**：已处理 Excel 公式注入，但字段内双引号没有按 CSV 规范转换为 `""`，换行也未统一处理。
- **影响**：特殊 camera ID 或配置文本会破坏列结构。
- **修复建议**：使用标准库 `csv.writer` 和 `io.StringIO`，继续保留公式注入防护。
- **验收标准**：包含逗号、双引号、换行和公式前缀的字段在 Excel 中正确分列且不执行公式。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`server.py`、`src/db.py`、`tests/test_persistence_alerts.py`
- 根因修复摘要：CSV 改从 SQLite 权威源导出，使用 `io.StringIO(newline="")` 与标准库 `csv.writer` 处理逗号、双引号和 CR/LF；保留 UTF-8 BOM，并对 `= + - @ tab CR` 前缀添加单引号防止 Excel 公式注入。
- 测试证据：`test_csv_export_round_trips_special_fields` 使用 `csv.reader` 回读包含中文、逗号、双引号、换行和公式前缀的数据；真实 HTTP 导出解析为 2774 行，与 SQLite/API total 一致。
- 剩余风险：导出当前是全量内存构建；30 天保留期下可控，若记录量扩大应改为分页流式生成。
- 状态：已完成

### P2-11 FrameHub 只按宽度决定是否缩放

- **状态**：已完成
- **涉及文件**：`src/frame_hub.py:121-126`
- **问题**：输入恰好 640x480 时宽度满足条件，因此不会缩成声明的 640x360。
- **影响**：输出流分辨率和宽高比不稳定，前端 HUD 的固定规格更加不可信。
- **修复建议**：同时比较宽高；明确采用拉伸、contain 或 letterbox 策略。
- **验收标准**：任意输入尺寸输出均符合统一尺寸和宽高比策略。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/frame_hub.py`、`tests/test_frame_hub.py`
- 根因修复摘要：JPEG 编码前同时比较宽度和高度，任一维不符合配置即按既有策略缩放到统一输出尺寸。
- 测试证据：`test_resize_checks_both_width_and_height` 覆盖输入宽度相同但高度不同的场景。
- 剩余风险：既有固定尺寸策略会拉伸非 16:9 输入；如需保持原始比例，应统一改成 letterbox 并同步 ROI 坐标策略。
- 状态：已完成

### P2-12 `tick()` 返回的告警 ID 与实际广播记录不同

- **涉及文件**：`src/alerter.py:267-286`、`src/alerter.py:346-361`
- **问题**：真实告警已构建、写入和广播后，返回值再次调用 `_build_alert()`，生成新的 UUID 和时间戳。
- **影响**：未来调用方若使用返回值，会拿到数据库中不存在的 alert ID。
- **修复建议**：保存已构建 alert 列表并原样返回。
- **验收标准**：返回、SQLite、JSONL 和 WebSocket 中同一事件的 alert ID 完全一致。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/alerter.py`、`tests/test_persistence_alerts.py`
- 根因修复摘要：`tick()` 只构建一次事件对象，并将实际送入统一发布链的对象加入返回列表；不再在函数尾部二次调用 `_build_alert()`。截图、SQLite、JSONL、广播、通知和返回值共同消费同一事件 ID 与时间戳。
- 测试证据：`test_tick_uses_one_id_for_return_log_database_broadcast_and_snapshot` 同时断言 MISSING 返回值、广播缓存、JSONL、SQLite 和截图文件名五处 ID 一致；`test_scene_exit_uses_one_id_across_all_delivery_channels` 对 SCENE_EXIT 验证同一契约。WARNING 与 ALERT 各自只构建一次。
- 剩余风险：事件 dict 在发布后约定只读；未来若消费者修改对象，应在 broadcaster 边界复制或引入不可变 DTO。
- 状态：已完成

### P2-13 Broadcaster 启动前队列不会迁移

- **涉及文件**：`main.py:123-153`、`src/alerter.py:64-72`、`server.py:493-501`
- **问题**：pipeline 先于 Web 服务启动，早期告警进入同步 queue；事件循环建立 async queue 后只消费新队列，旧消息永久滞留。`--no-web` 模式下 queue 还会无限增长。
- **修复建议**：在切换队列时迁移旧消息；无 WebSocket 消费者时不要入广播队列；设置容量和丢弃策略。
- **验收标准**：启动窗口告警只广播一次且同步 queue 最终为空；no-web 长时运行队列不增长。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`main.py`、`server.py`、`src/alerter.py`、`tests/test_operations.py`
- 根因修复摘要：事件循环接入时在锁内把启动前同步队列迁移到有界 async queue，并清空旧队列；迁移只执行一次。Broadcaster 增加启用开关和容量/丢弃策略，`--no-web` 模式不再累积无人消费的广播消息。
- 测试证据：`test_pre_start_messages_move_to_async_queue_once` 验证早期消息迁移一次且顺序保持；`test_no_web_mode_does_not_grow_delivery_queue` 验证长期 no-web 发布队列仍为空。
- 剩余风险：队列满时按策略丢弃实时推送，但权威事件仍已落 SQLite/JSONL，客户端可通过历史 API 补齐。
- 状态：已完成

### P2-14 启动状态可能误报成功且日志重定向不完整

- **涉及文件**：`server.py:540-549`、`main.py:146-167`、`start.ps1`
- **问题**：Web 线程创建后立即记录“已启动”，固定等待 0.5 秒但不检查端口；PowerShell 只重定向 stdout，而 Python logging 默认输出到 stderr。
- **影响**：端口占用、Uvicorn 启动失败或模型仍在加载时，脚本可能宣称成功；`outputs/server.log` 缺少关键日志。
- **修复建议**：同时重定向 stdout/stderr；启动脚本轮询 `/healthz` 或端口直到超时；服务线程异常传回主线程。
- **验收标准**：端口占用时启动明确失败；成功提示前健康接口已可访问；日志包含 Python logging 输出。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`server.py`、`start.ps1`、`tests/test_operations.py`
- 根因修复摘要：`start_server_thread()` 在建线程前执行独占端口探测，启动后轮询真实 `/healthz`，线程提前退出或超时会抛错；启动脚本同时重定向 stdout/stderr，最多等待 120 秒健康响应，失败时清理本次 PID 文件并报告日志位置。
- 测试证据：`test_port_probe_rejects_an_existing_listener` 通过；真实调用 `start_server_thread(host='127.0.0.1', port=8768)` 返回存活的 `web-server` 线程；`start.ps1` PowerShell Parser 检查通过。
- 剩余风险：未执行会加载检测/ReID 权重的完整 `start.ps1` 冷启动，以避免任务期间下载模型；首次模型下载时间仍受网络影响，脚本的 120 秒超时可按部署环境调整。
- 状态：已完成

### P2-15 停止脚本可能丢数据或终止无关进程

- **涉及文件**：`stop.ps1`、`src/calibrator.py:43`、`src/alerter.py:107-128`
- **问题**：脚本使用 `Stop-Process -Force`，正常退出钩子不保证执行；PID 文件失效时还会终止所有占用 8000 端口的进程。
- **影响**：校准脏数据可能未落盘；误杀其他本地服务。
- **修复建议**：优先发送可处理的终止信号或提供管理关闭接口；核验 PID 的命令行和工作目录；超时后才强制终止。
- **验收标准**：正常停止会 flush 校准数据并关闭日志；PID 不属于本项目时拒绝终止。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`main.py`、`server.py`、`stop.ps1`、`src/calibrator.py`、`src/alerter.py`、`src/db.py`、`tests/test_operations.py`
- 根因修复摘要：新增仅允许 loopback 调用的 `/api/admin/shutdown`；main 收到请求后设置 stop event，停止 pipeline/ticker，flush 校准器，关闭告警日志和数据库。停止脚本只信任 PID 文件，并同时核验进程是运行 `main.py` 的 Python 且拥有 8000 监听端口；先请求安全关闭，20 秒后才强制终止已验证进程，不再扫描并杀死任意端口占用者。
- 测试证据：`test_shutdown_endpoint_is_loopback_only` 通过；临时真实服务健康后 POST shutdown 返回 `{"status":"stopping"}`，进程在 10 秒内以 exit code 0 退出；`stop.ps1` Parser 检查通过。
- 剩余风险：底层 RTSP 驱动若忽略超时仍可能拖到 20 秒兜底强停；该条件由 RISK-05 继续约束。
- 状态：已完成

### P2-16 响应式布局和无障碍支持不足

- **涉及文件**：`static/css/components/alert-list.css`、`static/css/components/cards.css`、`static/css/components/roi-canvas.css`、`static/index.html:172-305`、`static/js/modules/modals.js:8-16`
- **问题**：
  - 缺少媒体查询，右栏和统计区使用固定宽度/列数。
  - 800x600 下主监控区和 ReID 内容被裁切。
  - ROI 高度固定，窄屏下坐标映射失真。
  - 普通 modal 缺少 `role="dialog"`、`aria-modal`、焦点进入/约束/恢复和统一 Esc 处理。
  - 部分可点击 div 无键盘语义。
- **修复建议**：增加桌面窄屏断点；ROI 使用稳定 16:9 aspect-ratio；建立统一 modal focus manager；交互元素改用 button/link。
- **验收标准**：800x600、1280x720、1920x1080 下无内容裁切；键盘可完成核心弹窗和告警操作；读屏能识别弹窗标题与状态。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`static/index.html`、`static/css/layout.css`、`static/css/components/alert-list.css`、`static/css/components/cards.css`、`static/css/components/modals.css`、`static/css/components/roi-canvas.css`、`static/js/modules/grid.js`、`static/js/modules/modals.js`、`static/js/modules/websocket.js`
- 根因修复摘要：增加窄屏断点，使主布局纵向滚动、统计卡 2 列、ReID 指标换行、日志栏按可用宽度排列；ROI 使用稳定 aspect-ratio。所有 modal 增加 dialog ARIA，打开后聚焦首个控件，Tab/Shift+Tab 约束在顶层 modal，Esc 只关闭顶层并恢复触发元素；动态告警和 camera 卡补齐键盘语义。
- 测试证据：Playwright 在 800x600 验证无横向溢出、统计 2 列、视频区域可见；1280x720 焦点日志 200 条时视频仍有 856x585 可见区；1920x1080 有 1398x909 可见区。dialog ARIA、打开焦点、Esc 关闭和焦点恢复均通过。
- 剩余风险：页面适配的是桌面值班屏和最低 800x600；更窄的手机屏不是当前产品目标，复杂拓扑编辑在触屏设备上仍需专门交互设计。
- 状态：已完成

### P2-17 依赖清单不完整

- **涉及文件**：`requirements.txt`
- **问题**：未显式声明 `opencv-python`、`numpy`、`psutil`。当前环境能运行主要依赖传递安装和本机已有包，不代表新环境可靠。
- **影响**：上游依赖调整后安装可能缺包；`psutil` 缺失时内存指标静默显示 0。
- **修复建议**：列出所有直接依赖并约束兼容版本；为 CPU/GPU 安装方式保留明确说明；CI 从空环境安装验证。
- **验收标准**：新虚拟环境只执行文档安装步骤即可通过 import、服务启动和最小推流检查。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`requirements.txt`、`docs/deployment.md`
- 根因修复摘要：显式加入直接依赖 `numpy>=1.24,<3.0`、`opencv-python>=4.8.0` 和 `psutil>=5.9.0`；继续把 PyTorch/torchvision 按 CPU、CUDA 11.8、CUDA 12.1 分步安装，避免 requirements 固化错误硬件 wheel。
- 测试证据：`pip check` 返回 `No broken requirements found`；Python compileall、47 项测试和 FrameHub/OpenCV 路径全部通过；requirements 中的安装顺序与 `docs/deployment.md`、README 现有说明一致。
- 剩余风险：本轮没有创建全新虚拟环境重装大型模型依赖；CI 仍应增加空环境安装矩阵覆盖 CPU 和目标 CUDA 版本。
- 状态：已完成

### P2-18 历史回填后统计数字与可见列表不一致

- **涉及文件**：`static/js/app.js:125-130`、`static/js/modules/websocket.js:101-127`
- **问题**：历史记录使用 `silent=true`，可见列表可能已有 30 条正式告警，但“累计触发告警数”和 badge 仍为 0。
- **影响**：页面初始状态自相矛盾；接收一条新告警后计数变为 1，但列表实际有多条。
- **修复建议**：明确统计是“本次会话新增”还是“历史累计”；若是会话新增，修改文案；若是历史累计，从服务端统计接口初始化。
- **验收标准**：统计定义、文案和列表数据一致，刷新后不产生误导。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/db.py`、`server.py`、`static/js/app.js`、`static/js/modules/websocket.js`
- 根因修复摘要：历史 API 返回按 stage 聚合的 `summary.alerts/warnings`；页面 hydrate 后用 summary 初始化历史累计统计，当前视图 badge 单独按可见正式告警维护。历史回填不响铃、不重复增加累计数；新事件才同时更新累计和当前视图；清空视图不重置历史累计。
- 测试证据：真实页面初始累计告警显示 2774，与历史 API summary 一致；Playwright 验证重复 ID 不改变统计，清空后 badge 归零而累计值保持，新 ID 只增加 1。
- 剩余风险：当前 summary 每次历史请求执行聚合查询；30 天保留期内成本可控，数据量继续增长时可增加汇总表或索引评估。
- 状态：已完成

### P2-19 前端视频规格和焦点状态标签为固定假值

- **状态**：已完成
- **涉及文件**：`static/js/modules/grid.js:52`、`static/index.html:273-283`、`static/index.html:300-301`
- **问题**：主画面固定显示 `1080P MJPEG`，实际 FrameHub 输出 640x360；焦点固定显示 `30 FPS`，CPU 模式实际为 15 FPS，离线时也不更新。
- **影响**：值班界面展示虚假遥测。
- **修复建议**：由后端状态接口返回实际输出分辨率、FPS 和在线状态，前端统一绑定动态字段。
- **验收标准**：CPU、GPU、离线和自定义输出尺寸下标签均与真实值一致。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/frame_hub.py`、`static/js/modules/grid.js`、`static/js/modules/modals.js`、`static/index.html`
- 根因修复摘要：FrameHub 状态返回实际输出宽高、目标 FPS 和 JPEG 质量；网格 HUD 与焦点标签绑定状态数据，不再写死 `1080P` 或 `30 FPS`。
- Playwright 验证：模拟 640x360、15 FPS camera 时焦点精确显示 `15 FPS` 与 `640x360 MJPEG`，离线状态不再显示为 LIVE。
- 剩余风险：FPS 表示 pipeline 当前配置/状态值，不是独立的端到端浏览器解码帧率测量；网络抖动时用户实际观感可能更低。
- 状态：已完成

---

## 5. 条件性运行风险

以下问题依赖部署环境或负载，不应与已复现 Bug 混淆，但在上线前必须处理或明确接受。

### RISK-01 服务无认证且监听所有网卡

- **状态**：已完成
- 修复前 `server.py` 默认监听 `0.0.0.0`。
- 修复前视频、身份、轨迹、告警、ROI 和拓扑写接口均无认证授权。
- 在非完全隔离网络中，任何可达主机都能查看监控并修改配置。

#### 修复记录

- 完成日期：2026-07-28
- 修复提交：工作区改动
- 修改文件：
  - `main.py`
  - `server.py`
  - `tests/test_topology.py`
  - `docs/deployment.md`
- 根因修复摘要：默认监听地址从 `0.0.0.0` 收紧为 `127.0.0.1`；新增覆盖 HTTP、视频流和 WebSocket 握手的 HTTP Basic 校验。绑定任意非本机地址时，若未同时设置 `LAB_MONITOR_USERNAME` 与 `LAB_MONITOR_PASSWORD`，服务在创建线程前同步拒绝启动。
- 测试命令：
  - `python -m unittest -v tests.test_topology`
  - `curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8766/api/topology`
  - `curl.exe -s -o NUL -w "%{http_code}" -u audit-user:audit-pass http://127.0.0.1:8766/api/topology`
- Playwright/API 验证：启用凭据的临时服务中，未认证请求返回 401，正确凭据返回 200；默认本机服务无需凭据，保持当前单机使用方式。
- 剩余风险：HTTP Basic 只负责认证，不提供传输加密；非本机部署必须在可信反向代理后启用 HTTPS。该要求后续同步到部署文档。
- 状态：已完成

### RISK-02 身份和日志没有保留策略

- `IdentityStore` 不淘汰身份。
- JSONL、SQLite 告警不轮转、不归档。
- Scene Exit 每 0.5 秒遍历全部身份，长期运行后内存、磁盘和 ticker 耗时会持续增长。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`main.py`、`src/db.py`、`src/identity_store.py`、`docs/deployment.md`、`tests/test_operations.py`
- 根因修复摘要：默认保留最近 30 天的 SQLite 告警、身份轨迹、JSONL 镜像和截图，启动时在 JSONL 迁移完成后统一清理过期数据；`LAB_MONITOR_RETENTION_DAYS` 可调整期限。IdentityStore 默认最多 10000 个运行时身份，可由 `LAB_MONITOR_MAX_IDENTITIES` 调整，appearance 内存窗口仍为 200。
- 测试证据：`test_database_retention_removes_only_expired_events` 验证只删除截止时间前的数据；文件清理使用同一 cutoff，部署文档已记录默认行为、环境变量和生物特征数据注意事项。
- 剩余风险：保留策略在启动时执行，不是持续后台归档；服务极长期不重启时磁盘仍会增长，生产环境应配合磁盘监控和计划性维护窗口。
- 状态：已完成

### RISK-03 同步 IO 阻塞 FastAPI 事件循环

- 历史 JSONL 全文件扫描、SQLite 查询、CSV 导出和配置写盘都直接发生在 async 路由中。
- 日志变大或磁盘变慢后会同时影响 API 和 WebSocket 广播。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`server.py`、`src/db.py`
- 根因修复摘要：历史查询、summary、身份轨迹数据库读取和 JSON 配置读写通过 `run_in_threadpool` 移出事件循环；CSV 导出改为同步 FastAPI 路由，由线程池执行；在线历史不再扫描 JSONL 全文件。
- 测试证据：真实 HTTP 的 2774 条历史、过滤、CSV 导出和 WebSocket 页面并行工作正常；47 项测试与浏览器轮询回归均通过。
- 剩余风险：线程池只能隔离阻塞，不能消除慢磁盘吞吐上限；超大 CSV 仍会占用一个工作线程和内存，后续可改流式导出。
- 状态：已完成

### RISK-04 共享 JSONL 句柄缺少独立写锁

- 多个 pipeline 和 ticker 可同时调用 `_write_log()`。
- 高并发下存在日志行交叉、关闭竞态或部分写入风险。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/alerter.py`、`tests/test_operations.py`
- 根因修复摘要：告警 JSONL 写入和关闭使用独立可重入锁；每条记录在锁内完成序列化、单行写入和 flush，关闭过程与并发发布互斥。
- 测试证据：`test_concurrent_jsonl_writes_remain_parseable` 以 40 个并发写入验证行数完整、每一行均可独立 JSON 解析且 ID 不串行。
- 剩余风险：单进程锁不覆盖多个进程同时写同一 JSONL；当前部署模型是单主进程，未来多实例必须改用集中式日志或进程级文件锁。
- 状态：已完成

### RISK-05 RTSP 连接和读取没有超时

- OpenCV `VideoCapture` 打开和 `read()` 可能长期阻塞。
- `_stop_event` 无法中断正在阻塞的底层调用，停止服务可能长时间无响应。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/pipeline.py`、`requirements.txt`、`docs/deployment.md`、`tests/test_frame_hub.py`
- 根因修复摘要：RTSP 创建 Capture 时在 OpenCV 支持的情况下设置 `CAP_PROP_OPEN_TIMEOUT_MSEC` 和 `CAP_PROP_READ_TIMEOUT_MSEC`，默认均为 10000ms；连接/读取异常进入可停止的重连循环，Capture 始终在 `finally` 释放。依赖最低版本提升到 `opencv-python>=4.8.0`。
- 测试证据：`test_rtsp_read_exception_reconnects_and_releases_each_capture` 验证读取异常会释放每个 Capture 并重连；安全停止集成测试在正常服务路径下 exit code 0。
- 剩余风险：具体 RTSP timeout 是否被执行仍取决于 OpenCV 构建和 FFmpeg/GStreamer 后端；部署时必须用目标摄像头做断网、黑洞地址和认证失败测试。
- 状态：已完成

### RISK-06 RTSP 凭据会被明文记录

- `main.py` 和 `pipeline.py` 会记录完整 source。
- 使用 `rtsp://user:password@host/...` 时，日志会泄漏用户名和密码。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`main.py`、`src/pipeline.py`、`tests/test_operations.py`
- 根因修复摘要：新增统一 `redact_source()`，任何启动、连接、重试和错误日志在输出 source 前先移除 URL userinfo；无法可靠解析的 RTSP 地址整体替换为 `rtsp://<redacted>`，原始 URL 只保留在内存连接参数中。
- 测试证据：`test_rtsp_credentials_are_redacted` 验证用户名、密码和编码后的凭据均不会出现在日志字符串，同时保留无敏感信息的 host/path 诊断价值。
- 剩余风险：第三方 OpenCV/FFmpeg 自身的原生日志不受 Python formatter 完全控制；生产日志仍应限制文件 ACL 和集中采集权限。
- 状态：已完成

### RISK-07 WebSocket 广播缺少慢客户端隔离

- 服务端逐客户端串行 `send_text()`，没有超时。
- 单个慢客户端可拖慢所有客户端并导致无界 async queue 增长。

#### 修复记录

- 完成日期：2026-07-28
- 修改文件：`src/alerter.py`、`server.py`、`tests/test_operations.py`
- 根因修复摘要：广播循环对客户端使用 `asyncio.gather(..., return_exceptions=True)` 并发发送，每个 `send_text()` 用 2 秒 `asyncio.wait_for()` 限时；超时或断开的客户端单独移除，不阻塞其他连接。Broadcaster async queue 同时设为有界。
- 测试证据：`test_slow_client_times_out` 验证慢客户端被超时隔离，快客户端正常收到消息；no-web 和启动迁移测试验证队列不会无界增长或重复消费。
- 剩余风险：高并发客户端仍受单进程事件循环和网络带宽限制；大规模部署应增加连接上限、反向代理和水平扩展方案。
- 状态：已完成

---

## 6. 配置异常待确认

当前 `config/topology.json` 中存在以下时间：

| 路径 | 当前期望时间 | 换算 |
| --- | ---: | ---: |
| `cam_01 → cam_03` | 10000 秒 | 约 2.8 小时 |
| `cam_01 → cam_02` | 30000 秒 | 约 8.3 小时 |
| `cam_04 → cam_01` | 40 秒 | 40 秒 |

如果 10000/30000 原意是 100/300 秒，这会导致 MISSING_PERSON 数小时不触发。该项属于业务配置确认，修复代码时不能擅自修改。

---

## 7. 推荐修复批次

### 批次 A：安全和配置原子性

- P0-01 拓扑持久化 XSS
- P0-05 拓扑先写后校验
- RISK-01 接口认证授权
- 目标：确保外部输入不能执行脚本、破坏文件或修改未授权配置。

### 批次 B：跨摄身份和告警正确性

- P0-02 跨摄校准共享状态
- P0-03 Ratio Test 一致性
- P0-04 非预期摄像头 resolve
- P1-12 ReID 指标
- P1-16 重复消失告警
- 目标：先保证人是谁、从哪来、是否按预期到达这三个核心结论可信。

### 批次 C：摄像头生命周期和视频真实性

- P0-06 离线状态与旧帧
- P1-01 pipeline 异常边界
- P1-04 隐藏 MJPEG 流
- P1-05 拓扑放大层级
- P1-07 前端状态残留
- P1-15 JPEG 缓存竞态
- P2-07 重复 MJPEG
- P2-11 输出尺寸
- P2-19 虚假规格标签
- 目标：任何时候页面展示都能明确区分实时、离线、重连和历史画面。

### 批次 D：告警历史、截图和持久化

- P1-02 WebSocket 去重
- P1-08 告警截图
- P1-09 身份恢复
- P1-13 历史查询
- P2-02 清空日志语义
- P2-10 CSV
- P2-12 alert ID 一致性
- P2-18 历史统计初始化
- 目标：每个告警只有一个 ID、一条权威记录、一张对应快照，并能跨重启查询。

### 批次 E：前端竞态和资源治理

- P1-03 ROI 串数据
- P1-06 API 错误处理
- P2-04 轮询防重入
- P2-05 弹窗响应竞态
- P2-06 焦点 DOM 上限
- P2-08 AudioContext
- P2-09 ROI 映射
- P2-16 响应式与无障碍
- 目标：慢网络、高频告警和窄屏环境下仍保持数据一致和可操作。

### 批次 F：部署与运维

- P1-10 健康检查
- P2-13 Broadcaster 队列切换
- P2-14 启动状态和日志
- P2-15 安全停止
- P2-17 依赖清单
- RISK-02 至 RISK-07
- 目标：服务可可靠启动、检查、停止、恢复和长期运行。

---

## 8. 每项修复的通用完成标准

问题只有同时满足以下条件才可从“待修复”改为“已完成”：

1. 根因已修复，不是只隐藏前端现象或增加临时条件分支。
2. 添加与风险匹配的最小自动化测试；没有测试框架时先补对应模块的小型测试。
3. 后端改动至少通过 Python 编译、相关单元测试和 API 行为测试。
4. 前端改动至少通过全部 JS 语法检查；涉及布局、状态或交互时使用 Playwright 验证。
5. 涉及并发时必须包含竞态复现测试或压力验证，不能只靠单线程用例。
6. 涉及配置写入时必须验证非法输入不会改变磁盘和运行时状态。
7. 涉及持久化时必须验证重启前后数据一致。
8. 涉及告警时必须验证 REST、SQLite、JSONL、WebSocket、页面计数和快照使用同一 alert ID。
9. `git diff --check` 通过，且没有覆盖用户已有未提交改动。
10. 在本文档对应条目下补充修复提交、测试命令、验证结果和完成日期。

---

## 9. 修复记录模板

后续完成单项修复时，在对应问题下追加以下内容：

```markdown
#### 修复记录

- 完成日期：YYYY-MM-DD
- 修复提交：`<commit hash>`（未提交时写“工作区改动”）
- 修改文件：
  - `path/to/file.py`
- 根因修复摘要：
- 测试命令：
  - `python -m ...`
- Playwright/API 验证：
- 剩余风险：无 / 具体说明
- 状态：已完成
```

---

## 10. 最终验证与迁移结论

### 10.1 自动化与静态检查

- `python -m unittest -v`：47 项全部通过，覆盖核心状态机、Scene Exit 阶段升级/重启/竞态/全链路、拓扑原子更新、FrameHub/pipeline 生命周期、身份/告警持久化、JSONL 迁移、CSV、ROI schema、广播生命周期、保留策略、慢客户端隔离和运维安全。
- `python -m compileall -q main.py server.py src tests`：通过。
- `rg --files static/js -g '*.js'` 找到的 8 个 JavaScript 文件全部通过 `node --check`。
- `start.ps1`、`stop.ps1` 全部通过 PowerShell AST Parser 检查。
- `pip check`：`No broken requirements found`。
- `git diff --check`：通过；只有 Git 关于工作区 LF 将来可能转换为 CRLF 的提示，没有空白错误。

### 10.2 真实 API 与运维验证

- `/healthz` 返回 200，schema 为 `status/service/cameras_online/timestamp`。
- 历史 API：`total=2774`；limit=2 的 offset 0/2 两页无重复；`cam_03` 过滤 total=812，抽样行 camera 全部正确；不存在的 Global ID 返回 total=0、空数组。
- CSV：HTTP 导出经 PowerShell `ConvertFrom-Csv` 解析为 2774 行，与 SQLite 和历史 API 一致。
- `start_server_thread(host='127.0.0.1', port=8768)` 完成端口探测和真实健康检查，返回存活的 `web-server` 线程。
- 临时 Uvicorn 服务调用 `/api/admin/shutdown` 返回 `{"status":"stopping"}`，进程在 10 秒内以 exit code 0 退出。
- 未执行完整 `start.ps1` 模型冷启动：该路径可能下载 YOLO/OSNet 权重并占用 GPU；端口、健康检查、日志重定向和脚本语法已分别验证。正式部署仍需在目标硬件执行一次冷启动验收。

### 10.3 Playwright UI 回归

- 800x600：无横向溢出；统计区为 2 列；ReID 指标可换行；视频区域可见；右侧日志转为纵向布局；无残留 active modal。
- 1280x720：焦点左侧日志存在 200 条时，视频实际可见区域 856x585；日志 DOM 与计数均封顶 200；无横向溢出。
- 1920x1080：焦点视频实际可见区域 1398x909；日志仍为 200 条且可滚动；无横向溢出。
- 焦点视频使用可解码 640x360 图像验证，`naturalWidth/naturalHeight=640/360`；解决了“焦点放大后左侧带日志时视频不显示”的原始问题。
- 无 `screenshot_url` 的告警详情显示“该事件无可用现场快照”，监控窗口内 `/stream/` 请求为 0，历史详情不再回退直播。
- 竞态回归：ROI A 慢/B 快只保存 B；轨迹 A 慢/B 快只显示 B；状态接口 2.5 秒延迟时最大并发为 1；同一 alert ID 只接收一次；清空当前视图后旧 ID 不复活。
- Scene Exit 回归：页面显示“目标全域场景消失失联”，同一 ID 第二次被拒绝且累计只增加 1；详情正确展示 camera、Global ID、300 秒、全域监控区和无快照状态。
- 可访问性：dialog ARIA、打开后焦点进入、Tab 约束、Esc 关闭顶层、关闭后焦点恢复均通过。
- 增加复用 `/static/logo.png` 的 favicon 后，页面控制台为 0 errors、0 warnings。
- 本轮 Playwright 截图、console 日志和 YAML 快照已清理，没有把临时验证产物留在项目中。

### 10.4 数据迁移结论

- 迁移前：SQLite 1732 条，JSONL 2775 行。
- 备份：`outputs/lab_monitor.pre-alert-migration.2026-07-28.db`。
- 迁移后：SQLite 2774 条，2774 个唯一 `alert_id`。
- 新增导入 1042 条；682 条无 ID 旧事件获得稳定 `legacy_<sha256>` ID；1 条完全重复事件合并。
- 第二次执行新增 0 条，证明幂等；原 `outputs/alerts.jsonl` 未修改。
- 在线查询和 CSV 已统一读取 SQLite；JSONL 作为追加审计镜像保留，不再参与模糊回退。

### 10.5 最终边界

- 代码层 41 项缺陷和 7 项风险治理均已完成；每个编号均包含修改文件、根因修复、测试证据和剩余风险。
- `config/topology.json` 中 10000/30000 秒仍是业务配置待确认项，未擅自修改。
- 本轮未创建分支、未提交、未推送，也未回滚用户已有工作区改动。

## 11. 完整状态矩阵

| 分类 | 编号 | 数量 | 最终状态 |
| --- | --- | ---: | --- |
| P0 | P0-01、P0-02、P0-03、P0-04、P0-05、P0-06 | 6 | 全部已完成 |
| P1 | P1-01、P1-02、P1-03、P1-04、P1-05、P1-06、P1-07、P1-08、P1-09、P1-10、P1-11、P1-12、P1-13、P1-14、P1-15、P1-16 | 16 | 全部已完成 |
| P2 | P2-01、P2-02、P2-03、P2-04、P2-05、P2-06、P2-07、P2-08、P2-09、P2-10、P2-11、P2-12、P2-13、P2-14、P2-15、P2-16、P2-17、P2-18、P2-19 | 19 | 全部已完成 |
| RISK | RISK-01、RISK-02、RISK-03、RISK-04、RISK-05、RISK-06、RISK-07 | 7 | 全部已完成治理并记录剩余风险 |
| 合计 | 41 组确定性缺陷 + 7 组条件性风险 | 48 | 全部已完成 |
