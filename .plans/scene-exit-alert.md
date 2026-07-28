# 场景消失告警功能实现方案

> 状态：已完成实现与回归验证
> 完成日期：2026-07-29
> 最终语义：末端摄像头人员全域消失 300 秒触发；拓扑路径先触发 `MISSING_PERSON`，若之后继续失联 300 秒再升级为 `SCENE_EXIT`；重新出现后允许进入下一轮。

## 需求
人员从**所有摄像头**消失超过 **5 分钟**后触发 `SCENE_EXIT` 告警。

---

## 设计思路

### 1. 数据源
`IdentityStore.PersonRecord.last_seen` 已在每帧检测时实时更新（via `update_appearance()`），提供全局"最后出现时间戳"。AlertManager 只跟踪本次运行中实际出现过或进入 watch 的身份，避免持久化身份在服务重启后形成批量补报警。

### 2. 检测时机
在 `AlertManager.tick()` 中新增 `_check_scene_exits()` 方法，每 2 秒扫描一次身份库。

### 3. 触发逻辑
```python
for each global_id in identity_store:
    elapsed = now - rec.last_seen

    # 尚未达到阈值 → 等待；真实重新出现由 pipeline 调用 mark_seen()
    if elapsed < scene_exit_seconds:
        continue

    # 正在被拓扑路径监听 → 交给 MISSING_PERSON 处理，避免冲突
    if gid in _watches:
        continue

    # MISSING_PERSON 刚触发 → 再等待一个 scene_exit_seconds 升级周期
    if gid in _missing_person_alerted and now - _missing_person_alerted[gid] < scene_exit_seconds:
        continue

    # 同一轮已经报过 SCENE_EXIT → 不重复
    if gid in _scene_exit_alerted:
        continue

    # ✅ 触发 SCENE_EXIT 告警
    _scene_exit_alerted[gid] = now
    fire_alert(...)
```

### 4. 与现有机制的关系
| 场景 | 告警类型 | 触发时机 |
|---|---|---|
| 拓扑路径监控 (cam_01→cam_02) | `MISSING_PERSON` | 45s 内未到达下一跳 |
| 末端摄像头 (cam_02/cam_03) | `SCENE_EXIT` | 连续 300s 未在任何摄像头出现 ✅ |
| 拓扑路径超时后继续失联 | `SCENE_EXIT` | MISSING_PERSON 触发后，5 分钟仍未出现 |

**互斥逻辑**：如果人员正在被拓扑路径监听（`_watches` 中存在），跳过 SCENE_EXIT 检查，避免重复告警。

**最终并发约束**：`mark_seen()`、`watch()`、`resolve()` 都会推进身份的 observation generation。ticker 在读取 IdentityStore 快照后、正式选中告警前再次核验 generation；若人员恰好重新出现，旧快照不能触发 SCENE_EXIT。

**重启约束**：AlertManager 不直接扫描所有持久化历史身份。只有本次进程已通过 `mark_seen()`、`watch()` 或 `resolve()` 激活的身份才进入 Scene Exit 候选集合，防止重启后对历史身份产生告警风暴。

---

## 实现清单

### 文件 1: `src/alerter.py`
**修改点 1** — `__init__` 签名扩展：
```python
def __init__(
    self,
    alert_log: str | Path = "outputs/alerts.jsonl",
    notifier=None,
    broadcaster: AlertBroadcaster | None = None,
    identity_store = None,              # 新增：IdentityStore 实例（避免循环导入用 Any）
    scene_exit_seconds: float = 300.0,  # 新增：场景消失阈值（秒）
):
    ...
    self._identity_store = identity_store
    self._scene_exit_seconds = scene_exit_seconds
    self._scene_exit_alerted: dict[str, float] = {}  # gid → 已报警时间戳
    self._missing_person_alerted: dict[str, float] = {}  # gid → MISSING 触发时间
    self._scene_exit_tracked: set[str] = set()            # 本次运行候选
    self._scene_exit_generation: dict[str, int] = {}      # 并发快照版本
```

**修改点 2** — `tick()` 末尾新增调用：
```python
def tick(self, screenshot_dir: Path | None = None) -> list[dict]:
    ...
    # 现有 WARNING/ALERT 逻辑
    ...

    # 场景消失检测（新增）
    if self._identity_store is not None:
        triggered_alerts.extend(self._check_scene_exits(now))

    return triggered_alerts
```

**修改点 3** — 新增方法 `_check_scene_exits()`：
```python
def _check_scene_exits(self, now: float) -> list[dict]:
    """检测从所有摄像头消失超过阈值时间的人员"""
    triggered_alerts = []
    with self._lock:
        tracked = [(gid, self._scene_exit_generation.get(gid, 0))
                   for gid in self._scene_exit_tracked]

    for gid, generation in tracked:
        rec = self._identity_store.get(gid)
        if rec is None or rec.last_seen == 0.0:
            continue

        elapsed = now - rec.last_seen

        # 未满阈值只等待；真正重新出现由 mark_seen() 清理状态
        if elapsed < self._scene_exit_seconds:
            continue

        with self._lock:
            missing_at = self._missing_person_alerted.get(gid)
            if self._scene_exit_generation.get(gid, 0) != generation:
                continue
            if gid in self._watches:
                continue
            if missing_at is not None and now - missing_at < self._scene_exit_seconds:
                continue
            if gid in self._scene_exit_alerted:
                continue
            self._scene_exit_alerted[gid] = now

        alert = {
            "alert_id": _new_alert_id("alert_exit"),
            "alert_type": "SCENE_EXIT",
            "stage": "ALERT",
            "global_id": gid,
            "last_camera": rec.last_camera,
            "last_seen": rec.last_seen,
            "last_bbox": self._identity_store.get_last_bbox(gid),
            "elapsed_seconds": round(elapsed, 1),
            "risk_level": "MEDIUM",
            "timestamp": now,
            "expected_cameras": ["全域监控区"],  # 占位符，保持前端字段完整
        }

        self._write_log(alert)
        logger.warning(
            "👻 SCENE_EXIT: 人员 %s 已从所有摄像头消失 %.0f 秒（最后位置: %s）",
            gid, elapsed, rec.last_camera,
        )

        if self._broadcaster:
            self._broadcaster.push(alert)
        if self._notifier:
            try:
                self._notifier.send(alert)
            except Exception as e:
                logger.error("通知发送失败: %s", e)

        triggered_alerts.append(alert)

    return triggered_alerts
```

---

### 文件 2: `main.py`
**修改点** — 第 92-96 行，传递新参数：
```python
alert_manager = AlertManager(
    alert_log=ALERT_LOG,
    notifier=notifier,
    broadcaster=broadcaster,
    identity_store=identity_store,   # 新增
    scene_exit_seconds=300.0,        # 新增：5 分钟阈值
)
```

---

### 文件 3: `static/js/modules/websocket.js`
**修改点** — `addAlert()` 函数第 48-77 行，扩展告警标题逻辑：
```javascript
const isWarning = alert.stage === 'WARNING';
const isIntrusion = alert.alert_type === 'INTRUSION';
const isCrowd = alert.alert_type === 'CROWD_DENSITY';
const isSceneExit = alert.alert_type === 'SCENE_EXIT';  // 新增

// ...计数器逻辑...

let titleText;
if (isIntrusion)      titleText = 'ROI 区域非法越界入侵';
else if (isCrowd)     titleText = '区域人流密度超限预警';
else if (isSceneExit) titleText = '目标全域场景消失失联';  // 新增
else if (isWarning)   titleText = '路径通行超时预警';
else                  titleText = '目标通行超时失联';
```

---

## 验证方案与结果

### 自动化状态机验证

- 末端摄像头：299.9 秒不触发，300.0 秒触发一次；人员重新出现后可在下一次消失周期再次触发。
- 活跃拓扑 watch：即使全域 `last_seen` 已超过 300 秒，也由 MISSING 机制独占，不提前发 SCENE_EXIT。
- 拓扑升级：45 秒触发 MISSING 后，同一 tick 不发 SCENE_EXIT；再持续失联 299.9 秒仍不发，满 300 秒后升级一次。
- 持久化恢复：启动时的陈旧身份不会报警；本次运行实际出现后才纳入候选。
- 并发重现：ticker 读取旧快照期间若 pipeline 调用 `mark_seen()`，generation 变化会阻止旧快照报警。

### 交付链验证

- `tick()` 返回、SQLite、JSONL、WebSocket recent 和 JPEG 文件名使用同一个 `alert_id`。
- JSONL 真实写入 `"alert_type": "SCENE_EXIT"`；SQLite 可按 Global ID 查询同一事件；存在最新帧时生成可解码 JPEG 和 `screenshot_url`。
- Playwright 注入真实结构事件后，主列表显示“目标全域场景消失失联”；同一 ID 第二次被拒绝，累计只增加 1。
- 详情 dialog 正确显示 camera、Global ID、300 秒、`全域监控区` 和无快照降级状态。

### 对应测试

- `tests.test_core_logic.AlertStateTests`
- `tests.test_persistence_alerts.AlertPersistenceTests.test_scene_exit_uses_one_id_across_all_delivery_channels`
- 全量命令：`python -m unittest -v`

---

## 后续优化空间（非本次交付范围）

以下项目需要新增产品配置、人员分级或交互设计，不属于本方案“全域消失告警”基础交付和验收范围：
- 将 `scene_exit_seconds` 写入配置文件（如 `config/alert.json`），支持运行时调整
- 为不同风险等级的人员设置差异化阈值（VIP 1 分钟，普通人员 5 分钟）
- 前端展示"失联倒计时"动画（类似 WARNING 阶段进度条）
