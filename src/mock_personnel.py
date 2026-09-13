"""
mock_personnel.py — 行人姓名模拟数据生成（批次五）。

为什么需要它
------------
`personnel` / `personnel_photos` 实测 0 行、6 个身份里 0 个绑名：批次三的实名档案
与批次四的视频检索**从来没有真实数据流过**。没有数据就无法验收这两批交付，
也无法继续做缩略图与以图搜人。本模块负责"造人"，`scripts/seed_personnel_mock.py`
负责落库。

三条不能妥协的约束（来自 docs/PLAN_2026-09-12_personnel_mock_data.md §0.1 实测）
-------------------------------------------------------------------------------
1. **特征必须让 `IdentityStore._restore()` 认出来。** 它做 `feature_space` **全等**
   比较 + `len(blob) == dim*4` 校验（identity_store.py:218-224），不满足的行被静默跳过，
   表现为"库里有人但 `/api/search/person` 404"（server.py:1023 只查内存 store）。
   → 所以 `feature_space` 由调用方从 `get_reid_weight(None).feature_space` 传入，
     **本模块不硬编码那个字符串**（换权重后它会变成 msmt17/dukemtmcreid）。
2. **时间戳必须在保留期内。** `main.py:285` 每次启动无条件跑 `apply_retention(30)`，
   按墙钟 timestamp 删 appearances / identities。
   → 默认只生成 now-7d 之内的数据，并提供 shift 平移。
3. **相机 id 必须来自 sources.json。** `search.py:81-86` 会把孤儿相机整行丢弃。
   → 行走链只在传入的 `known_cameras` 图上走。

本模块**故意不 import torch / cv2 / PIL**：姓名与轨迹的生成必须能在无模型环境下
跑单测，也保证 seed 脚本不会因为重量依赖而起不来。图片产出由调用方负责。

合成特征的诚实声明
------------------
`synth_feature()` 出来的向量彼此近似正交，1:N 匹配会**表现完美**。那是随机数的
性质，不是识别能力（真实语料 45 个身份间余弦 p50 曾达 0.984，见
docs/PLAN_2026-09-12_identity_search_resolution.md:15）。用它演示检索精度 = 造假。
→ source 必须写 'synthetic'，前端强制打 SIM 徽标。要演示真实识别请走 'real' 模式。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# 词库（不引 faker：requirements.txt 里没有它，且中文人名词库自己写更可控）        #
# --------------------------------------------------------------------------- #

def _chars(text: str) -> tuple[str, ...]:
    """把连续中文字符串拆成单字元组（姓氏表用，避免手抄漏分隔符）。"""
    return tuple(ch for ch in text if not ch.isspace())


#: 常见单字姓（百家姓主干 + 现役大姓）。
SURNAMES: tuple[str, ...] = _chars(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮齐康伍余元卜顾孟平黄"
    "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计成戴谈宋茅庞熊纪舒屈项祝董梁"
    "杜阮蓝闵季贾路娄江童颜郭梅盛林刁钟徐邱高夏蔡田樊胡凌霍虞万支柯"
    "管卢莫房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆"
    "荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车"
    "侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶"
    "郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴鬱胥能苍双闻莘党"
    "翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑桂濮寿牛通边扈燕冀郏浦尚农温别"
    "庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡"
    "国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空"
    "曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
)

#: 复姓。出现概率刻意压低（约 4%），避免"欧阳/上官"刷屏显得不像真人。
COMPOUND_SURNAMES: tuple[str, ...] = (
    "欧阳 太史 端木 上官 司马 东方 独孤 南宫 万俟 闻人 夏侯 诸葛 尉迟 公羊 鲜于 "
    "闾丘 司徒 司空 亓官 司寇 巫马 公西 颛孙 公良 漆雕 乐正 宰父 谷梁 西门 第五 "
    "纳兰 长孙 宇文 令狐 慕容 轩辕 宗政 濮阳 大司 东郭".split()
)

#: 单字名用字
GIVEN_1: str = (
    "伟芳娜敏静丽强磊军洋勇艳杰娟涛明超秀霞平刚桂英华玉萍红飞玲斌辉欣"
    "瑞诚洁雪琴岚峰波亮倩颖晨悦然浩瀚宇轩泽晗菲莉莎曼柔嘉彤迪凯越楠毅"
)

#: 双字名（更贴近真实命名分布：当代 2 字名占多数）
GIVEN_2: tuple[str, ...] = (
    "建国 建军 建华 志强 志明 志刚 秀英 秀兰 秀珍 凤英 海燕 海峰 海波 文博 文轩 文杰 "
    "俊杰 英杰 子轩 子涵 子墨 梓涵 梓萱 梓豪 一诺 依诺 思远 思琪 思涵 雨桐 雨欣 雨泽 "
    "明辉 明杰 晓明 晓峰 晓东 晓燕 国庆 国栋 国华 春梅 春燕 春花 海涛 海军 立新 建新 "
    "建文 建平 红梅 红兵 玲玉 龙飞 飞龙 天龙 天宇 天翼 浩然 浩宇 皓轩 皓月 佳琪 佳怡 "
    "佳明 佳伟 嘉怡 嘉豪 嘉懿 宇轩 宇航 泽宇 泽阳 泽楷 睿渊 睿哲 致远 雅静 雅婷 雅琳 "
    "梦洁 梦琪 梦璐 诗涵 诗琪 语嫣 语桐 欣怡 欣妍 文博 伟诚 磊鑫 弘博 兴涛 冠宇"
).split()

#: 部门取自 camera_map.json 里的真实区域名 + 超算中心常见科室，保证"看起来是这个单位的"
DEPARTMENTS: tuple[str, ...] = (
    "高性能机房", "网络运维", "系统软件", "存储系统",
    "动力保障", "安全管理", "科研支持", "行政后勤",
)

#: 工号前缀。docs/PLAN_2026-09-12_identity_search_resolution.md:343 用的是 "QLU-1234"，
#: 这里统一成 "QLU-<YY>-<4 位序号>" 以便肉眼读、口播、按年份分组。
EMPLOYEE_NO_PREFIX = "QLU"

#: 展示用的画面尺寸。必须与 frame_hub 推送给前端的一致，否则详情页框选位置会偏。
DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 270

#: 「垃圾桶身份」护栏。实测 identities 里有单身份 total_appearances=1561 的聚合行
#: （c156549c），那是 ReID 塌缩时期把多人吸进一个身份的产物，给它绑名 = 给一群人起名。
#: 绑名前必须过这道闸（见 filter_bindable_identities）。
AGGREGATE_IDENTITY_THRESHOLD = 300


# --------------------------------------------------------------------------- #
# 数据结构                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Appearance:
    """一条 identity_appearances 增量行（落库前的中间态）。"""
    global_id: str
    camera_id: str
    timestamp: float
    bbox: list[float]
    asset_id: int | None
    video_frame: int | None
    video_ts: float | None

    def as_new_appearance_dict(self) -> dict:
        """适配 Database.save_identity(new_appearance=...) 的键名（camera/time/bbox）。"""
        return {
            "camera": self.camera_id,
            "time": self.timestamp,
            "bbox": self.bbox,
            "asset_id": self.asset_id,
            "video_frame": self.video_frame,
            "video_ts": self.video_ts,
        }


@dataclass
class MockIdentity:
    """一个匿名身份（gid）。一个人 1~3 个，模拟"同一个人被拆成多个身份"的真实形态。"""
    global_id: str
    feature: np.ndarray
    feature_space: str
    feature_dim: int
    appearances: list[Appearance] = field(default_factory=list)
    name_confidence: float = 0.85

    @property
    def total_appearances(self) -> int:
        return len(self.appearances)

    def blob(self) -> bytes:
        return self.feature.astype(np.float32).tobytes()


@dataclass
class MockPerson:
    """一个人 = 1 条 personnel + 1~3 个 MockIdentity。"""
    person_id: str
    name: str
    employee_no: str
    department: str
    note: str
    source: str
    identities: list[MockIdentity] = field(default_factory=list)
    thumb_path: str | None = None
    gallery_feature: np.ndarray | None = None

    @property
    def total_appearances(self) -> int:
        return sum(len(i.appearances) for i in self.identities)

    @property
    def last_seen(self) -> float:
        stamps = [a.timestamp for i in self.identities for a in i.appearances]
        return max(stamps) if stamps else 0.0


# --------------------------------------------------------------------------- #
# 姓名生成                                                                       #
# --------------------------------------------------------------------------- #

def make_name(rng: np.random.Generator, used: set[str] | None = None,
              max_attempts: int = 400) -> str:
    """
    随机中文姓名。约 55% 双字名、约 40% 单字名、约 4% 复姓。

    传入 used 则保证不与已用名重复（并把新名写回 used）。超出 max_attempts 时
    退化为"名 + 数字后缀"，避免词库容量不足导致死循环。
    """
    for _ in range(max_attempts):
        if rng.random() < 0.04:
            surname = str(rng.choice(COMPOUND_SURNAMES))
        else:
            surname = str(rng.choice(SURNAMES))
        if rng.random() < 0.55:
            given = str(rng.choice(GIVEN_2))
        else:
            given = str(rng.choice(list(GIVEN_1)))
        name = f"{surname}{given}"
        if used is None or name not in used:
            if used is not None:
                used.add(name)
            return name
    # 兜底：加数字后缀（真实单位重名靠工号区分，这里只为避免 seed 中断）
    base = make_name(rng, None)
    suffix = 1
    while used is not None and f"{base}{suffix}" in used:
        suffix += 1
    name = f"{base}{suffix}" if used is not None else base
    if used is not None:
        used.add(name)
    return name


def make_employee_no(index: int, year_two_digits: int) -> str:
    """可读工号，如 QLU-26-0007。person_id 保持 uuid 短哈希不变（见方案 §3.1）。"""
    return f"{EMPLOYEE_NO_PREFIX}-{year_two_digits:02d}-{index:04d}"


def make_person_id(rng: np.random.Generator | None = None) -> str:
    """与 PersonnelGallery.create 同口径：uuid4 前 8 位（personnel.py:148）。"""
    return uuid.uuid4().hex[:8]


def make_global_id() -> str:
    """与 IdentityStore.register 同口径：uuid4 前 8 位（identity_store.py:665）。"""
    return str(uuid.uuid4())[:8]


# --------------------------------------------------------------------------- #
# 轨迹生成                                                                       #
# --------------------------------------------------------------------------- #

def build_adjacency(topology: dict, known_cameras: Iterable[str]) -> dict[str, list[dict]]:
    """
    过滤拓扑图，只保留 sources.json 里真实存在的相机。

    不这么做的后果是实测过的：库里留有 rnd_03 的 6,447 条孤儿轨迹（search.py:73 注释），
    检索返回一个已下线相机，前端点进去 404。
    """
    allowed = set(known_cameras)
    adjacency: dict[str, list[dict]] = {}
    for from_cam, edges in (topology or {}).items():
        if from_cam not in allowed:
            continue
        keep = [e for e in (edges or []) if e.get("next") in allowed]
        if keep:
            adjacency[from_cam] = keep
    return adjacency


def pick_start_camera(adjacency: dict[str, list[dict]],
                      all_cameras: list[str],
                      rng: np.random.Generator) -> str:
    """优先从有出边的相机起步（否则走两步就断链）。"""
    nodes = sorted(adjacency) or sorted(all_cameras)
    return str(rng.choice(nodes)) if nodes else ""


def make_walking_chain(adjacency: dict[str, list[dict]],
                       all_cameras: list[str],
                       rng: np.random.Generator,
                       n_stops: int) -> list[str]:
    """
    沿真拓扑随机游走，返回相机 id 序列（长度 = n_stops）。

    相邻两站的间隔由 make_visit_timestamps 用 expected_seconds ± tolerance_seconds 决定，
    所以这里只负责"顺序合法"。走到死路就随机换一路继续（现实中人会从楼梯折返，
    拓扑图本来也不完备）。
    """
    chain: list[str] = []
    current = pick_start_camera(adjacency, all_cameras, rng)
    if not current:
        return chain
    for _ in range(max(1, n_stops)):
        chain.append(current)
        edges = adjacency.get(current) or []
        # 不立刻回头走同一对边（避免出现 A-B-A-B-A 这种一眼假的乒乓）
        candidates = [e for e in edges if not chain or e["next"] != (chain[-2] if len(chain) >= 2 else None)]
        candidates = candidates or edges
        if not candidates:
            current = pick_start_camera(adjacency, all_cameras, rng)
            continue
        edge = candidates[int(rng.integers(0, len(candidates)))]
        current = str(edge["next"])
    return chain


def make_bbox(rng: np.random.Generator,
              direction: str = "walk",
              width: int = DISPLAY_WIDTH,
              height: int = DISPLAY_HEIGHT) -> list[float]:
    """
    生成一个合法的人物 bbox（x1<x2, y1<y2, 不越界）。

    尺寸按 480x270 的展示分辨率估：真人占画面高约 1/3 ~ 2/3。
    direction 只影响横向起终点（让"行走"看起来有方向感），不追求物理精确。
    """
    person_h = float(rng.uniform(height * 0.30, height * 0.66))
    person_w = person_h * float(rng.uniform(0.32, 0.46))
    person_w = min(person_w, width * 0.5)
    y1 = float(rng.uniform(0.0, max(1.0, height - person_h)))
    x1 = float(rng.uniform(0.0, max(1.0, width - person_w)))
    if direction == "left":
        x1 = max(0.0, width - person_w - x1)
    return [round(x1, 1), round(y1, 1),
            round(x1 + person_w, 1), round(y1 + person_h, 1)]


def _bbox_along_path(start: list[float], rng: np.random.Generator,
                     steps: int, width: int = DISPLAY_WIDTH,
                     height: int = DISPLAY_HEIGHT) -> list[list[float]]:
    """从 start 出发沿水平方向平移出 steps 个 bbox，末帧仍在画面内。"""
    w = start[2] - start[0]
    h = start[3] - start[1]
    direction = 1.0 if rng.random() < 0.5 else -1.0
    speed = float(rng.uniform(6.0, 22.0))          # px / 采样点
    boxes: list[list[float]] = []
    x = start[0]
    y = start[1]
    for _ in range(max(1, steps)):
        x = x + direction * speed
        if x < 0 or x + w > width:
            direction *= -1
            x = min(max(x, 0.0), width - w)
        y = min(max(y + float(rng.normal(0, 1.5)), 0.0), height - h)
        boxes.append([round(x, 1), round(y, 1),
                      round(x + w, 1), round(y + h, 1)])
    return boxes


def make_visit_timestamps(edge: dict | None, rng: np.random.Generator,
                          dwell_rows: int, sample_interval: float) -> tuple[float, float]:
    """
    返回 (进入某相机的时刻偏移, 在该相机停留的秒数)。

    间隔优先用拓扑标称值 expected_seconds ± tolerance_seconds —— 这样模拟数据的
    跨相机时序与 alerter 的 MISSING_PERSON 判据是同一套口径，不会造出"永远不该报警"
    或"一直在报警"的数据。
    """
    expected = float((edge or {}).get("expected_seconds", 60.0) or 60.0)
    tolerance = float((edge or {}).get("tolerance_seconds", 30.0) or 30.0)
    gap = max(sample_interval, expected + float(rng.uniform(-tolerance, tolerance)))
    dwell = max(sample_interval * 2, float(dwell_rows) * sample_interval)
    return gap, dwell


def make_person_appearances(
    global_id: str,
    chain: list[str],
    adjacency: dict[str, list[dict]],
    assets_by_camera: dict[str, dict],
    start_ts: float,
    rng: np.random.Generator,
    loops: int = 3,
    rows_per_dwell: tuple[int, int] = (4, 10),
    sample_interval: float = 1.0,
) -> list[Appearance]:
    """
    为单个身份生成轨迹行。

    loops 的作用：素材只有几十秒~8 分钟却被反复回放，同一段像素会被记录多次。
    把 chain 重复 loops 遍（每遍整体平移一个"素材周期"），search.py:88-104 才能算出
    有意义的 loop_factor，前端才会显示"循环 xN"而不是"出现 3 万次"。

    每一行都带 asset_id / video_frame / video_ts（方案 §6.1：绝不留 NULL 走
    _resolve_asset 的 fallback，否则 video_ts 会定位到另一个文件上）。
    """
    appearances: list[Appearance] = []
    if not chain:
        return appearances

    ts = start_ts
    for loop_idx in range(max(1, int(loops))):
        for idx, camera in enumerate(chain):
            asset = assets_by_camera.get(camera) or {}
            duration = float(asset.get("duration_real") or 0.0)
            fps = float(asset.get("fps_declared") or 25.0) or 25.0
            frames_real = int(asset.get("frames_real") or 0)

            edge = None
            if idx > 0:
                prev_edges = adjacency.get(chain[idx - 1]) or []
                edge = next((e for e in prev_edges if e.get("next") == camera), None)
            gap, _dwell = make_visit_timestamps(edge, rng, rows_per_dwell[0], sample_interval)
            ts += gap if idx else 0.0

            rows = int(rng.integers(rows_per_dwell[0], rows_per_dwell[1] + 1))
            # 素材时长可能很短（rnd_05 实测 18.4 s），按实际可用秒数收敛采样点数
            if duration > 0:
                rows = max(2, min(rows, int(max(2.0, duration / max(sample_interval, 0.5)))))
            start_box = make_bbox(rng, direction="walk" if idx % 2 == 0 else "left")
            boxes = _bbox_along_path(start_box, rng, rows)
            for row_idx in range(rows):
                row_ts = ts + row_idx * sample_interval
                if duration > 0:
                    # 视频内位置：在整个素材时长内均匀铺开，按 loop 错开起点
                    video_ts = ((row_idx * sample_interval + loop_idx * 7.3)
                                % duration)
                    video_frame = int(min(frames_real - 1, video_ts * fps)) if frames_real else None
                else:
                    video_ts = None
                    video_frame = None
                appearances.append(Appearance(
                    global_id=global_id,
                    camera_id=camera,
                    timestamp=round(row_ts, 3),
                    bbox=boxes[row_idx],
                    asset_id=asset.get("asset_id"),
                    video_frame=video_frame,
                    video_ts=round(video_ts, 3) if video_ts is not None else None,
                ))
            ts += (rows - 1) * sample_interval
    return appearances


# --------------------------------------------------------------------------- #
# 合成特征（仅 'synthetic' 模式）                                                 #
# --------------------------------------------------------------------------- #

def synth_feature(rng: np.random.Generator, dim: int = 512) -> np.ndarray:
    """
    生成一个**人**的基准特征向量（L2 归一化随机数）。

    每个 gid 的特征由它经 jitter_feature() 派生 —— 不要直接拿它当身份特征，
    那会让同一人的多个身份互相正交、看起来像不同人（见 jitter_feature 说明）。

    ⚠ 跨人近似正交 → 匹配会"指哪打哪"。这只用于打通链路与演示界面，
      不能用于评估识别精度（见模块 docstring）。
    """
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


def jitter_feature(base: np.ndarray, rng: np.random.Generator,
                   scale_range: tuple[float, float] = (0.30, 0.55)) -> tuple[np.ndarray, float]:
    """
    在"本人基准向量"上加噪声派生出单个身份的特征，返回 (归一化特征, 与基准的余弦)。

    为什么必须有这一步：每个人的各身份如果用**互相独立**的随机向量，
    那么任何身份都匹配不上自己的档案（余弦 ≈ 0 < 0.68），
    `PersonnelGallery.match()` 恒为 None —— 自动命名链路在模拟数据上**根本不会触发**，
    演示时就只剩"手工绑定"一条路，批次三的能力一等于没演示。
    派生后同一人的多个 gid 聚成一簇、跨人仍近似正交，链路才成立。

    scale 与余弦的关系（d 维标准正态 n，噪声 = n/|n| · scale，即**单位方向**上按长度缩放）：
      cos = 1/sqrt(1+scale²)  →  scale 0.30 → 0.958   scale 0.55 → 0.871
    都落在真实系统会命中的区间（阈值 0.68），且 <0.99 不至于假到一眼看穿。

    ⚠ 必须先归一化 noise 再乘 scale。直接 `base + rng.standard_normal(d) * scale` 是错的：
      d=512 时 |standard_normal| ≈ √512 ≈ 22.6，等效 scale 高达 6.8，
      实测余弦掉到 0.154 —— 身份匹配不上自己的档案，链路静默失效。
    """
    scale = float(rng.uniform(*scale_range))
    noise = rng.standard_normal(base.shape).astype(np.float32)
    noise_norm = float(np.linalg.norm(noise))
    if noise_norm <= 1e-8:
        return base.copy(), 1.0
    vec = base + (noise / noise_norm) * scale
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return base.copy(), 1.0
    vec = vec / norm
    return vec, float(np.dot(vec, base))


# --------------------------------------------------------------------------- #
# 组装：一次生成 N 个人                                                          #
# --------------------------------------------------------------------------- #

def generate_persons(
    count: int,
    *,
    known_cameras: list[str],
    topology: dict,
    assets_by_camera: dict[str, dict],
    feature_space: str,
    feature_dim: int = 512,
    rng: np.random.Generator,
    source: str = "synthetic",
    window_start: float,
    window_span: float = 6 * 86400.0,
    ids_per_person: tuple[int, int] = (1, 3),
    stops_per_visit: tuple[int, int] = (3, 6),
    loops: int = 2,
    year_two_digits: int = 26,
    note_tag: str = "模拟数据 · 批次五",
    align_tail_to: float | None = None,
) -> list[MockPerson]:
    """
    生成 count 个人（含档案、1~3 个身份、轨迹行）。不落库，纯内存结构。

    window_start / window_span 控制墙钟落点（约束 2：必须在 apply_retention 窗口内）。

    align_tail_to：把**最晚**一条轨迹对齐到该时刻（默认不处理）。
    不处理的话，"每人随机起点"会让最活跃的人停在 1 天前，前端"最近出现"整列
    全是"1 天前~7 天前"，看起来像系统坏了；对齐后至少有人是"几分钟前"。
    平移是整体等量的，不改变人与人、相机与相机之间的相对时序。
    """
    adjacency = build_adjacency(topology, known_cameras)
    used_names: set[str] = set()
    persons: list[MockPerson] = []

    for index in range(1, max(0, int(count)) + 1):
        person_id = make_person_id(rng)
        name = make_name(rng, used_names)
        department = str(rng.choice(DEPARTMENTS))
        gid_count = int(rng.integers(ids_per_person[0], ids_per_person[1] + 1))
        # 每人一条"活动区间"起点，窗口内均匀铺开，让"最近出现"有区分度
        person_start = window_start + float(rng.uniform(0.0, max(1.0, window_span - 3600.0)))

        identities: list[MockIdentity] = []
        # 先定"这个人"的基准向量，各身份由它派生（见 jitter_feature 说明）。
        # 注册照直接用基准向量：等价于"底库照与抓拍是同一个人"的理想情形。
        base_feature = synth_feature(rng, dim=feature_dim)
        for _ in range(gid_count):
            gid = make_global_id()
            n_stops = int(rng.integers(stops_per_visit[0], stops_per_visit[1] + 1))
            chain = make_walking_chain(adjacency, known_cameras, rng, n_stops)
            appearances = make_person_appearances(
                gid, chain, adjacency, assets_by_camera, person_start, rng, loops=loops,
            )
            feature, cosine = jitter_feature(base_feature, rng)
            identities.append(MockIdentity(
                global_id=gid,
                feature=feature,
                feature_space=feature_space,
                feature_dim=feature_dim,
                appearances=appearances,
                # 与真实链路同口径：自动命名时 name_confidence 就是匹配得分
                # （personnel._merge_locked 用 score 写入），所以这里直接用余弦，
                # 而不是另抽一个随机数 —— 否则前端显示的置信度与任何计算都对不上。
                name_confidence=round(cosine, 3),
            ))

        persons.append(MockPerson(
            person_id=person_id,
            name=name,
            employee_no=make_employee_no(index, year_two_digits),
            department=department,
            note=f"{note_tag} · {department}",
            source=source,
            identities=identities,
            gallery_feature=base_feature.copy(),
        ))

    if align_tail_to is not None:
        stamps = [a.timestamp for p in persons for i in p.identities for a in i.appearances]
        if stamps:
            shift = float(align_tail_to) - max(stamps)
            for p in persons:
                for ident in p.identities:
                    for a in ident.appearances:
                        a.timestamp = round(a.timestamp + shift, 3)
    return persons


def filter_bindable_identities(database, persons: list[MockPerson],
                              max_appearances: int = AGGREGATE_IDENTITY_THRESHOLD) -> int:
    """
    剥掉"垃圾桶身份"的绑定意图，返回被剥掉的数量。

    只作用于**真实库里已有**的身份（seed 自己造的新 gid 不受影响）。
    给聚合身份绑名 = 给一整个聚类起一个人名，是错误归因的直接来源。
    """
    stripped = 0
    existing: dict[str, int] = {}
    try:
        for row in database.load_identities():
            existing[row["global_id"]] = int(row.get("total_appearances") or 0)
    except Exception:  # noqa: BLE001 - 读不到就跳过这道保护，不阻断 seed
        return 0
    for person in persons:
        for identity in person.identities:
            if existing.get(identity.global_id, 0) > max_appearances:
                identity.name_confidence = 0.0
                stripped += 1
    return stripped
