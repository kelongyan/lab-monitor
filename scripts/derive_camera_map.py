"""
derive_camera_map.py — 从 desc + topology 自动推导摄像头点位坐标

为什么需要它
------------
`config/camera_map.json` 的 22 路 `map_xy` 全为 null（`plan_id` 空串、`facing_deg` 全 0），
而楼层平面图属客户图纸、不入库，所以轨迹路线图拿不到节点坐标。本脚本**不依赖任何图纸**，
只从现有配置推导出一版「拓扑正确、方位大致正确」的坐标，让轨迹图先能用；
日后拿到图纸再用手工标注（outputs/dev/calibrate_floorplan.html）逐点覆盖。

推导依据（两条都是实测结论，不是猜测）
--------------------------------------
1. **desc 编码了位置**：命名规则是 `L2{区域}{走廊走向}{相机朝向}`，例如
   「L2东侧走廊南北向南」= L2 层 / 东侧走廊 / 走廊南北向 / 相机朝南。
   按区域关键词匹配即可 100% 归组（实测 7 个区域、22 台全部命中）。
2. **topology 是巡检路线图**：`expected_seconds <= 10` 的边连接的是**同一条走廊上
   两个朝向互拍的相机**（共位反向对），物理上是同一个位置。把它们合并成「站点」后
   22 台相机只剩 15 个物理点位 —— 这是轨迹图清晰度的关键。

坐标系
------
与 `src/floorplan.py` 完全一致：
- `map_xy = [x, y]` 归一化到 [0,1]，原点左上，x 向右、y 向下
- 同一站点内的多台相机写入**同一坐标**（前端按拓扑合并成单节点后取该坐标）

用法
----
    python scripts/derive_camera_map.py                  # 只打印，不写盘（默认）
    python scripts/derive_camera_map.py --write          # 写回 config/camera_map.json
    python scripts/derive_camera_map.py --json out.json  # 额外导出站点结构供前端/调试用
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

# Windows 控制台默认 GBK，打印中文区域名会炸（项目 AGENTS.md 记过这个坑）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMERA_MAP_FILE = PROJECT_ROOT / "config" / "camera_map.json"
TOPOLOGY_FILE = PROJECT_ROOT / "config" / "topology.json"

#: 判定「共位反向相机对」的时延上限（秒）。
#: topology.json 里的语义分层非常干净：5s 表示同点位两个朝向互拍，
#: 20s 起步才是真正的位移。取 10 作为分界，两边都有余量。
COPLANAR_MAX_SECONDS = 10.0

#: L2 层平面近似布局（归一化 [0,1]，原点左上，x 向右、y 向下）。
#:
#: 依据 desc 里的方位词推断的真实楼面结构：
#:   东/南/西/北侧走廊环绕一圈，中间走廊横穿东西，两处机房通道在楼面南端。
#:
#: 每个区域占一条**横向带状区域** `(x_start, x_end, y)`，区域内的站点在该带内
#: 水平等距展开（见 _layout_region）。用「带」而不是「单点锚点」的原因：
#: 单点锚点只约束区域中心，西侧(2 站)/中间(3 站)/东侧(2 站) 三个区域的站点
#: 会全部落在 y≈0.40 这一行上，7 个节点横向挤压、彼此重叠（实测截图可见）。
#: 带状定义把「西侧贴左、东侧贴右、中间居中」的物理关系直接表达出来。
#:
#: 改图纸/换楼层时只需改这张表，推导逻辑不用动。
REGION_BANDS: dict[str, tuple[float, float, float]] = {
    "北侧走廊": (0.35, 0.65, 0.08),
    "西侧走廊": (0.02, 0.38, 0.30),
    "东侧走廊": (0.62, 0.98, 0.30),
    "中间走廊": (0.25, 0.75, 0.50),
    "南侧走廊": (0.33, 0.67, 0.68),
    "高性能机房02通道": (0.02, 0.38, 0.88),
    "高性能机房04通道": (0.62, 0.98, 0.88),
}

#: 站点节点在画布上的归一化半宽（= NODE_W/2 / VIEW_W = 79/1000）。
#: 与前端 static/js/modules/traj_graph.js 的 NODE_W / VIEW_SIZE 对应，改动需同步。
SITE_HALF_WIDTH = 0.079

#: 无法归入任何已知区域的相机，统一放到这个兜底带（会打 warning）。
UNKNOWN_REGION_BAND = (0.35, 0.65, 0.55)
UNKNOWN_REGION_NAME = "未识别区域"


# --------------------------------------------------------------------------- #
# 推导                                                                        #
# --------------------------------------------------------------------------- #


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_region(desc: str) -> str | None:
    """
    从中文描述里解析区域。

    必须按关键词长度倒序匹配：`高性能机房02通道` 与 `高性能机房04通道`
    共享前缀，先匹配短的会串区（虽然当前两个都写了全名，但 desc 编排规则
    以后可能简化成「机房02通道」这类写法，按长度排序能扛住）。
    """
    if not desc:
        return None
    for keyword in sorted(REGION_BANDS, key=len, reverse=True):
        if keyword in desc:
            return keyword
    return None


def find_coplanar_pairs(topology: dict, max_seconds: float = COPLANAR_MAX_SECONDS) -> list[tuple[str, str]]:
    """找出共位反向相机对（短边），按首次出现顺序去重，不排序以免丢失方向语义。"""
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for camera, hops in topology.items():
        for hop in hops:
            nxt = hop.get("next")
            if nxt is None:
                continue
            if float(hop.get("expected_seconds", 0) or 0) > max_seconds:
                continue
            key = frozenset((camera, nxt))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((camera, nxt))
    return pairs


def _neighbors(topology: dict, camera: str) -> tuple[set[str], set[str]]:
    """返回 (直接前驱集合, 直接后继集合)。"""
    successors = {hop["next"] for hop in topology.get(camera, []) if hop.get("next")}
    predecessors = {
        other
        for other, hops in topology.items()
        for hop in hops
        if hop.get("next") == camera
    }
    return predecessors, successors


def _is_parallel_group(topology: dict, group: tuple[str, ...]) -> bool:
    """
    组内相机是否在拓扑上**并列**（共享直接前驱或直接后继）。

    并列 + desc 相同 ⇒ 同一点位的两个朝向；只有 desc 相同但分属两条互不相连的
    路线时，不能认定同一点位 —— 命名粒度可能就只到走廊级。
    """
    if len(group) < 2:
        return False
    for i, left in enumerate(group):
        left_pred, left_succ = _neighbors(topology, left)
        for right in group[i + 1:]:
            right_pred, right_succ = _neighbors(topology, right)
            if (left_pred & right_pred) or (left_succ & right_succ):
                return True
    return False


def find_duplicate_desc_groups(
    camera_map: dict,
    topology: dict,
    exclude: set[str],
) -> list[tuple[str, ...]]:
    """
    desc 完全相同**且在拓扑上并列**的相机视为同一视点（例：rnd_21 / rnd_22
    都写「L2高性能机房04通道西南向北」，且共享前驱 rnd_04、共享后继 rnd_19）。

    这类相机靠时延判据抓不到（拓扑里不是 5s 短边而是并列分支），必须单独识别，
    否则同一位置会被画成两个节点，让「跨了几处点位」被高估。
    后端 snapshots 接口也提示过这组相机（server.py 的 docstring）。

    **同时需要并列约束**：`reg_08` 与 `rnd_10` 的 desc 也完全相同
    （都是「L2南侧走廊西西向东」），但它们分属常规环与随机链两条互不相连的路线，
    只能说明命名粒度到走廊级，不能据此认定同一点位 —— 早期版本漏了这条约束，
    把两者并成了一个站点。
    """
    by_desc: dict[str, list[str]] = {}
    for camera, meta in camera_map.items():
        if camera in exclude:
            continue
        desc = (meta or {}).get("desc") or ""
        if not desc:
            continue
        by_desc.setdefault(desc, []).append(camera)

    groups: list[tuple[str, ...]] = []
    for desc, cameras in sorted(by_desc.items()):
        if len(cameras) < 2:
            continue
        group = tuple(sorted(cameras))
        # 同 desc 超过 2 台时逐个拆出并列的子组，避免把三台不相关的相机捆在一起
        if len(group) == 2:
            if _is_parallel_group(topology, group):
                groups.append(group)
            continue
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                if _is_parallel_group(topology, (left, right)):
                    groups.append((left, right))
    return groups


def topology_order(topology: dict) -> list[str]:
    """
    沿拓扑边走一遍，得到相机的全局访问顺序（用于区域内站点排序）。

    图里存在环（reg_08 → reg_01 → reg_06 → reg_10 → reg_08），所以：
    先从入度为 0 的节点做 BFS（随机路线链的起点），再把环上和孤立的节点按 ID 追加。
    不追求严格的拓扑序，只要能给同区域的站点一个稳定、贴近动线的排列即可。
    """
    nodes: set[str] = set(topology)
    for hops in topology.values():
        for hop in hops:
            if hop.get("next"):
                nodes.add(hop["next"])

    indegree = {node: 0 for node in nodes}
    for camera, hops in topology.items():
        for hop in hops:
            nxt = hop.get("next")
            if nxt in indegree:
                indegree[nxt] += 1

    starts = sorted(node for node in nodes if indegree.get(node, 0) == 0)
    order: list[str] = []
    seen: set[str] = set()
    queue = list(starts)
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        for hop in topology.get(node, []):
            nxt = hop.get("next")
            if nxt and nxt not in seen:
                queue.append(nxt)
    for node in sorted(nodes):
        if node not in seen:
            order.append(node)
    return order


def build_sites(
    camera_map: dict,
    topology: dict,
) -> tuple[list[dict], list[str]]:
    """
    把相机图合并成站点列表。

    返回 (sites, warnings)；每个 site 形如
    `{"id", "cameras", "region", "anchor", "order_index"}`（坐标由 _layout 填）。
    """
    warnings: list[str] = []

    pairs = find_coplanar_pairs(topology)
    grouped: set[str] = {camera for pair in pairs for camera in pair}
    dup_groups = find_duplicate_desc_groups(camera_map, topology, grouped)

    sites: list[dict] = []
    for pair in pairs:
        cameras = list(pair)
        sites.append({"id": "+".join(cameras), "cameras": cameras})
    for group in dup_groups:
        grouped.update(group)
        sites.append({"id": "+".join(group), "cameras": list(group)})

    for camera in camera_map:
        if camera in grouped:
            continue
        sites.append({"id": camera, "cameras": [camera]})

    order = topology_order(topology)
    order_index = {camera: i for i, camera in enumerate(order)}

    for site in sites:
        regions = OrderedDict()
        for camera in site["cameras"]:
            desc = (camera_map.get(camera) or {}).get("desc") or ""
            region = detect_region(desc)
            if region is None:
                warnings.append(
                    f"{camera} 的 desc 无法识别区域（desc={desc!r}），已归入 {UNKNOWN_REGION_NAME}"
                )
                region = UNKNOWN_REGION_NAME
            regions.setdefault(region, 0)
            regions[region] += 1
        # 一个站点内的相机理论上必属同一区域；真出现跨区，按多数派归属并告警
        site["region"] = max(regions.items(), key=lambda kv: kv[1])[0]
        if len(regions) > 1:
            warnings.append(
                f"站点 {site['id']} 的相机跨区域 {list(regions)}，已按多数派归入 {site['region']}"
            )
        site["band"] = REGION_BANDS.get(site["region"], UNKNOWN_REGION_BAND)
        site["order_index"] = min(
            (order_index.get(camera, len(order)) for camera in site["cameras"]),
            default=len(order),
        )

    return sites, warnings


def layout_sites(sites: list[dict]) -> None:
    """
    就地写入 `map_xy`：在区域带内水平等距展开。

    每个区域占一条带 `(x_start, x_end, y)`，站点中心在
    `[x_start + half, x_end - half]` 区间内等距分布（half = 节点半宽），
    保证节点完整落在自己那条带里、不会越到邻区去。

    带与带之间是「不同 y」的关系，所以同 x 区间（如西侧与机房02）不会打架；
    同一行内只有互为左右对称的两个区域（西侧/东侧、机房02/机房04），
    它们的 x 区间天然分离（见 REGION_BANDS）。
    """
    by_region: dict[str, list[dict]] = OrderedDict()
    for site in sites:
        by_region.setdefault(site["region"], []).append(site)

    for region, members in by_region.items():
        members.sort(key=lambda s: (s["order_index"], s["id"]))
        x_start, x_end, y = REGION_BANDS.get(region, UNKNOWN_REGION_BAND)
        left = x_start + SITE_HALF_WIDTH
        right = x_end - SITE_HALF_WIDTH
        if right < left:  # 带比节点还窄，退化为居中（配置写错时不至于算出负间距）
            left = right = (x_start + x_end) / 2
        count = len(members)
        span = right - left
        for index, site in enumerate(members):
            offset = 0.0 if count == 1 else -span / 2 + index * (span / (count - 1))
            site["map_xy"] = [round((left + right) / 2 + offset, 4), round(y, 4)]


# --------------------------------------------------------------------------- #
# 输出                                                                        #
# --------------------------------------------------------------------------- #


def apply_to_camera_map(camera_map: dict, sites: list[dict]) -> dict:
    """在原有条目上**只改 `map_xy`**，其余字段（plan_id / desc / facing_deg / fov_deg）原样保留。"""
    coordinate = {}
    for site in sites:
        for camera in site["cameras"]:
            coordinate[camera] = site["map_xy"]

    updated: dict = OrderedDict()
    for camera, meta in camera_map.items():
        entry = OrderedDict()
        entry["plan_id"] = (meta or {}).get("plan_id", "")
        entry["desc"] = (meta or {}).get("desc", "")
        entry["map_xy"] = coordinate.get(camera, (meta or {}).get("map_xy"))
        entry["facing_deg"] = (meta or {}).get("facing_deg", 0)
        entry["fov_deg"] = (meta or {}).get("fov_deg", 80)
        updated[camera] = entry
    return updated


def write_atomic(path: Path, payload: dict) -> None:
    """原子写：tmp + os.replace（与 ROI / topology 写盘的既有约定一致）。"""
    import os

    temp = path.with_name(f".{path.name}.derive.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def report(camera_map: dict, sites: list[dict], pairs: list[tuple[str, str]], warnings: list[str]) -> None:
    print("=" * 72)
    print("区域分组")
    print("=" * 72)
    by_region: dict[str, list[str]] = OrderedDict()
    for camera, meta in camera_map.items():
        region = detect_region((meta or {}).get("desc") or "") or UNKNOWN_REGION_NAME
        by_region.setdefault(region, []).append(camera)
    for region, cameras in by_region.items():
        band = REGION_BANDS.get(region, UNKNOWN_REGION_BAND)
        print(f"  {region:<18} 带={band}  {len(cameras)} 台: {' '.join(cameras)}")

    print()
    print("=" * 72)
    print(f"共位反向相机对（拓扑 expected_seconds <= {COPLANAR_MAX_SECONDS:g}s）{len(pairs)} 组")
    print("=" * 72)
    for left, right in pairs:
        left_desc = (camera_map.get(left) or {}).get("desc", "")
        right_desc = (camera_map.get(right) or {}).get("desc", "")
        print(f"  {left:>8} + {right:<8}  {left_desc} / {right_desc}")

    print()
    print("=" * 72)
    print(f"站点表：{len(camera_map)} 台相机 → {len(sites)} 个站点")
    print("=" * 72)
    for index, site in enumerate(sorted(sites, key=lambda s: (s["band"][2], s["band"][0], s["id"])), 1):
        cameras = "+".join(site["cameras"])
        xy = site["map_xy"]
        print(f"  #{index:<3} {site['region']:<18} {cameras:<24} map_xy=[{xy[0]:.3f}, {xy[1]:.3f}]")

    if warnings:
        print()
        print("告警：")
        for item in warnings:
            print(f"  ! {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description="从 desc + topology 推导摄像头点位坐标")
    parser.add_argument("--write", action="store_true", help="写回 config/camera_map.json（默认只打印）")
    parser.add_argument("--json", metavar="PATH", help="额外导出站点结构 JSON（调试/前端常量用）")
    args = parser.parse_args()

    if not CAMERA_MAP_FILE.exists():
        print(f"找不到 {CAMERA_MAP_FILE}", file=sys.stderr)
        return 1
    if not TOPOLOGY_FILE.exists():
        print(f"找不到 {TOPOLOGY_FILE}", file=sys.stderr)
        return 1

    camera_map = load_json(CAMERA_MAP_FILE)
    topology = load_json(TOPOLOGY_FILE)

    already_mapped = [c for c, m in camera_map.items() if (m or {}).get("map_xy")]
    if already_mapped:
        print(
            f"提示：{len(already_mapped)} 台相机已有 map_xy（人工标注过）。"
            f"本脚本会**全部覆盖**，如需保留请先备份或用标注工具逐点修正。"
        )

    sites, warnings = build_sites(camera_map, topology)
    layout_sites(sites)
    pairs = find_coplanar_pairs(topology)
    report(camera_map, sites, pairs, warnings)

    if args.json:
        payload = {
            "regions": {name: list(band) for name, band in REGION_BANDS.items()},
            "sites": [
                {
                    "id": site["id"],
                    "cameras": site["cameras"],
                    "region": site["region"],
                    "map_xy": site["map_xy"],
                }
                for site in sites
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n站点结构已导出：{args.json}")

    if args.write:
        write_atomic(CAMERA_MAP_FILE, apply_to_camera_map(camera_map, sites))
        print(f"\n已写回：{CAMERA_MAP_FILE}")
    else:
        print("\n（默认 dry-run，未写盘。加 --write 落盘）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
