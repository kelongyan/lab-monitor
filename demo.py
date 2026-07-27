"""
demo.py — 无需真实视频，生成合成测试视频跑通完整流程
用矩形模拟人员在摄像头间移动，验证检测→跟踪→ReID→预警链路
"""

import cv2
import numpy as np
import time
import logging
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------ #
# 合成视频生成                                                          #
# ------------------------------------------------------------------ #

def make_person_video(
    path: str,
    label: str,
    person_color: tuple[int, int, int],
    frames: int = 200,
    fps: int = 10,
    w: int = 640,
    h: int = 480,
    appear_frame: int = 10,
    leave_frame: int = 170,
) -> None:
    """
    生成一段合成监控视频：背景灰色，一个彩色矩形模拟人员移动
    person_color: BGR 颜色，同一人在不同摄像头用相近颜色（模拟外观一致性）
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(frames):
        # 灰色背景 + 噪点（模拟真实场景）
        frame = np.full((h, w, 3), 80, dtype=np.uint8)
        noise = np.random.randint(-15, 15, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 人员矩形（在出现帧到离开帧之间显示）
        if appear_frame <= i < leave_frame:
            progress = (i - appear_frame) / max(leave_frame - appear_frame - 1, 1)
            px = int(50 + progress * (w - 150))
            py = int(h * 0.3)
            bw, bh = 60, 120  # 宽60，高120 — 模拟人体比例
            cv2.rectangle(frame, (px, py), (px + bw, py + bh), person_color, -1)
            # 头部圆形
            cx, cy = px + bw // 2, py - 20
            cv2.circle(frame, (cx, cy), 18, person_color, -1)

        # 摄像头标签
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, f"frame {i}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        out.write(frame)

    out.release()
    logger.info("合成视频已生成: %s", path)


def generate_demo_videos() -> None:
    """
    生成 3 路测试视频，同一个"人"（相近颜色矩形）先出现在 cam_01，
    然后在 cam_02 出现（测试正常命中），cam_03 延迟出现以触发预警
    """
    # 同一人：蓝色系（B=200, G=100, R=50）— 模拟外观特征相近
    person_color = (200, 100, 50)

    make_person_video(
        "videos/cam01.mp4", "CAM-01 入口",
        person_color=person_color,
        frames=150, fps=10,
        appear_frame=5, leave_frame=120,
    )
    make_person_video(
        "videos/cam02.mp4", "CAM-02 走廊",
        person_color=(190, 110, 60),   # 略微变化，模拟角度差异
        frames=150, fps=10,
        appear_frame=45, leave_frame=130,  # cam_01 离开后约4.5s出现 → 在时间窗口内
    )
    make_person_video(
        "videos/cam03.mp4", "CAM-03 机房",
        person_color=(180, 90, 70),
        frames=150, fps=10,
        appear_frame=140, leave_frame=149,  # 出现太晚 → 触发预警
    )
    logger.info("全部合成视频生成完毕，可运行: python main.py")


# ------------------------------------------------------------------ #
# 快速验证：不依赖 YOLO，直接模拟检测结果跑通 ReID + 预警链路            #
# ------------------------------------------------------------------ #

def run_mock_pipeline() -> None:
    """
    不加载真实模型，用 mock 数据验证身份匹配和预警逻辑
    适合在依赖未安装完成时快速验证系统逻辑
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from src.identity_store import IdentityStore
    from src.topology import CameraTopology
    from src.alerter import AlertManager

    store  = IdentityStore()
    topo   = CameraTopology("config/topology.json")
    alerter = AlertManager("outputs/alerts.jsonl")

    # 模拟：用随机向量代替 ReID 特征（同一人特征相近）
    rng = np.random.default_rng(42)
    base_feat = rng.standard_normal(2048).astype(np.float32)
    base_feat /= np.linalg.norm(base_feat)

    def similar_feat(noise=0.05):
        # 按维度数归一化噪声幅度，保证扰动向量与原向量余弦相似度高
        scale = noise / np.sqrt(2048)
        f = base_feat + rng.standard_normal(2048).astype(np.float32) * scale
        f /= np.linalg.norm(f)
        return f

    from src.reid import match_feature

    print("\n--- Mock Pipeline 验证 ---")

    # 1. cam_01 出现并注册
    gid = store.register(base_feat)
    store.update_appearance(gid, "cam_01", base_feat, [100, 50, 160, 170])
    print(f"[cam_01] 注册新身份: {gid}")

    # 2. cam_01 人员离开，开启预警监听
    hops = topo.next_hops("cam_01")
    max_dl = max(h.deadline_window[1] for h in hops) if hops else 30
    alerter.watch(gid, "cam_01", [h.camera_id for h in hops], max_dl)
    print(f"[cam_01] 人员离开，监听预警窗口 {max_dl}s")

    # 3. cam_02 出现相似特征 → 应命中匹配
    feat2 = similar_feat(noise=0.05)
    gallery = store.get_gallery()
    matched = match_feature(feat2, gallery, threshold=0.75)
    if matched:
        store.update_appearance(matched, "cam_02", feat2, [80, 60, 140, 180])
        resolved = alerter.resolve(matched, "cam_02")
        print(f"[cam_02] 匹配到 {matched}，预警解除: {resolved}")
    else:
        print("[cam_02] ⚠ 未匹配到已知身份（ReID 阈值过高或特征差异大）")

    # 4. 模拟 cam_03 超时（不调用 resolve，直接推进时间触发 tick）
    gid2 = store.register(base_feat.copy())
    alerter.watch(gid2, "cam_01", ["cam_03"], deadline_offset=0.1)  # 0.1s 超时
    time.sleep(0.3)
    alerts = alerter.tick()
    if alerts:
        a = alerts[0]
        print(f"\n⚠ ALERT: {a['alert_type']} | ID={a['global_id']} | "
              f"最后位置={a['last_camera']} | 超时={a['elapsed_seconds']}s | 风险={a['risk_level']}")
    else:
        print("[cam_03] 未触发预警（检查 alerter.tick 逻辑）")

    print("\n--- Mock Pipeline 验证完毕 ---\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="运行 mock 验证（无需安装模型依赖）")
    p.add_argument("--gen", action="store_true", help="生成合成测试视频")
    args = p.parse_args()

    if args.mock:
        run_mock_pipeline()
    elif args.gen:
        generate_demo_videos()
    else:
        # 默认：生成视频 + 运行 mock 验证
        generate_demo_videos()
        run_mock_pipeline()
