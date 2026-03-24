"""카메라 초기화 및 YOLOE 모델 로드 모듈."""

import sys

import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] pyrealsense2가 설치되어 있지 않습니다.")
    print("  pip install pyrealsense2")
    sys.exit(1)

from ultralytics import YOLOE

from config import YOLOE_MODELS, FRAME_WIDTH, FRAME_HEIGHT, FPS


def setup_camera() -> tuple:
    """카메라를 초기화. RealSense D455 우선, 실패 시 웹캠 폴백.

    Returns:
        (source, align, use_realsense)
    """
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("[INFO] RealSense D455 연결 완료")
        return pipeline, align, True
    except Exception as e:
        print(f"[WARN] RealSense 연결 실패: {e}")
        print("[INFO] 웹캠으로 폴백합니다...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠도 사용할 수 없습니다.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    print("[INFO] 웹캠 연결 완료")
    return cap, None, False


def load_model_pf(model_key: str = "l") -> YOLOE:
    """Prompt-free 모델을 로드 (전체 객체 검출)."""
    _, pf_name, label = YOLOE_MODELS[model_key]
    print(f"[INFO] Prompt-free 모델 로드: {pf_name} ({label})")
    return YOLOE(pf_name)


def load_model_with_classes(classes: list[str], model_key: str = "l") -> YOLOE:
    """지정된 클래스로 open-vocabulary 모델을 로드."""
    ov_name, _, label = YOLOE_MODELS[model_key]
    model = YOLOE(ov_name)
    model.set_classes(classes, model.get_text_pe(classes))
    print(f"[INFO] Open-vocabulary 클래스 설정: {classes} ({label})")
    return model
