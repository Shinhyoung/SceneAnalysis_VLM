"""
YOLOE-26 Open-Vocabulary Real-Time Object Detection & Segmentation
+ VLM On-Device Scene Analysis + Safety Monitoring
with Intel RealSense D455 Camera
"""

import time

import cv2
import numpy as np
import torch

from config import (
    YOLOE_MODELS, DEFAULT_MODEL_KEY, CONFIDENCE_THRESHOLD, SCREENSHOT_DIR,
    VLM_MODELS, DANGER_LEVELS,
)
from camera import setup_camera, load_model_pf, load_model_with_classes
from tts import TTSSpeaker
from vlm import SceneAnalyzer
from ui import generate_colors, draw_results, draw_vlm_panel, draw_input_bar


def main():
    print("=" * 60)
    print("  YOLOE-26 Open-Vocabulary Detection & Segmentation")
    print("  + VLM On-Device Scene Analysis")
    print("  Camera: Intel RealSense D455")
    print("=" * 60)
    print()
    print("  [1]   YOLOE Nano  (빠름, 가벼움)")
    print("  [2]   YOLOE Small (균형)")
    print("  [3]   YOLOE Large (정확, 기본)")
    print("  [t]   텍스트 입력 → 특정 객체만 검출")
    print("  [ESC] 전체 객체 검출 (prompt-free)")
    print("  [7]   VLM: Qwen2.5-VL-3B (가벼움)")
    print("  [8]   VLM: Qwen3-VL-8B   (고품질, 기본)")
    print("  [a]   즉시 장면 분석 (+ TTS 음성 출력)")
    print("  [d]   자동 분석 ON/OFF 토글")
    print("  [f]   TTS 음성 ON/OFF 토글")
    print("  [w]   안전 감시 모드 ON/OFF 토글")
    print("  [s]   스크린샷 저장")
    print("  [q]   종료")
    print()

    # ── 초기화 ──
    current_model_key = DEFAULT_MODEL_KEY
    model = load_model_pf(current_model_key)
    colors = generate_colors(200)
    current_filter = ""

    tts = TTSSpeaker()
    analyzer = SceneAnalyzer(tts=tts)

    print("[INFO] 카메라 연결 중...")
    cam_source, align, use_realsense = setup_camera()

    prev_time = time.time()
    fps_smooth = 0.0
    input_active = False
    input_text = ""
    device = 0 if torch.cuda.is_available() else "cpu"

    SCREENSHOT_DIR.mkdir(exist_ok=True)

    analyzer.start_loading()
    print("[VLM] 모델 자동 로딩 시작...")

    try:
        while True:
            # ── 프레임 획득 ──
            depth_frame = None
            if use_realsense:
                frames = cam_source.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cam_source.read()
                if not ret:
                    continue

            # ── YOLOE 추론 ──
            results = model.predict(
                frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device=device,
            )
            result = results[0]
            annotated = draw_results(frame, result, colors, depth_frame)

            # ── VLM 자동 분석 ──
            analyzer.check_auto_analysis(frame)

            # ── FPS 계산 ──
            curr_time = time.time()
            instant_fps = 1.0 / max(curr_time - prev_time, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
            prev_time = curr_time

            # ── HUD 표시 ──
            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            n_det = len(result.boxes) if result.boxes is not None else 0
            cv2.putText(annotated, f"Objects: {n_det}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            _, _, model_label = YOLOE_MODELS[current_model_key]
            cv2.putText(annotated, f"YOLOE: {model_label} [1/2/3]", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2, cv2.LINE_AA)

            if current_filter:
                cv2.putText(annotated, f"Filter: {current_filter}", (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(annotated, "Mode: Detect ALL", (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)

            vlm_color = (0, 255, 255) if analyzer.analyzing else \
                        (0, 200, 255) if analyzer.loaded else (150, 150, 150)
            cv2.putText(annotated, analyzer.get_status_text(), (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, vlm_color, 2, cv2.LINE_AA)

            tts_status = analyzer.get_tts_status()
            if tts_status:
                tts_color = (0, 255, 100) if tts.speaking else \
                            (0, 200, 255) if tts.enabled else (100, 100, 100)
                cv2.putText(annotated, tts_status, (10, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, tts_color, 2, cv2.LINE_AA)

            if analyzer.safety_mode:
                level_info = DANGER_LEVELS.get(analyzer.danger_level, DANGER_LEVELS["안전"])
                cv2.putText(annotated,
                            f"Safety: {analyzer.danger_level} ({level_info['label']})",
                            (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            level_info["border"], 2, cv2.LINE_AA)

            annotated = draw_vlm_panel(annotated, analyzer)
            annotated = draw_input_bar(annotated, input_text, input_active)

            cv2.imshow("YOLOE-26 | RealSense D455", annotated)

            # ── 키 입력 처리 ──
            key = cv2.waitKey(1) & 0xFF

            if input_active:
                if key == 27:
                    input_active = False
                    input_text = ""
                elif key == 13:
                    input_active = False
                    query = input_text.strip()
                    input_text = ""
                    if query:
                        classes = [c.strip() for c in query.split(",") if c.strip()]
                        model = load_model_with_classes(classes, current_model_key)
                        colors = generate_colors(max(len(classes), 80))
                        current_filter = ", ".join(classes)
                    else:
                        model = load_model_pf(current_model_key)
                        colors = generate_colors(200)
                        current_filter = ""
                elif key == 8:
                    input_text = input_text[:-1]
                elif 32 <= key <= 126:
                    input_text += chr(key)
            else:
                if key == ord("q"):
                    print("[INFO] 종료합니다.")
                    break
                elif key == ord("t"):
                    input_active = True
                    input_text = ""
                elif key == 27:
                    if current_filter:
                        model = load_model_pf(current_model_key)
                        colors = generate_colors(200)
                        current_filter = ""
                        print("[INFO] 전체 객체 검출 모드로 전환")
                elif key in (ord("1"), ord("2"), ord("3")):
                    new_key = {"1": "n", "2": "s", "3": "l"}[chr(key)]
                    if new_key != current_model_key:
                        current_model_key = new_key
                        if current_filter:
                            classes = [c.strip() for c in current_filter.split(",")]
                            model = load_model_with_classes(classes, current_model_key)
                            colors = generate_colors(max(len(classes), 80))
                        else:
                            model = load_model_pf(current_model_key)
                            colors = generate_colors(200)
                        _, _, lbl = YOLOE_MODELS[current_model_key]
                        print(f"[INFO] YOLOE 모델 변경: {lbl}")
                elif key in (ord("7"), ord("8")):
                    new_vlm_key = chr(key)
                    if new_vlm_key != analyzer.model_key or not analyzer.loaded:
                        _, label, _ = VLM_MODELS[new_vlm_key]
                        print(f"[VLM] {label} 로딩 시작...")
                        analyzer.start_loading(model_key=new_vlm_key)
                    else:
                        print(f"[VLM] {analyzer.model_label} 이미 로드되어 있습니다.")
                elif key == ord("a"):
                    if analyzer.loaded:
                        analyzer.request_analysis(frame)
                        print("[VLM] 장면 분석 요청")
                    elif not analyzer.loading:
                        print("[VLM] 먼저 [7/8]로 모델을 로드하세요.")
                elif key == ord("d"):
                    if analyzer.loaded:
                        analyzer.auto_mode = not analyzer.auto_mode
                        print(f"[VLM] 자동 분석: {'ON' if analyzer.auto_mode else 'OFF'}")
                    elif not analyzer.loading:
                        print("[VLM] 먼저 [7/8]로 모델을 로드하세요.")
                elif key == ord("f"):
                    state = tts.toggle()
                    print(f"[TTS] 음성 출력: {'ON' if state else 'OFF'}")
                elif key == ord("w"):
                    analyzer.safety_mode = not analyzer.safety_mode
                    analyzer.danger_level = "안전"
                    analyzer.current_description = ""
                    mode_name = "안전감시" if analyzer.safety_mode else "일반분석"
                    print(f"[VLM] 분석 모드: {mode_name}")
                    if analyzer.auto_mode and analyzer.loaded:
                        analyzer.request_analysis(frame)
                elif key == ord("s"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = SCREENSHOT_DIR / f"screenshot_{ts}.png"
                    cv2.imwrite(str(path), annotated)
                    print(f"[INFO] 스크린샷 저장: {path}")

    finally:
        analyzer.stop()
        tts.stop()
        if use_realsense:
            cam_source.stop()
        else:
            cam_source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
