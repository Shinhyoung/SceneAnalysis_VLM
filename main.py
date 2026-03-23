"""
YOLOE-26 Open-Vocabulary Real-Time Object Detection & Segmentation
+ Qwen2.5-VL-3B On-Device Scene Analysis
with Intel RealSense D455 Camera
"""

import sys
import time
import threading
import queue
from pathlib import Path

import cv2
import numpy as np
import torch
import asyncio
import tempfile
from PIL import Image, ImageDraw, ImageFont

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] pyrealsense2가 설치되어 있지 않습니다.")
    print("  pip install pyrealsense2")
    sys.exit(1)

from ultralytics import YOLOE

# ── 설정 ──────────────────────────────────────────────────────────────
# YOLOE 모델 크기별 설정 (1=nano, 2=small, 3=large)
YOLOE_MODELS = {
    "n": ("yoloe-26n-seg.pt", "yoloe-26n-seg-pf.pt", "Nano"),
    "s": ("yoloe-26s-seg.pt", "yoloe-26s-seg-pf.pt", "Small"),
    "l": ("yoloe-26l-seg.pt", "yoloe-26l-seg-pf.pt", "Large"),
}
current_model_key = "l"  # 기본: Large
CONFIDENCE_THRESHOLD = 0.35
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
MASK_ALPHA = 0.45                       # 세그멘테이션 마스크 투명도

SCREENSHOT_DIR = Path("screenshots")

# ── VLM 설정 ─────────────────────────────────────────────────────────
VLM_MODELS = {
    "7": ("Qwen/Qwen2.5-VL-3B-Instruct",  "Qwen2.5-VL-3B",  "Qwen2_5_VLForConditionalGeneration"),
    "8": ("Qwen/Qwen3-VL-8B-Instruct",     "Qwen3-VL-8B",    "Qwen3VLForConditionalGeneration"),
}
VLM_DEFAULT_KEY = "8"
VLM_AUTO_INTERVAL = 2.0                 # 자동 분석 주기 (초)
VLM_MAX_TOKENS = 200
VLM_PROMPT = "이 장면을 한국어로 간결하게 설명해주세요. 보이는 사물, 사람, 행동, 환경을 포함해 2~3문장으로 요약하세요."

# ── UI 설정 ──────────────────────────────────────────────────────────
INPUT_BAR_HEIGHT = 40
INPUT_BAR_COLOR = (50, 50, 50)
INPUT_TEXT_COLOR = (255, 255, 255)
PROMPT_HINT_COLOR = (150, 150, 150)

VLM_PANEL_COLOR = (30, 30, 30)
VLM_PANEL_ALPHA = 0.75
VLM_TEXT_COLOR = (255, 255, 200)

# ── TTS 설정 (edge-tts 신경망 음성) ──────────────────────────────────
# 한국어 음성 목록: ko-KR-SunHiNeural (여성), ko-KR-InJoonNeural (남성),
#                  ko-KR-BongJinNeural, ko-KR-GookMinNeural,
#                  ko-KR-JiMinNeural, ko-KR-SeoHyeonNeural,
#                  ko-KR-SoonBokNeural, ko-KR-YuJinNeural
TTS_VOICE = "ko-KR-SunHiNeural"        # 자연스러운 한국어 여성 음성
TTS_RATE = "+0%"                        # 속도 조절 (-50% ~ +100%)
TTS_VOLUME = "+0%"                      # 볼륨 조절 (-50% ~ +100%)

# ── 한글 폰트 로드 ──────────────────────────────────────────────────
def load_korean_font(size: int = 20) -> ImageFont.FreeTypeFont | None:
    """시스템에서 한글 폰트를 찾아 로드."""
    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
        "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return None


KOREAN_FONT = load_korean_font(18)
KOREAN_FONT_TITLE = load_korean_font(16)


def put_korean_text(img: np.ndarray, text: str, pos: tuple[int, int],
                    font: ImageFont.FreeTypeFont | None, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """OpenCV 이미지 위에 한글 텍스트를 렌더링 (PIL 사용)."""
    if font is None:
        # 폰트 없으면 영문 폴백
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── TTS 음성 출력 (edge-tts + pygame, 백그라운드 스레드) ──────────────
class TTSSpeaker:
    """edge-tts 신경망 음성을 별도 스레드에서 실행하여 자연스러운 TTS 출력."""

    def __init__(self):
        self.enabled = True
        self.speaking = False
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._last_spoken = ""
        self._temp_dir = tempfile.mkdtemp(prefix="yoloe_tts_")

        # pygame mixer 초기화
        import pygame
        pygame.mixer.init()
        self._pygame = pygame

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[TTS] edge-tts 초기화 완료 (음성: {TTS_VOICE})")

    def _worker(self):
        """백그라운드에서 edge-tts로 음성 생성 후 pygame으로 재생."""
        import edge_tts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.enabled:
                continue

            self.speaking = True
            old_path = None
            try:
                # edge-tts로 mp3 생성 (고유 파일명으로 잠금 충돌 방지)
                audio_path = Path(self._temp_dir) / f"tts_{int(time.time()*1000)}.mp3"
                communicate = edge_tts.Communicate(
                    text,
                    voice=TTS_VOICE,
                    rate=TTS_RATE,
                    volume=TTS_VOLUME,
                )
                loop.run_until_complete(communicate.save(str(audio_path)))

                # 이전 재생 중인 음악 정지 후 로드
                self._pygame.mixer.music.stop()
                self._pygame.mixer.music.unload()
                self._pygame.mixer.music.load(str(audio_path))
                self._pygame.mixer.music.play()

                # 이전 임시 파일 삭제
                if old_path and old_path.exists():
                    try:
                        old_path.unlink()
                    except OSError:
                        pass
                old_path = audio_path

                while self._pygame.mixer.music.get_busy():
                    if self._stop_event.is_set():
                        self._pygame.mixer.music.stop()
                        break
                    time.sleep(0.05)

            except Exception as e:
                print(f"[TTS] 오류: {e}")
            finally:
                self.speaking = False

        loop.close()

    def speak(self, text: str):
        """텍스트를 TTS 큐에 추가. 동일 텍스트 반복 방지."""
        if not self.enabled or not text or text == self._last_spoken:
            return
        # 이전 대기 텍스트 비우고 새 텍스트만 유지
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)
        self._last_spoken = text

    def toggle(self) -> bool:
        """TTS ON/OFF 토글, 현재 상태 반환."""
        self.enabled = not self.enabled
        return self.enabled

    def stop(self):
        """스레드 종료 및 임시 파일 정리."""
        self._stop_event.set()
        self._thread.join(timeout=3)
        self._pygame.mixer.quit()
        # 임시 파일 정리
        import shutil
        shutil.rmtree(self._temp_dir, ignore_errors=True)


# ── VLM 장면 분석기 (백그라운드 스레드) ──────────────────────────────
class SceneAnalyzer:
    """VLM을 백그라운드 스레드에서 실행하여 장면을 분석. 모델 전환 지원."""

    def __init__(self, tts: TTSSpeaker | None = None, model_key: str = VLM_DEFAULT_KEY):
        self.model = None
        self.processor = None
        self.loaded = False
        self.loading = False
        self.load_error = ""
        self.tts = tts
        self.model_key = model_key
        self.model_label = VLM_MODELS[model_key][1]

        # 분석 상태
        self.current_description = ""
        self.analyzing = False
        self.auto_mode = False
        self.last_analysis_time = 0.0

        # 스레드 공유 프레임
        self._frame_lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start_loading(self, model_key: str | None = None):
        """모델을 백그라운드에서 로드. model_key 지정 시 모델 전환."""
        if self.loading:
            return
        if model_key and model_key != self.model_key:
            # 모델 전환: 기존 모델 해제
            self._unload_model()
            self.model_key = model_key
            self.model_label = VLM_MODELS[model_key][1]
        elif self.loaded:
            return
        self.loading = True
        self.load_error = ""
        t = threading.Thread(target=self._load_model, daemon=True)
        t.start()

    def _unload_model(self):
        """현재 모델을 메모리에서 해제."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[VLM] 기존 모델 메모리 해제 완료")

    def _load_model(self):
        """VLM 모델을 INT4 양자화로 로드."""
        try:
            import transformers
            from transformers import AutoProcessor, BitsAndBytesConfig

            model_id, label, class_name = VLM_MODELS[self.model_key]
            model_class = getattr(transformers, class_name)

            print(f"[VLM] {label} ({model_id}) 로딩 중 (INT4 양자화)...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = model_class.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.loaded = True
            self.loading = False
            print(f"[VLM] {label} 로드 완료!")

            # 워커 스레드 시작 (아직 실행 중이 아닌 경우)
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_event.clear()
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()

        except Exception as e:
            self.load_error = str(e)
            self.loading = False
            print(f"[VLM] 모델 로드 실패: {e}")

    def _worker_loop(self):
        """백그라운드에서 분석 요청을 처리하는 워커 루프."""
        while not self._stop_event.is_set():
            frame = None
            with self._frame_lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None

            if frame is not None:
                self._run_inference(frame)
            else:
                time.sleep(0.05)

    def _run_inference(self, frame: np.ndarray):
        """프레임 한 장에 대해 VLM 추론."""
        self.analyzing = True
        try:
            # BGR → RGB, 리사이즈 (VLM 입력 크기 축소로 속도 향상)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb).resize((640, 360))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": VLM_PROMPT},
                    ],
                }
            ]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=VLM_MAX_TOKENS,
                    do_sample=False,
                )

            # 입력 토큰 제거 후 디코딩
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            description = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            self.current_description = description
            self.last_analysis_time = time.time()

            # TTS로 분석 결과 음성 출력
            if self.tts:
                self.tts.speak(description)

        except Exception as e:
            self.current_description = f"[분석 오류] {e}"
        finally:
            self.analyzing = False

    def request_analysis(self, frame: np.ndarray):
        """분석 요청 (워커 스레드에 프레임 전달)."""
        if not self.loaded or self.analyzing:
            return
        with self._frame_lock:
            self._pending_frame = frame.copy()

    def check_auto_analysis(self, frame: np.ndarray):
        """자동 모드일 때 주기적으로 분석 요청."""
        if not self.auto_mode or not self.loaded or self.analyzing:
            return
        if time.time() - self.last_analysis_time >= VLM_AUTO_INTERVAL:
            self.request_analysis(frame)

    def stop(self):
        """워커 스레드 종료."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)

    def get_status_text(self) -> str:
        """현재 VLM 상태 텍스트."""
        name = self.model_label
        if self.load_error:
            return f"VLM: Error ({name})"
        if self.loading:
            return f"VLM: Loading {name}..."
        if not self.loaded:
            return "VLM: [v] to load"
        if self.analyzing:
            return f"VLM: Analyzing ({name})"
        if self.auto_mode:
            return f"VLM: Auto ON ({name})"
        return f"VLM: {name} [7/8]"

    def get_tts_status(self) -> str:
        """TTS 상태 텍스트."""
        if not self.tts:
            return ""
        if self.tts.speaking:
            return "TTS: Speaking..."
        return f"TTS: {'ON' if self.tts.enabled else 'OFF'}"


# ── 색상 팔레트 (클래스별 고유 색상) ──────────────────────────────────
def generate_colors(n: int) -> list[tuple[int, int, int]]:
    """HSV 색상환을 균등 분할하여 고유 BGR 색상 리스트를 생성."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


def setup_camera() -> tuple:
    """카메라를 초기화. RealSense D455 우선, 실패 시 웹캠 폴백.
    Returns:
        (source, align, use_realsense)
        - RealSense: (rs.pipeline, rs.align, True)
        - 웹캠:     (cv2.VideoCapture, None, False)
    """
    # 1) RealSense 시도
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

    # 2) 웹캠 폴백
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


def draw_input_bar(frame: np.ndarray, text: str, active: bool) -> np.ndarray:
    """화면 하단에 텍스트 입력 바를 그림."""
    h, w = frame.shape[:2]
    bar = frame.copy()

    if not active:
        hint = "[t] Search  [ESC] All  [7/8] VLM  [a] Analyze  [d] Auto  [f] TTS"
        cv2.putText(bar, hint, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PROMPT_HINT_COLOR, 1, cv2.LINE_AA)
        return bar

    cv2.rectangle(bar, (0, h - INPUT_BAR_HEIGHT), (w, h), INPUT_BAR_COLOR, -1)
    display = "Search: " + text + "|"
    cv2.putText(bar, display, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, INPUT_TEXT_COLOR, 1, cv2.LINE_AA)

    hint = "Enter=Apply  ESC=Cancel"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(bar, hint, (w - hw - 10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, PROMPT_HINT_COLOR, 1, cv2.LINE_AA)
    return bar


def draw_results(frame: np.ndarray, result, colors: list, depth_frame=None) -> np.ndarray:
    """검출 결과(바운딩 박스 + 세그멘테이션 마스크 + 텍스트)를 프레임에 그림."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    boxes = result.boxes
    masks = result.masks
    names = result.names

    if boxes is None or len(boxes) == 0:
        return frame

    for i, box in enumerate(boxes):
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf < CONFIDENCE_THRESHOLD:
            continue

        color = colors[cls_id % len(colors)]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = names.get(cls_id, f"class_{cls_id}")

        # 세그멘테이션 마스크 오버레이
        if masks is not None and i < len(masks):
            mask_data = masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized > 0.5
            overlay[mask_bool] = (
                np.array(color, dtype=np.float32) * MASK_ALPHA
                + overlay[mask_bool].astype(np.float32) * (1 - MASK_ALPHA)
            ).astype(np.uint8)

        # 바운딩 박스
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # 텍스트 라벨
        label = f"{class_name} {conf:.2f}"

        # 깊이 정보 추가
        if depth_frame is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            depth_m = depth_frame.get_distance(cx, cy)
            if depth_m > 0:
                label += f" {depth_m:.2f}m"

        # 텍스트 배경
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(overlay, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def draw_vlm_panel(frame: np.ndarray, analyzer: SceneAnalyzer) -> np.ndarray:
    """화면 우측에 VLM 장면 분석 결과 패널을 그림."""
    description = analyzer.current_description
    if not description:
        return frame

    h, w = frame.shape[:2]
    margin = 10
    panel_w = min(400, w // 3)            # 화면 폭의 1/3 이하로 제한
    panel_x = w - panel_w - margin
    panel_y = margin
    line_height = 24
    padding = 10

    # 텍스트를 줄 단위로 분리 (패널 폭에 맞춰 자동 줄바꿈)
    lines = []
    max_chars = max((panel_w - padding * 2) // 11, 10)  # 한글 기준 글자 수
    for paragraph in description.split("\n"):
        while len(paragraph) > max_chars:
            lines.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        lines.append(paragraph)

    # 타이틀 추가
    title = "Scene Analysis"
    max_lines = (h - margin * 2 - padding * 2) // line_height - 1  # 화면 높이에 맞춤
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    all_lines = [title] + lines
    panel_h = len(all_lines) * line_height + padding * 2

    # 반투명 패널 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  VLM_PANEL_COLOR, -1)
    frame = cv2.addWeighted(overlay, VLM_PANEL_ALPHA, frame, 1 - VLM_PANEL_ALPHA, 0)

    # 패널 테두리
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (100, 100, 100), 1)

    # 타이틀
    text_x = panel_x + padding
    text_y = panel_y + padding + line_height - 4
    cv2.putText(frame, title, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # 분석 결과 텍스트 (한글 지원)
    for i, line in enumerate(lines):
        y = panel_y + padding + (i + 1) * line_height + line_height - 4
        frame = put_korean_text(frame, line, (text_x, y - 14), KOREAN_FONT, VLM_TEXT_COLOR)

    # 분석 시각 표시
    if analyzer.last_analysis_time > 0:
        elapsed = time.time() - analyzer.last_analysis_time
        time_text = f"{elapsed:.0f}s ago"
        cv2.putText(frame, time_text, (panel_x + panel_w - 70, panel_y + panel_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, PROMPT_HINT_COLOR, 1, cv2.LINE_AA)

    return frame


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
    print("  [7]   VLM: Qwen2.5-VL-3B (가벼움, 기본)")
    print("  [8]   VLM: Qwen3-VL-8B   (고품질)")
    print("  [a]   즉시 장면 분석 (+ TTS 음성 출력)")
    print("  [d]   자동 분석 ON/OFF 토글")
    print("  [f]   TTS 음성 ON/OFF 토글")
    print("  [s]   스크린샷 저장")
    print("  [q]   종료")
    print()

    # YOLOE 모델 로드 (기본: Large, prompt-free 전체 검출)
    global current_model_key
    model = load_model_pf(current_model_key)
    colors = generate_colors(200)
    current_filter = ""

    # TTS 엔진 (edge-tts 신경망 음성)
    tts = TTSSpeaker()

    # VLM 분석기 (TTS 연동)
    analyzer = SceneAnalyzer(tts=tts)

    # 카메라 초기화 (RealSense 우선, 웹캠 폴백)
    print("[INFO] 카메라 연결 중...")
    cam_source, align, use_realsense = setup_camera()

    # FPS 계산용
    prev_time = time.time()
    fps_smooth = 0.0

    # 텍스트 입력 상태
    input_active = False
    input_text = ""

    device = 0 if torch.cuda.is_available() else "cpu"

    SCREENSHOT_DIR.mkdir(exist_ok=True)

    # VLM 모델 자동 로드 (백그라운드)
    analyzer.start_loading()
    print("[VLM] 모델 자동 로딩 시작...")

    try:
        while True:
            # 프레임 획득
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

            # YOLOE 추론
            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                device=device,
            )

            # 결과 시각화
            result = results[0]
            annotated = draw_results(frame, result, colors, depth_frame)

            # VLM 자동 분석 체크
            analyzer.check_auto_analysis(frame)

            # FPS 계산 (지수 이동 평균)
            curr_time = time.time()
            instant_fps = 1.0 / max(curr_time - prev_time, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
            prev_time = curr_time

            # ── HUD 표시 ──
            # FPS
            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # 검출 객체 수
            n_detections = len(result.boxes) if result.boxes is not None else 0
            cv2.putText(annotated, f"Objects: {n_detections}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # 모델 크기 표시
            _, _, model_label = YOLOE_MODELS[current_model_key]
            cv2.putText(annotated, f"YOLOE: {model_label} [1/2/3]", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2, cv2.LINE_AA)

            # 검출 모드
            if current_filter:
                mode_text = f"Filter: {current_filter}"
                mode_color = (0, 255, 255)
            else:
                mode_text = "Mode: Detect ALL"
                mode_color = (0, 200, 0)
            cv2.putText(annotated, mode_text, (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)

            # VLM 상태
            vlm_status = analyzer.get_status_text()
            vlm_color = (0, 200, 255) if analyzer.loaded else (150, 150, 150)
            if analyzer.analyzing:
                vlm_color = (0, 255, 255)
            cv2.putText(annotated, vlm_status, (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, vlm_color, 2, cv2.LINE_AA)

            # TTS 상태
            tts_status = analyzer.get_tts_status()
            if tts_status:
                tts_color = (0, 200, 255) if tts.enabled else (100, 100, 100)
                if tts.speaking:
                    tts_color = (0, 255, 100)
                cv2.putText(annotated, tts_status, (10, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, tts_color, 2, cv2.LINE_AA)

            # VLM 분석 결과 패널
            annotated = draw_vlm_panel(annotated, analyzer)

            # 입력 바
            annotated = draw_input_bar(annotated, input_text, input_active)

            cv2.imshow("YOLOE-26 | RealSense D455", annotated)

            # ── 키 입력 처리 ──
            key = cv2.waitKey(1) & 0xFF

            if input_active:
                # 텍스트 입력 모드
                if key == 27:  # ESC: 입력 취소
                    input_active = False
                    input_text = ""
                elif key == 13:  # Enter: 입력 확정
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
                elif key == 8:  # Backspace
                    input_text = input_text[:-1]
                elif 32 <= key <= 126:  # 일반 문자
                    input_text += chr(key)
            else:
                # 일반 모드
                if key == ord("q"):
                    print("[INFO] 종료합니다.")
                    break
                elif key == ord("t"):
                    input_active = True
                    input_text = ""
                elif key == 27:  # ESC: 전체 객체 검출로 복귀
                    if current_filter:
                        model = load_model_pf(current_model_key)
                        colors = generate_colors(200)
                        current_filter = ""
                        print("[INFO] 전체 객체 검출 모드로 전환")
                elif key in (ord("1"), ord("2"), ord("3")):
                    # 모델 크기 변경: 1=Nano, 2=Small, 3=Large
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
                    # VLM 모델 선택: 7=Qwen2.5-VL-3B, 8=Qwen3-VL-8B
                    new_vlm_key = chr(key)
                    if new_vlm_key != analyzer.model_key or not analyzer.loaded:
                        _, label, _ = VLM_MODELS[new_vlm_key]
                        print(f"[VLM] {label} 로딩 시작...")
                        analyzer.start_loading(model_key=new_vlm_key)
                    else:
                        print(f"[VLM] {analyzer.model_label} 이미 로드되어 있습니다.")
                elif key == ord("a"):
                    # 즉시 장면 분석
                    if analyzer.loaded:
                        analyzer.request_analysis(frame)
                        print("[VLM] 장면 분석 요청")
                    elif not analyzer.loading:
                        print("[VLM] 먼저 [v]로 모델을 로드하세요.")
                elif key == ord("d"):
                    # 자동 분석 토글
                    if analyzer.loaded:
                        analyzer.auto_mode = not analyzer.auto_mode
                        state = "ON" if analyzer.auto_mode else "OFF"
                        print(f"[VLM] 자동 분석: {state}")
                    elif not analyzer.loading:
                        print("[VLM] 먼저 [v]로 모델을 로드하세요.")
                elif key == ord("f"):
                    # TTS ON/OFF 토글
                    state = tts.toggle()
                    print(f"[TTS] 음성 출력: {'ON' if state else 'OFF'}")
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
