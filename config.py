"""전역 설정 상수."""

from pathlib import Path

# ── YOLOE 설정 ────────────────────────────────────────────────────────
YOLOE_MODELS = {
    "n": ("yoloe-26n-seg.pt", "yoloe-26n-seg-pf.pt", "Nano"),
    "s": ("yoloe-26s-seg.pt", "yoloe-26s-seg-pf.pt", "Small"),
    "l": ("yoloe-26l-seg.pt", "yoloe-26l-seg-pf.pt", "Large"),
}
DEFAULT_MODEL_KEY = "l"
CONFIDENCE_THRESHOLD = 0.35
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
MASK_ALPHA = 0.45

SCREENSHOT_DIR = Path("screenshots")

# ── VLM 설정 ──────────────────────────────────────────────────────────
VLM_MODELS = {
    "7": ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B", "Qwen2_5_VLForConditionalGeneration"),
    "8": ("Qwen/Qwen3-VL-8B-Instruct",    "Qwen3-VL-8B",   "Qwen3VLForConditionalGeneration"),
}
VLM_DEFAULT_KEY = "8"
VLM_AUTO_INTERVAL = 2.0
VLM_MAX_TOKENS = 200
VLM_PROMPT = (
    "이 장면을 한국어로 간결하게 설명해주세요. "
    "보이는 사물, 사람, 행동, 환경을 포함해 2~3문장으로 요약하세요."
)

# ── 안전 분석 설정 ────────────────────────────────────────────────────
SAFETY_PROMPT = (
    "이 장면에서 안전 위험 요소를 분석하세요. "
    "화재, 연기, 쓰러진 사람, 위험한 물체, 비정상적 행동, 안전장비 미착용 등을 확인하고 "
    "반드시 첫 줄에 위험 수준을 [안전], [주의], [위험], [긴급] 중 하나로 표시한 뒤 "
    "구체적인 이유를 한국어 2~3문장으로 설명하세요."
)
DANGER_LEVELS = {
    "안전": {"color": (0, 140, 0),   "label": "SAFE",      "border": (0, 200, 0)},
    "주의": {"color": (0, 170, 210), "label": "CAUTION",   "border": (0, 220, 255)},
    "위험": {"color": (0, 0, 180),   "label": "DANGER",    "border": (0, 0, 255)},
    "긴급": {"color": (0, 0, 255),   "label": "EMERGENCY", "border": (0, 0, 255)},
}

# ── UI 설정 ───────────────────────────────────────────────────────────
INPUT_BAR_HEIGHT = 40
INPUT_BAR_COLOR = (50, 50, 50)
INPUT_TEXT_COLOR = (255, 255, 255)
PROMPT_HINT_COLOR = (150, 150, 150)

VLM_PANEL_COLOR = (30, 30, 30)
VLM_PANEL_ALPHA = 0.75
VLM_TEXT_COLOR = (255, 255, 200)

# ── TTS 설정 ──────────────────────────────────────────────────────────
TTS_VOICE = "ko-KR-SunHiNeural"
TTS_RATE = "+0%"
TTS_VOLUME = "+0%"
