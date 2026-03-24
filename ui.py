"""UI 렌더링 모듈 (HUD, 패널, 입력 바, 한글 텍스트)."""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    CONFIDENCE_THRESHOLD, MASK_ALPHA,
    INPUT_BAR_HEIGHT, INPUT_BAR_COLOR, INPUT_TEXT_COLOR, PROMPT_HINT_COLOR,
    VLM_PANEL_COLOR, VLM_PANEL_ALPHA, VLM_TEXT_COLOR,
    DANGER_LEVELS,
)


# ── 한글 폰트 ────────────────────────────────────────────────────────
def load_korean_font(size: int = 20) -> ImageFont.FreeTypeFont | None:
    """시스템에서 한글 폰트를 찾아 로드."""
    for fp in [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return None


KOREAN_FONT = load_korean_font(18)


def put_korean_text(
    img: np.ndarray, text: str, pos: tuple[int, int],
    font: ImageFont.FreeTypeFont | None, color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """OpenCV 이미지 위에 한글 텍스트를 렌더링 (PIL 사용)."""
    if font is None:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── 색상 팔레트 ──────────────────────────────────────────────────────
def generate_colors(n: int) -> list[tuple[int, int, int]]:
    """HSV 색상환을 균등 분할하여 고유 BGR 색상 리스트를 생성."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


# ── 그리기 함수들 ────────────────────────────────────────────────────
def draw_input_bar(frame: np.ndarray, text: str, active: bool) -> np.ndarray:
    """화면 하단에 텍스트 입력 바를 그림."""
    h, w = frame.shape[:2]
    bar = frame.copy()

    if not active:
        hint = "[t] Search [ESC] All [7/8] VLM [a] Analyze [d] Auto [w] Safety [f] TTS"
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

        if masks is not None and i < len(masks):
            mask_data = masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized > 0.5
            overlay[mask_bool] = (
                np.array(color, dtype=np.float32) * MASK_ALPHA
                + overlay[mask_bool].astype(np.float32) * (1 - MASK_ALPHA)
            ).astype(np.uint8)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf:.2f}"
        if depth_frame is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            depth_m = depth_frame.get_distance(cx, cy)
            if depth_m > 0:
                label += f" {depth_m:.2f}m"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(overlay, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def draw_vlm_panel(frame: np.ndarray, analyzer) -> np.ndarray:
    """화면 우측에 VLM 장면 분석 결과 패널을 그림."""
    description = analyzer.current_description
    if not description:
        return frame

    h, w = frame.shape[:2]
    margin = 10
    panel_w = min(400, w // 3)
    panel_x = w - panel_w - margin
    panel_y = margin
    line_height = 24
    padding = 10

    lines = []
    max_chars = max((panel_w - padding * 2) // 11, 10)
    for paragraph in description.split("\n"):
        while len(paragraph) > max_chars:
            lines.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        lines.append(paragraph)

    # 안전 모드 스타일
    is_safety = analyzer.safety_mode
    danger = analyzer.danger_level if is_safety else None

    if is_safety and danger:
        level_info = DANGER_LEVELS.get(danger, DANGER_LEVELS["안전"])
        title = f"Safety Monitor [{level_info['label']}]"
        panel_bg = level_info["color"]
        border_color = level_info["border"]
        if danger == "긴급":
            flash = int(time.time() * 4) % 2 == 0
            panel_bg = (0, 0, 255) if flash else (0, 0, 120)
            border_color = (255, 255, 255) if flash else (0, 0, 255)
        alpha = 0.85
        title_color = border_color
    else:
        title = "Scene Analysis"
        panel_bg = VLM_PANEL_COLOR
        border_color = (100, 100, 100)
        alpha = VLM_PANEL_ALPHA
        title_color = (0, 255, 255)

    max_lines = (h - margin * 2 - padding * 2) // line_height - 1
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    all_lines = [title] + lines
    panel_h = len(all_lines) * line_height + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  panel_bg, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  border_color, 1)

    text_x = panel_x + padding
    text_y = panel_y + padding + line_height - 4
    cv2.putText(frame, title, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_color, 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        y = panel_y + padding + (i + 1) * line_height + line_height - 4
        frame = put_korean_text(frame, line, (text_x, y - 14), KOREAN_FONT, VLM_TEXT_COLOR)

    if analyzer.last_analysis_time > 0:
        elapsed = time.time() - analyzer.last_analysis_time
        time_text = f"{elapsed:.0f}s ago"
        cv2.putText(frame, time_text, (panel_x + panel_w - 70, panel_y + panel_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, PROMPT_HINT_COLOR, 1, cv2.LINE_AA)

    return frame
