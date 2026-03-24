"""VLM 장면 분석기 모듈 (Qwen2.5-VL / Qwen3-VL)."""

import time
import threading

import cv2
import numpy as np
import torch
from PIL import Image

from config import (
    VLM_MODELS, VLM_DEFAULT_KEY, VLM_AUTO_INTERVAL, VLM_MAX_TOKENS,
    VLM_PROMPT, SAFETY_PROMPT,
)
from tts import TTSSpeaker


def parse_danger_level(text: str) -> str:
    """VLM 응답 앞부분에서 위험 수준 키워드를 추출. 심각한 순서로 탐색."""
    head = text[:100]
    for level in ("긴급", "위험", "주의", "안전"):
        if level in head:
            return level
    return "안전"


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

        self.current_description = ""
        self.analyzing = False
        self.auto_mode = False
        self.last_analysis_time = 0.0

        self.safety_mode = False
        self.danger_level = "안전"

        self._frame_lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start_loading(self, model_key: str | None = None):
        """모델을 백그라운드에서 로드. model_key 지정 시 모델 전환."""
        if self.loading:
            return
        if model_key and model_key != self.model_key:
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
        print("[VLM] 기존 모델 메모리 해제 완료")

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb).resize((640, 360))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": SAFETY_PROMPT if self.safety_mode else VLM_PROMPT},
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

            if self.safety_mode:
                self.danger_level = parse_danger_level(description)
            else:
                self.danger_level = "안전"

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
        tag = " [Safety]" if self.safety_mode else ""
        if self.auto_mode:
            return f"VLM: Auto ON ({name}){tag}"
        return f"VLM: {name} [7/8]{tag}"

    def get_tts_status(self) -> str:
        """TTS 상태 텍스트."""
        if not self.tts:
            return ""
        if self.tts.speaking:
            return "TTS: Speaking..."
        return f"TTS: {'ON' if self.tts.enabled else 'OFF'}"
