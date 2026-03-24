"""edge-tts 기반 한국어 TTS 음성 출력 모듈."""

import time
import queue
import asyncio
import threading
import tempfile
from pathlib import Path

from config import TTS_VOICE, TTS_RATE, TTS_VOLUME


class TTSSpeaker:
    """edge-tts 신경망 음성을 별도 스레드에서 실행하여 자연스러운 TTS 출력."""

    def __init__(self):
        self.enabled = True
        self.speaking = False
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._last_spoken = ""
        self._temp_dir = tempfile.mkdtemp(prefix="yoloe_tts_")

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
                audio_path = Path(self._temp_dir) / f"tts_{int(time.time()*1000)}.mp3"
                communicate = edge_tts.Communicate(
                    text, voice=TTS_VOICE, rate=TTS_RATE, volume=TTS_VOLUME,
                )
                loop.run_until_complete(communicate.save(str(audio_path)))

                self._pygame.mixer.music.stop()
                self._pygame.mixer.music.unload()
                self._pygame.mixer.music.load(str(audio_path))
                self._pygame.mixer.music.play()

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
        import shutil
        shutil.rmtree(self._temp_dir, ignore_errors=True)
