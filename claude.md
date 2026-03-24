## 프로젝트 개요

RealSense D455 카메라(또는 웹캠)로 입력된 실시간 영상에서 YOLOE-26 모델을 사용하여 한정되지 않은 어휘(open-vocabulary)로 객체를 검출하고, 세그멘테이션 마스크를 오버레이하는 프로그램이다. VLM(Vision-Language Model)으로 장면을 한국어로 분석하고, 안전 감시 모드를 통해 위험 상황을 실시간 감지한다. 분석 결과는 TTS로 음성 출력된다.

## 개발 환경

| 항목 | 사양 |
|------|------|
| **OS** | Windows 11 |
| **GPU** | NVIDIA RTX 5070 (12GB VRAM) |
| **카메라** | Intel RealSense D455 (또는 웹캠) |
| **Python** | 3.10 이상 |
| **CUDA** | 12.x (RTX 5070 Blackwell 아키텍처 호환) |

## 프로젝트 구조

```
SceneAnalysis_VLM/
├── main.py              # 엔트리포인트 (메인 루프 + 키 입력 처리)
├── config.py            # 전역 설정 상수 (모델, UI, TTS, 안전 분석)
├── camera.py            # 카메라 초기화 (RealSense/웹캠 폴백) + YOLOE 모델 로드
├── vlm.py               # VLM 장면 분석기 (SceneAnalyzer 클래스 + 위험 수준 파서)
├── tts.py               # TTS 음성 출력 (TTSSpeaker 클래스, edge-tts 기반)
├── ui.py                # UI 렌더링 (HUD, 분석 패널, 입력 바, 한글 폰트)
├── requirements.txt     # Python 패키지 목록
├── CLAUDE.md            # 이 파일 (프로젝트 명세)
├── README.md            # 프로젝트 설명서
└── screenshots/         # 스크린샷 저장 폴더 (자동 생성)
```

## 핵심 기능

### 1. YOLOE-26 객체 검출 + 세그멘테이션
- **Open-Vocabulary**: 텍스트 프롬프트(`t` 키)로 임의의 영어 객체명을 지정하여 검출
- **Prompt-Free**: 1200+ 카테고리 자동 검출 (기본 모드)
- **모델 크기 전환**: `1`/`2`/`3` 키로 Nano(빠름) / Small(균형) / Large(정확) 실시간 전환
- **뎁스 거리**: RealSense 깊이 센서로 객체까지 거리(m) 표시

### 2. VLM 온디바이스 장면 분석
- **Qwen2.5-VL-3B** (`7` 키): 가벼움, VRAM ~2-3GB
- **Qwen3-VL-8B** (`8` 키, 기본): 고품질, VRAM ~5-6GB
- INT4 양자화(BitsAndBytes)로 GPU 메모리 효율화
- 백그라운드 스레드에서 추론, 메인 루프 블로킹 없음
- `a` 키로 즉시 분석, `d` 키로 자동 분석(2초 주기)

### 3. 안전 감시 모드
- `w` 키로 토글
- 위험 요소 분석 전용 프롬프트로 VLM 추론
- 위험 수준 4단계: 안전(초록) / 주의(노랑) / 위험(빨강) / 긴급(빨강 깜빡임)
- 자동 분석 모드와 연동하여 지속적 안전 감시 가능

### 4. TTS 음성 출력
- edge-tts 기반 한국어 신경망 음성 (SunHiNeural)
- 장면 분석 / 안전 감시 결과를 자연스러운 음성으로 출력
- `f` 키로 ON/OFF 토글

### 5. 카메라
- RealSense D455 우선 연결, 실패 시 웹캠 자동 폴백
- 1280x720 @ 30fps

## 단축키

| 키 | 기능 |
|----|------|
| `1`/`2`/`3` | YOLOE 모델 전환 (Nano/Small/Large) |
| `t` | 텍스트 입력 → 특정 객체만 검출 (영어) |
| `ESC` | 전체 객체 검출 모드 (prompt-free) 복귀 |
| `7`/`8` | VLM 모델 전환 (Qwen2.5-VL-3B / Qwen3-VL-8B) |
| `a` | 즉시 장면 분석 + TTS |
| `d` | 자동 분석 ON/OFF |
| `w` | 안전 감시 모드 ON/OFF |
| `f` | TTS ON/OFF |
| `s` | 스크린샷 저장 |
| `q` | 종료 |

## 사용 모델

| 모델 | 용도 | 크기 |
|------|------|------|
| YOLOE-26 (n/s/l)-seg | 객체 검출 + 세그멘테이션 | 15~80MB |
| YOLOE-26 (n/s/l)-seg-pf | Prompt-free 전체 검출 | 15~80MB |
| Qwen2.5-VL-3B-Instruct | 장면 분석 (INT4) | ~2GB |
| Qwen3-VL-8B-Instruct | 장면 분석 (INT4, 기본) | ~6GB |
| MobileCLIP | Open-vocab 텍스트 인코딩 | ~242MB |

## 실행 방법

```bash
# 1. 가상환경 생성
python -m venv venv
venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. PyTorch CUDA 버전 설치 (GPU 사용 시)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. 실행
python main.py
```

> 첫 실행 시 YOLOE 모델, VLM 모델, MobileCLIP이 자동 다운로드됩니다.

## 확장 가능 기능

1. **객체 추적**: ByteTrack 연동하여 동일 객체에 고유 ID 부여
2. **녹화 기능**: `r` 키로 결과 영상을 MP4로 녹화
3. **웹 스트리밍**: Flask/FastAPI로 결과 영상을 웹 브라우저에 스트리밍
4. **커스텀 모델**: 특정 도메인 객체 검출을 위한 fine-tuned 모델 적용
5. **다국어 TTS**: CosyVoice2 등 온디바이스 TTS 엔진으로 교체
