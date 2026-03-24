# SceneAnalysis_VLM

YOLOE-26 기반 **Open-Vocabulary 실시간 객체 검출 + 세그멘테이션** 및 **VLM 온디바이스 장면 분석** 프로그램

RealSense D455 카메라 (또는 웹캠) 영상에서 한정되지 않은 어휘(open-vocabulary)로 객체를 실시간 검출하고, 세그멘테이션 마스크를 오버레이합니다. VLM(Vision-Language Model)으로 장면을 한국어로 분석하고 TTS로 음성 출력합니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| **Open-Vocabulary 검출** | 텍스트 프롬프트로 임의의 객체를 검출 (YOLOE-26) |
| **Prompt-Free 검출** | 1200+ 카테고리 자동 검출 (기본 모드) |
| **세그멘테이션 마스크** | 검출 객체에 반투명 컬러 마스크 오버레이 |
| **뎁스 거리 표시** | RealSense 깊이 센서로 객체까지 거리(m) 표시 |
| **VLM 장면 분석** | Qwen2.5-VL-3B / Qwen3-VL-8B 온디바이스 모델로 장면을 한국어로 설명 |
| **TTS 음성 출력** | edge-tts 신경망 음성(SunHiNeural)으로 분석 결과를 자연스럽게 읽어줌 |
| **YOLOE 모델 전환** | Nano / Small / Large 모델을 단축키(`1`/`2`/`3`)로 실시간 전환 |
| **VLM 모델 전환** | Qwen2.5-VL-3B / Qwen3-VL-8B를 단축키(`7`/`8`)로 전환 |
| **안전 감시 모드** | VLM 기반 위험 상황 실시간 감지 (안전/주의/위험/긴급 4단계) |
| **웹캠 폴백** | RealSense 미연결 시 자동으로 웹캠 사용 |

## 개발 환경

| 항목 | 사양 |
|------|------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 5070 (12GB VRAM) |
| 카메라 | Intel RealSense D455 (또는 웹캠) |
| Python | 3.10 이상 |
| CUDA | 12.x |

## 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/Shinhyoung/SceneAnalysis_VLM.git
cd SceneAnalysis_VLM

# 2. 가상환경 생성
python -m venv venv
venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. PyTorch CUDA 버전 설치 (GPU 사용 시)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 5. 실행
python main.py
```

> 첫 실행 시 YOLOE 모델, VLM 모델, MobileCLIP 등이 자동 다운로드됩니다.

## 단축키

### 객체 검출

| 키 | 기능 |
|----|------|
| `t` | 텍스트 입력 모드 — 검출할 객체 이름 입력 (영어, 쉼표로 복수 지정) |
| `ESC` | 전체 객체 검출 모드 (prompt-free)로 복귀 |

### YOLOE 모델 전환

| 키 | 모델 | 특성 |
|----|------|------|
| `1` | YOLOE-26 **Nano** | 빠른 속도, 가벼움 (~15MB) |
| `2` | YOLOE-26 **Small** | 속도/정확도 균형 (~33MB) |
| `3` | YOLOE-26 **Large** | 높은 정확도 (기본, ~80MB) |

### VLM 장면 분석

| 키 | 기능 |
|----|------|
| `7` | VLM: **Qwen2.5-VL-3B** 로드 (가벼움, VRAM ~2-3GB) |
| `8` | VLM: **Qwen3-VL-8B** 로드 (고품질, VRAM ~5-6GB, 기본) |
| `a` | 즉시 장면 분석 + TTS 음성 출력 |
| `d` | 자동 분석 ON/OFF 토글 (2초 주기) |
| `w` | 안전 감시 모드 ON/OFF 토글 |
| `f` | TTS 음성 ON/OFF 토글 |

### 안전 감시 모드

`w` 키로 활성화하면 VLM이 장면의 위험 요소를 분석합니다.

| 위험 수준 | 패널 색상 | 설명 |
|-----------|-----------|------|
| **안전** (SAFE) | 초록 | 위험 요소 없음 |
| **주의** (CAUTION) | 노랑 | 잠재적 위험 요소 감지 |
| **위험** (DANGER) | 빨강 | 즉각적인 위험 감지 |
| **긴급** (EMERGENCY) | 빨강 깜빡임 | 긴급 대응 필요 |

분석 대상: 화재, 연기, 쓰러진 사람, 위험한 물체, 비정상적 행동, 안전장비 미착용 등

### 기타

| 키 | 기능 |
|----|------|
| `s` | 스크린샷 저장 (`screenshots/` 폴더) |
| `q` | 종료 |

## 화면 구성

```
+-----------------------------------------------------------+
| FPS: 30.0                         [Scene Analysis]        |
| Objects: 5                        | 책상 앞에 사람이      |
| YOLOE: Large [1/2/3]             | 노트북으로 작업 중.   |
| Mode: Detect ALL                  | 왼쪽에 커피잔이       |
| VLM: Qwen3-VL-8B [7/8]          | 놓여 있다.            |
| TTS: ON                           +----------------------+
| Safety: 안전 (SAFE)                                      |
|                                                           |
|   [ person 0.92 1.2m ]        [ laptop 0.88 0.8m ]      |
|   [     세그멘테이션 마스크 오버레이     ]                 |
|                                                           |
| [t] Search [ESC] All [7/8] VLM [a] Analyze [w] Safety   |
+-----------------------------------------------------------+
```

## 사용되는 모델

| 모델 | 용도 | 크기 (INT4) |
|------|------|-------------|
| YOLOE-26n-seg | 객체 검출 + 세그멘테이션 (Nano) | ~15MB |
| YOLOE-26s-seg | 객체 검출 + 세그멘테이션 (Small) | ~33MB |
| YOLOE-26l-seg | 객체 검출 + 세그멘테이션 (Large) | ~80MB |
| Qwen2.5-VL-3B-Instruct | 장면 분석 (가벼움) | ~2GB |
| Qwen3-VL-8B-Instruct | 장면 분석 (고품질, 기본) | ~6GB |
| MobileCLIP | 텍스트 인코딩 (open-vocab) | ~242MB |

## 아키텍처

```
[RealSense D455 / 웹캠]
        │
        ▼
   ┌─────────────────────────┐
   │   YOLOE-26 (n/s/l)      │ ← 실시간 객체 검출 + 세그멘테이션
   │   + MobileCLIP           │ ← Open-vocabulary 텍스트 인코딩
   └─────────┬───────────────┘
             │
             ▼
   ┌─────────────────────────┐
   │   OpenCV 시각화          │ ← 바운딩박스 + 마스크 + 뎁스 + HUD
   └─────────┬───────────────┘
             │
   ┌─────────┴───────────────┐
   │   VLM (백그라운드 스레드) │ ← Qwen2.5-VL-3B / Qwen3-VL-8B
   │   → 장면 분석 / 안전 감시│
   └─────────┬───────────────┘
             │
             ▼
   ┌─────────────────────────┐
   │   edge-tts (백그라운드)  │ ← 한국어 신경망 TTS 음성 출력
   └─────────────────────────┘
```

## 프로젝트 구조

```
SceneAnalysis_VLM/
├── main.py              # 엔트리포인트 (메인 루프 + 키 입력 처리)
├── config.py            # 전역 설정 상수 (모델, UI, TTS, 안전 분석)
├── camera.py            # 카메라 초기화 + YOLOE 모델 로드
├── vlm.py               # VLM 장면 분석기 (SceneAnalyzer 클래스)
├── tts.py               # TTS 음성 출력 (TTSSpeaker 클래스)
├── ui.py                # UI 렌더링 (HUD, 패널, 입력 바, 한글 폰트)
├── requirements.txt     # Python 패키지 목록
├── CLAUDE.md            # 프로젝트 명세
├── README.md            # 이 파일
├── .gitignore           # Git 제외 파일 목록
└── screenshots/         # 스크린샷 저장 폴더 (자동 생성)
```

## 라이선스

이 프로젝트는 학습 및 연구 목적으로 작성되었습니다.
