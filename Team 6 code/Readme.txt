# Team 6 — 실시간 원전 사고 진단 시스템 (대회 제출용)

- **Model**: TCN V3 (100ep, physics_v3 features)  
- **Logic**: CL Guard + ESDE Guard (LC 없음)

---

## 0) 로컬(Conda 없이) 실행 방법 ⭐️(이거만 보면 됨)

### CMD(명령 프롬프트)에서 실행
터미널 2개를 열고 아래 순서대로 실행하세요.

**[터미널 1] UDP 수신기**
```bat
cd "C:\Users\user\Desktop\Team 6 code\py"
set PYTHONIOENCODING=utf-8 && py -3.11 UDP_read.py
```

**[터미널 2] 추론 파이프라인**
```bat
cd "C:\Users\user\Desktop\Team 6 code\py"
set PYTHONIOENCODING=utf-8 && py -3.11 main.py
```

> ⚠️ `main.py`는 입력 파일이 들어오기 전까지(UDP 송신 전까지) 대기하는 게 정상입니다.

---

## 1) 환경 준비 (Windows)

- **Python 3.11 필수** (예: 3.11.9 검증 완료)
- Python 3.12+ 또는 3.14에서는 환경 충돌/설치 불가 가능성이 높습니다.

### 버전 확인
```bat
py -3.11 --version
```

---

## 2) 필수 라이브러리 설치 (버전 고정 권장)

> ⚠️ TensorFlow 2.20은 **numpy < 2.0** 조건이므로 `numpy==1.26.4` 유지 권장

### 한 줄 설치 (py launcher 사용)
```bat
py -3.11 -m pip install tensorflow==2.20.0 numpy==1.26.4 scikit-learn==1.8.0 joblib==1.5.3 keras==3.13.2 scipy==1.17.0 h5py==3.15.1 pandas
```

---

## 3) 폴더 구조

```
Team 6 code/
├── py/
│   ├── main.py              # 메인 추론 파이프라인 (파일 감지 → 추론 → UDP 전송)
│   ├── UDP_read.py          # UDP 수신 → data/test{n}/test{n}_sec{t}.csv 저장
│   └── practice/            # 피처 변환 관련 모듈
├── models/                  # 모델 파일 5종 세트
├── ref/bootcamp_list.dat    # 컬럼 타입 정보
├── data/                    # 실시간 데이터 저장 폴더 (자동 생성)
└── README.md
```

> ⚠️ 두 스크립트는 반드시 `py/` 폴더에서 실행해야 합니다.  
> (상대경로로 `../models`, `../data`, `../ref`를 참조합니다.)

---

## 4) 데이터 흐름

서버(송신)  
→ **UDP**  
→ `UDP_read.py` (수신)  
→ `data/test{n}/test{n}_sec{t}.csv` (1초 단위 저장)  
→ `main.py` (파일 감지)  
→ 전처리 → 모델 추론 → 진단 확정  
→ **UDP 전송** → 채점 서버

---

## 5) 네트워크 설정

### UDP_read.py (수신)
- `RECV_IP = 192.168.0.5`
- `RECV_PORT = 7001`

### main.py (송신)
- `TARGETS = [("192.168.0.3", 7001)]`

> 대회 환경에 맞게 IP/PORT 수정:
- `UDP_read.py`의 `RECV_IP`, `RECV_PORT`
- `main.py`의 `TARGETS`

---

## 6) 테스트 범위 설정

`main.py`에서 조절:
- `TEST_START, TEST_END = 1, 9`  (test ID 범위)
- `SEC_START, SEC_END = 1, 60`   (초 범위)

---

## 7) 진단 확정 로직

- **SGTR**: 2초 연속 동일 예측 시 확정  
- **ESDE**: 2초 연속 동일 예측 + ESDE Guard  
  - 최근 3초 중 LOCA 확률합이 `> 0.05`이면 확정 보류
- **LOCA 계열**: 4초 연속 동일 예측 시 확정  
  - **LOCA_HL / LOCA_RCP**: CL Guard 추가 적용  
    - 최근 5초 중 CL 확률 max가 `> 0.15`이면 확정 보류  
  - **LOCA_CL**: 4연속이면 즉시 확정
- **Grace Period**: 처음 3초는 확정하지 않음  
- **한 번 확정되면 변경 없음** (첫 확정이 최종)

---

## 8) 전송 데이터 형식

```
test{n} sec{t}|diagnostic_results,diagnostic_time,prob_1,prob_2,...,prob_9
```

- `diagnostic_results`: 레이블(str) 또는 `"None"`
- `diagnostic_time`: `round(sec + runtime, 1)`
- `prob_1..9`: 9개 클래스 확률(소수 6자리)

### 클래스 목록 (9개)
- 0: NORMAL  
- 1: LOCA_HL  
- 2: LOCA_CL  
- 3: LOCA_RCP  
- 4: SGTR_Loop1  
- 5: SGTR_Loop2  
- 6: SGTR_Loop3  
- 7: ESDE_in  
- 8: ESDE_out  

---

## 9) 모델 정보

`models/` 내 파일(세트):
- `__model.keras` : 모델 가중치
- `__scaler.pkl` : StandardScaler
- `__feature_transformer.pkl` : 피처 변환기 (raw → engineered)
- `__config.json` : 학습 설정
- `__class_mapping.npy` : 클래스 매핑

모델 I/O:
- 입력: `(batch, 3, 266)`  (3초 윈도우, 266 features)
- 출력: `(batch, 9)`       (softmax 확률)

---

## 10) 트러블슈팅

### TensorFlow 설치 오류: “No matching distribution found”
- Python 버전 문제일 가능성 큼  
→ `py --list` 확인 후 **Python 3.11**로 실행

### import tensorflow 시 DLL 오류
- Microsoft Visual C++ Redistributable 최신 버전 설치 필요

### 모델 파일 not found
- `py/` 폴더에서 실행했는지 확인  
  (`../models` 상대경로 사용)

### UDP 수신/송신이 안 됨
- 방화벽에서 UDP 포트(예: 7001) 허용
- IP/PORT가 대회 환경과 일치하는지 확인
