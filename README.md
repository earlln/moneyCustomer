# 금전 대고객 작업 분류 (MoneyCustomer)

작업의뢰 텍스트를 읽어 **금전 / 대고객 / 금전+대고객 / 비주요작업** 4개 클래스로
자동 분류하는 배치 처리 프로그램입니다.

> **v2.0.0 부터 Java(JVM)가 필요 없습니다.**
> 형태소 분석기를 KoNLPy `Okt`(Java 기반) 에서 `kiwipiepy`(C++ 기반) 로 교체했고,
> 파이썬조차 없는 PC 를 위해 Windows `.exe` 로 빌드해 배포합니다.

| 항목 | v1.8.0 | v2.1.0 |
|---|---|---|
| 형태소 분석 | KoNLPy `Okt` | `kiwipiepy` |
| **Java 필요 여부** | **필수** | **불필요** |
| 배포 형태 | 파이썬 소스 | 소스 + **Windows exe** |
| 전처리 | 한글만 보존 | 한글 + 영문 + 숫자 |
| 설정 가능 범위 | 컬럼 · threshold | + 토크나이저 · TF-IDF · 하이퍼파라미터 |
| 검증 정확도(holdout) | 0.8695 (86.95%) | 0.8975 (89.75%) |
| `evaluate_model` | 오타로 실행 불가 | 정상 동작 |

## 정확도에 대한 중요한 정정

v1.8 문서에 적힌 **96.58%** 는 *학습에 사용한 데이터를 그대로 평가에 사용해서* 나온
값이라 실제 성능보다 크게 부풀려져 있습니다. 학습에 쓰이지 않은 검증 구간(20%)에서
다시 측정한 값이 위 표의 "검증 정확도" 입니다.

| 측정 방식 | v1.8 모델 | v2.1.0 모델 |
|---|---|---|
| 전체 20,000행 (학습 데이터 포함 · 과대평가) | 0.9658 | 0.9769 |
| 검증 4,000행 (학습에 미사용 · **실제 성능**) | **0.8695 (86.95%)** | **0.8975 (89.75%)** |

v2.0.0 의 `evaluate_model` 은 두 수치를 모두 출력하되 검증 구간 값을 먼저 보여줍니다.

## 다운로드 및 설치 (stand-alone PC)

[Releases](../../releases) 에서 `MoneyCustomer_v2.1.0_win64.zip` 을 내려받아 압축을 풉니다.
**파이썬 · Java · 인터넷 · 관리자 권한 모두 필요 없습니다.** 파이썬 런타임과 형태소 사전이
`_internal/` 안에 함께 들어 있습니다.

```
MoneyCustomer_v2.1.0_win64/
├─ batch_predict.exe        추론
├─ train_model.exe          재학습
├─ evaluate_model.exe       성능 평가
├─ 자체점검.bat             설치 직후 동작 확인용
├─ _internal/               파이썬 런타임 · 라이브러리 · 형태소 사전 (삭제 금지)
├─ features.json            설정
├─ request_model.pkl        학습된 모델
├─ request_vectorizer.pkl   학습된 벡터라이저
├─ model_meta.json          학습 시점 정보
├─ train_20260527.csv       학습 데이터
├─ sample_input.csv         동작 확인용 예시
└─ 사용설명서.txt
```

### 설치 절차

1. zip 을 USB 등으로 대상 PC 에 옮깁니다. (압축 약 190MB, 해제 후 약 330MB)
2. **`C:\MoneyCustomer` 처럼 짧은 경로**에 압축을 풉니다.
   경로가 너무 길면 Windows 경로 길이 제한(260자)에 걸릴 수 있습니다.
3. `자체점검.bat` 을 더블클릭합니다. 예시 12건이 분류되면 설치 성공입니다.

### 설치 시 주의 사항

| 항목 | 내용 |
|---|---|
| 대상 OS | 64비트 Windows 10 이상 |
| 폴더 구조 | `_internal/` 은 exe 와 같은 위치에 있어야 합니다. exe 만 복사하면 실행되지 않습니다. |
| 설치 위치 | `C:\Program Files` 아래는 피하세요. 결과 CSV·로그를 프로그램 폴더에 쓰기 때문에 쓰기 권한이 필요합니다. |
| 백신 · SmartScreen | 코드 서명이 없어 경고가 뜰 수 있습니다. 사내 배포 시 예외 등록이 필요할 수 있습니다. |
| 파일 경로 기준 | 설정·모델·입출력 파일은 **exe 가 있는 폴더** 기준으로 찾습니다. 바로가기나 다른 폴더에서 실행해도 동작합니다. |

빌드 파이프라인은 매 릴리스마다 파이썬을 `PATH` 에서 제거하고 `JAVA_HOME` 을 비운 상태로
저장소 밖 폴더에 패키지를 복사해 세 exe 를 모두 실행하는 검증을 거칩니다.

## 사용 방법

### exe 로 실행 (Java · 파이썬 불필요)

```bat
batch_predict.exe                            :: features.json 설정값으로 실행
batch_predict.exe -i mydata.csv -o out.csv   :: 입출력 파일 지정
batch_predict.exe --show                     :: 결과 미리보기
batch_predict.exe --encoding cp949           :: 인코딩 강제 지정
batch_predict.exe --no-pause                 :: 스케줄러용 (대기 없이 종료)

train_model.exe                              :: 재학습
evaluate_model.exe                           :: 성능 평가
```

### 파이썬 소스로 실행

```bash
pip install -r requirements.txt   # Java 설치 불필요

python batch_predict.py
python train_model.py
python evaluate_model.py
```

### exe 직접 빌드 (Windows 에서만)

PyInstaller 는 크로스 컴파일을 지원하지 않으므로 Windows `.exe` 는 Windows 에서만
빌드할 수 있습니다.

```bat
pip install -r requirements-build.txt
pyinstaller --noconfirm --clean MoneyCustomer.spec
```

결과물은 `dist/MoneyCustomer/` 에 생성됩니다.
저장소의 **Actions → Release** 워크플로가 `windows-latest` 러너에서 같은 명령으로
빌드하고, Java 가 없는 상태에서 스모크 테스트까지 수행한 뒤 릴리스에 첨부합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `batch_predict.py` | 배치 추론 |
| `train_model.py` | 학습 |
| `evaluate_model.py` | 성능 평가 |
| `korean_tokenizer.py` | **Java 없는 한국어 토크나이저** (`kiwi` / `regex` 백엔드) |
| `tests/test_logic.py` | 판정 로직 단위 테스트 (`python tests/test_logic.py`) |
| `common.py` | 세 모듈이 공유하는 설정 · 인코딩 · CLI 유틸리티 |
| `mc_entry.py` | exe 번들 공용 진입점 (실행 파일 이름으로 명령 분기) |
| `MoneyCustomer.spec` | PyInstaller 빌드 스펙 |
| `features.json` | 컬럼 · threshold · 토크나이저 · 모델 하이퍼파라미터 설정 |
| `train_20260527.csv` | 학습 데이터 20,000행 (CP949) |
| `sample_input.csv` | 동작 확인용 예시 12행 |
| `사용설명서.txt` | 상세 사용 설명서 |

## 분류 기준

| 클래스 | 의미 |
|---|---|
| 0 | 비주요작업 |
| 1 | 금전작업 |
| 2 | 대고객작업 |
| 3 | 금전+대고객 |

클래스 1·2·3 중 하나라도 확률이 `threshold`(기본 0.5) 이상이면 **주요작업**으로 판정합니다.

## 출력 컬럼

| 컬럼 | 설명 |
|---|---|
| `prediction` | 예측 클래스 번호 |
| `label` | 예측된 분류명 |
| `prob_0` ~ `prob_3` | 클래스별 확률 (`class_labels` 에 맞춰 자동 구성) |
| `prob_major` | **주요작업일 확률의 합** (v2.0.0 신규) |
| `is_major` | 주요작업 여부 (`major_rule` 기준) |
| `major_label` | 주요작업 / 비주요작업 |

## 주요 설정 (`features.json`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `threshold` | `0.5` | 주요작업 판정 확률 기준 |
| `major_rule` | `"sum"` | `sum` = 주요 클래스 확률의 **합**이 threshold 이상 (v2.1.0 기본값, 확률적으로 정확)<br>`any` = 개별 확률 중 하나라도 threshold 이상 (v1.8 ~ v2.0.0 동작) |
| `blank_row_policy` | `"trailing"` | `trailing` = 파일 **끝**의 빈 행만 제외<br>`stop_at_first` = 첫 빈 행에서 중단 (v1.8 호환, 데이터 누락 위험) |
| `tokenizer.backend` | `"kiwi"` | `kiwi` / `regex` — 둘 다 Java 불필요 |
| `vectorizer` | 15,000 · 1~2gram | TF-IDF 설정 |
| `model_params` | — | LightGBM 하이퍼파라미터 |

**v2.1.0 에서 `major_rule` 기본값이 `any` → `sum` 으로 바뀌었습니다.** 학습 데이터
20,000행 기준 판정이 달라지는 행은 94건(0.47%)이고, 변경 방향은 비주요작업 → 주요작업
한쪽뿐이라 놓치던 건을 더 잡습니다. 주요작업 판정 정확도는 0.9834 → 0.9850 입니다.
이전 기준이 필요하면 `"any"` 로 되돌리면 되며, 두 값 모두 **재학습이 필요 없습니다.**
`tokenizer` 설정을 바꾼 경우에만 재학습이 필요합니다.

## 토크나이저 설정

`features.json` 의 `tokenizer.backend` 로 선택합니다. **두 백엔드 모두 Java 를 요구하지 않습니다.**

| backend | 구현 | 정확도 | 비고 |
|---|---|---|---|
| `kiwi` (기본) | `kiwipiepy` (C++) | 높음 | 권장 |
| `regex` | 순수 파이썬 규칙 기반 | 낮음 | 외부 의존성이 전혀 없는 대체 수단 |

토크나이저 설정을 바꾸면 **반드시 `train_model` 로 재학습**해야 합니다.
학습 설정은 `model_meta.json` 에 기록되며, 추론 시 설정이 다르면 경고가 출력됩니다.

## 라이선스 / 문의

문제 발생 시 입력 CSV, 출력 CSV, 화면에 출력된 오류 메시지를 함께 첨부해 문의하세요.
