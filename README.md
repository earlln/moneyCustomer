# 금전 대고객 작업 분류 (MoneyCustomer)

작업의뢰 텍스트를 읽어 **금전 / 대고객 / 금전+대고객 / 비주요작업** 4개 클래스로
자동 분류하는 배치 처리 프로그램입니다.

| 항목 | 내용 |
|---|---|
| 현재 버전 | **v1.8.0** |
| 모델 | TF-IDF (word 1~2gram, 15,000 features) + LightGBM |
| 형태소 분석 | KoNLPy `Okt` (**Java / JVM 필요**) |
| 학습 데이터 | `train_20260527.csv` (20,000행) |
| 검증 정확도 | **0.9658 (96.58%)** — `evaluate result.txt` 참고 |

## 파일 구성

| 파일 | 설명 |
|---|---|
| `batch_predict.py` | 배치 추론 모듈 (CSV 입력 → 분류 결과 CSV 출력) |
| `train_model.py` | 모델 학습 모듈 |
| `evaluate_model.py` | 모델 성능 평가 도구 |
| `features.json` | 컬럼·임계값·파일명 등 전체 설정 |
| `request_model.pkl` | 학습된 LightGBM 분류 모델 |
| `request_vectorizer.pkl` | 학습된 TF-IDF 벡터라이저 |
| `train_20260527.csv` | 학습 데이터 (CP949 인코딩) |
| `사용설명서.txt` | 상세 사용 설명서 |
| `evaluate result.txt` | v1.8 성능 평가 실행 결과 |
| `Text_Classification_v1.8.pptx` | 프로젝트 설명 자료 |

## 사전 준비

v1.8은 KoNLPy 의 `Okt` 형태소 분석기를 사용하므로 **Java 런타임(JDK/JRE)이 반드시 설치**되어
있어야 하며, 경우에 따라 `JAVA_HOME` 환경 변수 설정이 필요합니다.

```bat
pip install pandas scikit-learn lightgbm joblib konlpy
```

## 사용 방법

```bat
python batch_predict.py                          :: features.json 설정값으로 실행
python batch_predict.py -i mydata.csv -o out.csv :: 입출력 파일 직접 지정
python batch_predict.py --show                   :: 결과 미리보기 출력
python batch_predict.py --encoding cp949         :: 한글 깨질 때 인코딩 지정

python train_model.py                            :: 재학습
python evaluate_model.py                         :: 성능 평가
```

자세한 내용은 [`사용설명서.txt`](사용설명서.txt) 를 참고하세요.

## 분류 기준

| 클래스 | 의미 |
|---|---|
| 0 | 비주요작업 |
| 1 | 금전작업 |
| 2 | 대고객작업 |
| 3 | 금전+대고객 |

클래스 1·2·3 중 하나라도 확률이 `threshold`(기본 0.5) 이상이면 **주요작업**으로 판정합니다.

## 알려진 이슈 (v1.8.0)

- `evaluate_model.py` 의 3개 라인에 오타가 있어 실행 시 `NameError` 가 발생합니다
  (`file_//config`, `csv_//file_name`, `y_//true`). 다음 릴리스에서 수정됩니다.
- Java 가 설치되지 않은 환경에서는 KoNLPy 초기화에 실패하여 실행할 수 없습니다.
