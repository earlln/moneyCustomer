"""금전 대고객 작업 분류 - 모델 성능 평가 도구 (v2.0.0)

정답 컬럼이 있는 CSV 를 읽어 정확도·분류 리포트·혼동 행렬을 산출합니다.
v1.8 과 달리 Java(JVM)가 필요 없습니다.

v1.8 의 `evaluate_model.py` 는 학습에 사용한 데이터를 그대로 평가에 써서
실제보다 높은 정확도를 보고했습니다(문서상 96.58%). v2.0.0 은 학습에 쓰지
않은 검증 구간(holdout)을 분리해 함께 보고하므로, 이 값을 모델의 실제
일반화 성능으로 보아야 합니다.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import common
from korean_tokenizer import from_config

REPORT_FILENAME = "evaluation_report.txt"


def evaluate(y_true, y_pred, title, log):
    acc = accuracy_score(y_true, y_pred)
    log("")
    log("=" * 62)
    log(f" {title}")
    log("=" * 62)
    log(f" 정확도(Accuracy): {acc:.4f} ({acc * 100:.2f}%)   [{len(y_true):,}행]")
    log("")
    log("[분류 리포트]")
    log(classification_report(y_true, y_pred, zero_division=0))
    log("[혼동 행렬] (행: 실제값, 열: 예측값)")
    log(str(confusion_matrix(y_true, y_pred)))
    return acc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="금전 대고객 작업 분류 - 모델 성능 평가 v2.0.0 (Java 불필요)"
    )
    parser.add_argument("-d", "--data", default=None, help="평가 CSV 파일 (기본: features.json 의 train_data)")
    parser.add_argument("--encoding", default=None, help="CSV 인코딩 (기본: 자동 판별)")
    parser.add_argument(
        "--holdout-only",
        action="store_true",
        help="학습에 쓰지 않은 검증 구간만 평가 (전체 데이터 평가 생략)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="검증 구간 비율 (학습 때와 동일해야 함, 기본 0.2)")
    parser.add_argument("--no-pause", action="store_true", help="종료 시 Enter 대기 생략")
    args = parser.parse_args(argv)

    logs = []

    def log(message=""):
        text = str(message)
        print(text)
        logs.append(text)

    log("=" * 62)
    log(f" 금전 대고객 작업 분류 - 모델 성능 평가 v{common.APP_VERSION}")
    log("=" * 62)
    log(f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")

    config = common.load_config()
    files = config.get("files", {}) or {}

    vectorizer_path = common.resource_path(files.get("vectorizer", "request_vectorizer.pkl"))
    model_path = common.resource_path(files.get("model", "request_model.pkl"))
    for path, what in ((vectorizer_path, "벡터라이저"), (model_path, "모델")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} 파일을 찾을 수 없습니다: {path}")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    tokenizer = from_config(config)
    log(f"토크나이저 백엔드: {tokenizer.backend} (JVM 불필요)")

    meta = common.load_meta()
    common.check_tokenizer_match(meta, tokenizer, log=log)
    if meta:
        log(f"모델 학습 시각: {meta.get('trained_at', '알 수 없음')}")
        trained_metrics = (meta.get("metrics") or {})
        if trained_metrics:
            log(f"학습 시 기록된 검증 정확도: {trained_metrics.get('holdout_accuracy')}")

    csv_path = common.resource_path(args.data or files.get("train_data", "train_data.csv"))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"평가 데이터를 찾을 수 없습니다: {csv_path}\n"
            "  평가할 CSV 를 이 폴더에 두거나 -d 옵션으로 경로를 지정하세요."
        )

    log(f"평가 데이터: {csv_path}")
    df, used_encoding = common.read_csv_auto(csv_path, args.encoding)
    log(f"{len(df):,}행 로드 (인코딩: {used_encoding})")

    input_columns = config.get("input_columns", [])
    target_column = config.get("target_column", "금전대고객구분")
    if target_column not in df.columns:
        raise KeyError(
            f"정답 컬럼 '{target_column}' 이(가) CSV 에 없습니다.\n"
            f"  CSV 의 실제 컬럼: {list(df.columns)}"
        )

    df = df.dropna(subset=[target_column])
    log("\n전처리 및 예측 수행 중...")
    texts = tokenizer.transform_many(common.combine_columns(df, input_columns, log=log))
    y_true = df[target_column].to_numpy()
    y_pred = model.predict(vectorizer.transform(texts))

    is_training_data = bool(meta) and meta.get("train_data") == os.path.basename(csv_path)

    if is_training_data:
        # train_model.py 와 동일한 분할을 재현해 학습에 쓰이지 않은 구간을 골라낸다.
        indices = np.arange(len(df))
        _, holdout_idx = train_test_split(
            indices, test_size=args.test_size, random_state=42, stratify=y_true
        )
        evaluate(
            y_true[holdout_idx],
            y_pred[holdout_idx],
            "검증 구간(holdout) 평가 - 모델의 실제 일반화 성능",
            log,
        )
        if not args.holdout_only:
            evaluate(
                y_true, y_pred, "전체 데이터 평가 - 참고용(학습 데이터 포함, 과대평가됨)", log
            )
            log("")
            log("[해석] 위 '전체 데이터' 수치는 모델이 학습에 사용한 행을 포함하므로")
            log("       실제 성능보다 높게 나옵니다. 반드시 '검증 구간' 정확도를 기준으로")
            log("       모델 성능을 판단하세요.")
    else:
        evaluate(y_true, y_pred, "평가 결과 (학습에 사용되지 않은 외부 데이터)", log)

    report_path = common.resource_path(REPORT_FILENAME)
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs) + "\n")
        print(f"\n[완료] 평가 결과를 저장했습니다: {report_path}")
    except OSError as exc:
        print(f"\n[경고] 결과 파일 저장에 실패했습니다: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(common.run_cli(main))
