"""금전 대고객 작업 분류 - 배치 추론 모듈 (v2.0.0)

CSV 를 읽어 각 행을 4개 클래스로 분류하고 결과 CSV 를 저장합니다.
v1.8 과 달리 Java(JVM)가 필요 없습니다.
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd

# PyInstaller 가 의존 모듈을 놓치지 않도록 명시적으로 import 한다.
import sklearn  # noqa: F401
import sklearn.feature_extraction.text  # noqa: F401
import sklearn.preprocessing  # noqa: F401
import lightgbm  # noqa: F401

import common
from korean_tokenizer import from_config

ALL_CLASSES = (0, 1, 2, 3)
RESULT_COLUMNS = (
    "prediction", "label", "prob_0", "prob_1", "prob_2", "prob_3",
    "is_major", "major_label",
)


def load_artifacts(config):
    files = config.get("files", {}) or {}
    vectorizer_path = common.resource_path(files.get("vectorizer", "request_vectorizer.pkl"))
    model_path = common.resource_path(files.get("model", "request_model.pkl"))
    for path, what in ((vectorizer_path, "벡터라이저"), (model_path, "모델")):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{what} 파일을 찾을 수 없습니다: {path}\n"
                "  train_model 을 먼저 실행해 모델을 학습하세요."
            )
    return joblib.load(vectorizer_path), joblib.load(model_path)


def predict_batch(vectorizer, model, config, texts):
    """토큰화된 문서 리스트를 받아 행별 예측 결과 딕셔너리 리스트를 반환한다."""
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    model_classes = [int(c) for c in model.classes_]
    threshold = config.get("threshold", 0.5)
    major_classes = config.get("major_classes", [1, 2, 3])
    class_labels = config.get("class_labels", {})

    results = []
    for i in range(len(texts)):
        # 학습 데이터에 없던 클래스는 확률 0 으로 채워 컬럼 구성을 항상 동일하게 유지한다.
        prob_by_class = dict(zip(model_classes, (float(p) for p in probabilities[i])))
        probs = {c: prob_by_class.get(c, 0.0) for c in ALL_CLASSES}
        pred = int(predictions[i])
        is_major = any(probs.get(c, 0.0) >= threshold for c in major_classes)
        results.append(
            {
                "prediction": pred,
                "label": class_labels.get(str(pred), f"Unknown_{pred}"),
                "prob_0": probs[0],
                "prob_1": probs[1],
                "prob_2": probs[2],
                "prob_3": probs[3],
                "is_major": is_major,
                "major_label": "주요작업" if is_major else "비주요작업",
            }
        )
    return results


def truncate_at_blank(df, input_columns, log=print):
    """첫 입력 컬럼이 비어 있는 행부터 잘라낸다 (v1.8 과 동일한 동작).

    엑셀에서 저장한 CSV 뒤쪽에 붙는 빈 행을 처리하지 않기 위한 규칙이다.
    """
    if not input_columns:
        return df
    first = input_columns[0]
    if first not in df.columns:
        return df
    blank = (df[first].isna() | (df[first].astype(str).str.strip() == "")).to_numpy()
    if not blank.any():
        return df
    stop = int(blank.argmax())  # 첫 번째 빈 행의 위치
    log(f"[알림] {stop}행의 '{first}' 값이 비어 있어 이 지점부터 처리를 중단합니다.")
    return df.iloc[:stop]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="금전 대고객 작업 분류 - 배치 추론 v2.0.0 (Java 불필요)"
    )
    parser.add_argument("-i", "--input", default=None, help="입력 CSV 파일 (기본: features.json 설정값)")
    parser.add_argument("-o", "--output", default=None, help="출력 CSV 파일 (기본: features.json 설정값)")
    parser.add_argument("--show", action="store_true", help="결과 미리보기를 콘솔에 출력")
    parser.add_argument("--encoding", default=None, help="입력 CSV 인코딩 (기본: 자동 판별)")
    parser.add_argument("--no-pause", action="store_true", help="종료 시 Enter 대기 생략")
    args = parser.parse_args(argv)

    print("=" * 62)
    print(f" 금전 대고객 작업 분류 - 배치 추론 v{common.APP_VERSION} (Java 불필요)")
    print("=" * 62)

    config = common.load_config()
    files = config.get("files", {}) or {}
    input_file = common.resource_path(args.input or files.get("input", "input.csv"))
    output_file = common.resource_path(args.output or files.get("output", "output.csv"))

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")

    print("\n[1/4] 모델 및 토크나이저 로딩 중...")
    vectorizer, model = load_artifacts(config)
    tokenizer = from_config(config)
    print(f"      토크나이저 백엔드: {tokenizer.backend} (JVM 불필요)")
    common.check_tokenizer_match(common.load_meta(), tokenizer)

    print(f"[2/4] 입력 파일 읽기: {input_file}")
    df, used_encoding = common.read_csv_auto(input_file, args.encoding)
    print(f"      {len(df):,}행 로드 (인코딩: {used_encoding})")

    input_columns = config.get("input_columns", [])
    df = truncate_at_blank(df, input_columns)
    if df.empty:
        raise ValueError("처리할 데이터가 없습니다. 입력 CSV 를 확인하세요.")

    print("[3/4] 전처리 및 예측 중...")
    texts = tokenizer.transform_many(common.combine_columns(df, input_columns))
    results = predict_batch(vectorizer, model, config, texts)

    df = df.copy()
    for key in RESULT_COLUMNS:
        df[key] = [r[key] for r in results]

    print(f"[4/4] 결과 저장: {output_file}")
    # utf-8-sig 로 저장해야 엑셀에서 한글이 깨지지 않는다.
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    major = int(df["is_major"].sum())
    print("\n" + "=" * 62)
    print(" 결과 요약")
    print("=" * 62)
    print(f" 총 건수   : {len(df):,}")
    print(f" 주요작업  : {major:,}건")
    print(f" 비주요작업: {len(df) - major:,}건")
    print("\n[클래스별 건수]")
    labels = config.get("class_labels", {})
    counts = df["prediction"].value_counts().sort_index()
    for cls, count in counts.items():
        print(f"  {cls} {labels.get(str(int(cls)), ''):<12} {count:>7,}건")

    if args.show or len(df) <= 10:
        print("\n[결과 미리보기]")
        cols = [c for c in ("label", "major_label", "prob_0", "prob_1", "prob_2", "prob_3") if c in df.columns]
        with pd.option_context("display.width", 200):
            print(df[cols].head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(common.run_cli(main))
