"""금전 대고객 작업 분류 - 배치 추론 모듈 (v2.0.0)

CSV 를 읽어 각 행을 분류하고 결과 CSV 를 저장합니다.
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

BASE_RESULT_COLUMNS = ("prediction", "label", "prob_major", "is_major", "major_label")


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
    """토큰화된 문서 리스트를 받아 행별 예측 결과를 반환한다.

    ``is_major`` 판정 규칙은 features.json 의 ``major_rule`` 로 정한다.

    - ``"any"``  : major_classes 중 하나라도 확률이 threshold 이상 (v1.8 호환, 기본)
    - ``"sum"``  : major_classes 확률의 **합**이 threshold 이상

    "주요작업" 은 곧 "비주요작업(0)이 아님" 이므로 확률적으로 옳은 쪽은 ``sum``
    이다. 예를 들어 확률이 (0.40, 0.25, 0.20, 0.15) 이면 주요작업일 확률이 0.60
    인데도 ``any`` 규칙은 비주요작업으로 판정한다. 다만 기존 집계와의 연속성을
    위해 기본값은 v1.8 과 같은 ``any`` 로 두고, 두 규칙의 판정이 갈리는 건수를
    실행 시 알려 준다.
    """
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    model_classes = [int(c) for c in model.classes_]
    all_classes = common.resolve_classes(config, model)
    threshold = config.get("threshold", 0.5)
    major_classes = [int(c) for c in config.get("major_classes", [1, 2, 3])]
    class_labels = config.get("class_labels", {})
    rule = str(config.get("major_rule", "any")).lower()
    if rule not in ("any", "sum"):
        raise ValueError(
            f"features.json 의 major_rule 값이 잘못되었습니다: {rule!r} (any 또는 sum)"
        )

    results, disagreements = [], 0
    for i in range(len(texts)):
        # 학습 데이터에 없던 클래스는 확률 0 으로 채워 컬럼 구성을 항상 동일하게 유지한다.
        prob_by_class = dict(zip(model_classes, (float(p) for p in probabilities[i])))
        probs = {c: prob_by_class.get(c, 0.0) for c in all_classes}
        prob_major = sum(probs.get(c, 0.0) for c in major_classes)

        any_major = any(probs.get(c, 0.0) >= threshold for c in major_classes)
        sum_major = prob_major >= threshold
        if any_major != sum_major:
            disagreements += 1
        is_major = sum_major if rule == "sum" else any_major

        pred = int(predictions[i])
        row = {
            "prediction": pred,
            "label": class_labels.get(str(pred), f"Unknown_{pred}"),
            "prob_major": prob_major,
            "is_major": is_major,
            "major_label": "주요작업" if is_major else "비주요작업",
        }
        row.update({f"prob_{c}": probs[c] for c in all_classes})
        results.append(row)

    return results, all_classes, disagreements


def trim_blank_rows(df, input_columns, policy="trailing", log=print):
    """입력 CSV 의 빈 행을 처리한다.

    엑셀에서 저장한 CSV 는 끝에 빈 행이 붙는 경우가 많아 이를 걸러내야 한다.
    다만 v1.8 은 **첫 입력 컬럼이 빈 첫 행에서 무조건 처리를 중단**했기 때문에,
    파일 중간에 제목이 비어 있는 행이 하나만 있어도 그 뒤 데이터 전체가 조용히
    누락됐다. 기본 정책 ``"trailing"`` 은 파일 **끝**의 빈 행만 제거하고,
    중간의 빈 행은 건너뛰지 않은 채 경고만 남긴다.

    ``policy="stop_at_first"`` 로 두면 v1.8 과 동일하게 동작한다.
    """
    present = [c for c in input_columns if c in df.columns]
    if not present or df.empty:
        return df

    block = df[present]
    is_blank = (block.isna() | (block.astype(str).apply(lambda col: col.str.strip()) == "")).all(axis=1)
    blank = is_blank.to_numpy()
    if not blank.any():
        return df

    if policy == "stop_at_first":
        stop = int(blank.argmax())
        log(f"[알림] {stop}행이 비어 있어 여기서 처리를 중단합니다 (v1.8 호환 정책).")
        return df.iloc[:stop]

    trailing = 0
    for value in blank[::-1]:
        if not value:
            break
        trailing += 1

    interior = int(blank[: len(blank) - trailing].sum())
    if interior:
        log(
            f"[경고] 입력값이 모두 비어 있는 행이 파일 중간에 {interior:,}건 있습니다.\n"
            "       건너뛰지 않고 그대로 예측하므로 해당 행의 결과는 신뢰할 수 없습니다."
        )
    if trailing:
        log(f"[알림] 파일 끝의 빈 행 {trailing:,}건을 제외했습니다.")
        return df.iloc[: len(df) - trailing]
    return df


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
    df = trim_blank_rows(df, input_columns, config.get("blank_row_policy", "trailing"))
    if df.empty:
        raise ValueError("처리할 데이터가 없습니다. 입력 CSV 를 확인하세요.")

    print("[3/4] 전처리 및 예측 중...")
    texts = tokenizer.transform_many(common.combine_columns(df, input_columns))
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        print(
            f"      [경고] 분석할 단어가 하나도 남지 않은 행이 {empty:,}건 있습니다.\n"
            "             해당 행의 예측은 근거가 없으므로 결과를 확인하세요."
        )

    results, all_classes, disagreements = predict_batch(vectorizer, model, config, texts)

    df = df.copy()
    columns = list(BASE_RESULT_COLUMNS) + [f"prob_{c}" for c in all_classes]
    for key in columns:
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
    counts = df["prediction"].value_counts()
    for cls in all_classes:
        print(f"  {cls} {labels.get(str(cls), ''):<12} {int(counts.get(cls, 0)):>7,}건")

    if disagreements:
        rule = str(config.get("major_rule", "any")).lower()
        print(
            f"\n[참고] 주요작업 판정 규칙(any/sum)에 따라 결과가 달라지는 행이 {disagreements:,}건 있습니다.\n"
            f"       현재 규칙은 '{rule}' 입니다. prob_major 컬럼(주요작업 확률의 합)을 함께\n"
            "       확인하시고, 필요하면 features.json 의 major_rule 을 조정하세요."
        )

    if args.show or len(df) <= 10:
        print("\n[결과 미리보기]")
        preview = ["label", "major_label", "prob_major"] + [f"prob_{c}" for c in all_classes]
        cols = [c for c in preview if c in df.columns]
        with pd.option_context("display.width", 200):
            print(df[cols].head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(common.run_cli(main))
