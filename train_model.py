"""금전 대고객 작업 분류 - 모델 학습 모듈 (v2.1.0)

v1.8 과 달리 Java(JVM)가 필요 없습니다. 형태소 분석은 ``korean_tokenizer`` 가
담당하며 기본 백엔드는 순수 네이티브 구현인 ``kiwipiepy`` 입니다.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import common
from korean_tokenizer import from_config

LOG_FILENAME = "train_log.txt"


def make_logger(log_path):
    def log(message=""):
        text = str(message)
        print(text)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass

    return log


def build_vectorizer(config):
    """features.json 의 ``vectorizer`` 섹션으로 TF-IDF 벡터라이저를 만든다."""
    section = config.get("vectorizer", {}) or {}
    return TfidfVectorizer(
        max_features=section.get("max_features", 30000),
        ngram_range=tuple(section.get("ngram_range", [1, 2])),
        min_df=section.get("min_df", 1),
        sublinear_tf=section.get("sublinear_tf", True),
    )


def build_model(config):
    section = config.get("model_params", {}) or {}
    return LGBMClassifier(
        n_estimators=section.get("n_estimators", 500),
        learning_rate=section.get("learning_rate", 0.05),
        num_leaves=section.get("num_leaves", 31),
        class_weight=section.get("class_weight", "balanced"),
        random_state=section.get("random_state", 42),
        n_jobs=-1,
        importance_type="gain",
        verbose=-1,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="금전 대고객 작업 분류 - 모델 학습 v2.1.0 (Java 불필요)"
    )
    parser.add_argument("-d", "--data", default=None, help="학습 CSV 파일 (기본: features.json 설정값)")
    parser.add_argument("--encoding", default=None, help="CSV 인코딩 (기본: 자동 판별)")
    parser.add_argument("--test-size", type=float, default=0.2, help="검증 데이터 비율 (기본: 0.2)")
    parser.add_argument("--no-pause", action="store_true", help="종료 시 Enter 대기 생략")
    args = parser.parse_args(argv)

    log_path = common.resource_path(LOG_FILENAME)
    log = make_logger(log_path)

    log("=" * 62)
    log(f" 금전 대고객 작업 분류 - 모델 학습 v{common.APP_VERSION} (Java 불필요)")
    log("=" * 62)
    log(f"시작 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"작업 디렉터리: {common.base_dir()}")

    config = common.load_config()
    input_columns = config.get("input_columns")
    target_column = config.get("target_column")
    if not input_columns or not target_column:
        raise ValueError("features.json 에 input_columns 또는 target_column 설정이 없습니다.")

    files = config.get("files", {}) or {}
    csv_path = common.resource_path(args.data or files.get("train_data", "train_data.csv"))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"학습 데이터를 찾을 수 없습니다: {csv_path}")

    log(f"입력 컬럼: {input_columns}")
    log(f"정답 컬럼: {target_column}")
    log(f"학습 데이터: {csv_path}")

    log("\n[1/5] 토크나이저 준비 중...")
    tokenizer = from_config(config)
    log(f"      백엔드: {tokenizer.backend} (JVM 불필요)")

    log("[2/5] 데이터 로드 중...")
    df, used_encoding = common.read_csv_auto(csv_path, args.encoding)
    log(f"      {len(df):,}행 로드 (인코딩: {used_encoding})")

    before = len(df)
    df = df.dropna(subset=[target_column])
    if before != len(df):
        log(f"      정답이 비어 있는 {before - len(df):,}행 제외")

    log("[3/5] 형태소 분석 및 전처리 중...")
    raw_texts = common.combine_columns(df, input_columns, log=log)
    texts = tokenizer.transform_many(raw_texts)
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        log(f"      [알림] 토큰이 하나도 남지 않은 행 {empty:,}건")

    y = df[target_column]
    if not 0.0 < args.test_size < 1.0:
        raise ValueError(f"--test-size 는 0 과 1 사이여야 합니다: {args.test_size}")

    # stratify 분할은 클래스마다 최소 2건이 필요하다. 미리 확인해 sklearn 의
    # 난해한 예외 대신 원인을 짚어 주는 메시지를 낸다.
    counts = y.value_counts()
    too_few = counts[counts < 2]
    if len(too_few):
        raise ValueError(
            "학습 데이터가 너무 적은 클래스가 있어 분할할 수 없습니다.\n"
            + "\n".join(f"  클래스 {c}: {n}건 (최소 2건 필요)" for c, n in too_few.items())
        )
    if len(counts) < 2:
        raise ValueError(
            f"정답 컬럼 '{target_column}' 에 클래스가 하나뿐입니다. 분류 학습을 할 수 없습니다."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=args.test_size, random_state=42, stratify=y
    )
    log(f"      학습 {len(X_train):,}행 / 검증 {len(X_test):,}행")

    log("[4/5] TF-IDF 벡터화 및 LightGBM 학습 중...")
    vectorizer = build_vectorizer(config)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    log(f"      특징 수: {X_train_tfidf.shape[1]:,}")

    model = build_model(config)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    holdout_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, model.predict(X_train_tfidf))

    log("\n" + "=" * 62)
    log(" 모델 성능 (검증 데이터 = 학습에 사용하지 않은 데이터)")
    log("=" * 62)
    log(f" 검증 정확도(holdout) : {holdout_acc:.4f} ({holdout_acc * 100:.2f}%)")
    log(f" 학습 정확도(train)   : {train_acc:.4f} ({train_acc * 100:.2f}%)")
    log("\n[분류 리포트 - 검증 데이터]")
    log(classification_report(y_test, y_pred, zero_division=0))
    log("[혼동 행렬 - 검증 데이터] (행: 실제값, 열: 예측값)")
    log(str(confusion_matrix(y_test, y_pred)))

    log("\n[5/5] 산출물 저장 중...")
    model_path = common.resource_path(files.get("model", "request_model.pkl"))
    vectorizer_path = common.resource_path(files.get("vectorizer", "request_vectorizer.pkl"))
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    log(f"      모델: {model_path}")
    log(f"      벡터라이저: {vectorizer_path}")

    meta_path = common.save_meta(
        {
            "app_version": common.APP_VERSION,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "tokenizer": tokenizer.describe(),
            "input_columns": list(input_columns),
            "target_column": target_column,
            "train_data": os.path.basename(csv_path),
            # evaluate_model 이 동일한 분할을 재현해도 되는지 판단하는 근거
            "train_data_fingerprint": common.data_fingerprint(csv_path),
            "n_rows": int(len(df)),
            "n_features": int(X_train_tfidf.shape[1]),
            "classes": [int(c) for c in model.classes_],
            "metrics": {
                "holdout_accuracy": round(float(holdout_acc), 6),
                "train_accuracy": round(float(train_acc), 6),
                "test_size": args.test_size,
            },
            "library_versions": _library_versions(),
        }
    )
    log(f"      메타데이터: {meta_path}")
    log(f"\n완료 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


def _library_versions():
    import sklearn
    import lightgbm

    versions = {
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "pandas": pd.__version__,
    }
    try:
        import kiwipiepy

        versions["kiwipiepy"] = kiwipiepy.__version__
    except ImportError:
        pass
    return versions


if __name__ == "__main__":
    sys.exit(common.run_cli(main))
