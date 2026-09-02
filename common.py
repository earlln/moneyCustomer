"""세 실행 모듈(train / predict / evaluate)이 공유하는 유틸리티 (v2.1.0)."""

from __future__ import annotations

import hashlib
import json
import os
import sys

APP_VERSION = "2.1.0"
META_FILENAME = "model_meta.json"
_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp949", "euc-kr")


def configure_stdio() -> None:
    """표준 출력이 한글을 항상 안전하게 출력하도록 UTF-8 로 고정한다.

    Windows 콘솔에 직접 붙어 있을 때는 파이썬이 유니코드 API 를 쓰므로 문제가
    없지만, 출력을 파일이나 파이프로 넘기면(배치 스케줄러의 ``> log.txt``,
    다른 프로그램으로의 파이프 등) 시스템 ANSI 코드페이지가 쓰인다.
    영문 Windows(cp1252)에서는 이때 한글을 인코딩하지 못해
    ``UnicodeEncodeError`` 로 프로그램이 죽는다.

    출력 실패로 작업 자체가 중단되는 일이 없도록 인코딩을 UTF-8 로 맞추고,
    그래도 표현할 수 없는 문자는 예외 대신 대체 표기로 흘려보낸다.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            # 리다이렉션 방식에 따라 reconfigure 가 없거나 실패할 수 있다.
            # 그 경우에도 프로그램은 계속 진행되어야 한다.
            pass


configure_stdio()


def base_dir() -> str:
    """스크립트 또는 exe 가 놓인 디렉터리.

    PyInstaller 로 묶인 경우 임시 해제 경로(``sys._MEIPASS``)가 아니라
    **실행 파일이 있는 디렉터리**를 반환한다. 모델·설정·CSV 는 exe 옆에 두고
    사용자가 직접 교체할 수 있어야 하기 때문이다.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(filename: str) -> str:
    """상대 경로는 base_dir 기준으로 풀고, 절대 경로는 그대로 반환한다."""
    return filename if os.path.isabs(filename) else os.path.join(base_dir(), filename)


def load_config(filename: str = "features.json") -> dict:
    path = resource_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_auto(path, encoding=None):
    """인코딩을 자동 판별해 CSV 를 읽고 ``(DataFrame, 사용한 인코딩)`` 을 반환한다.

    사내 데이터가 CP949 와 UTF-8 로 섞여 들어오므로, 지정 인코딩 → UTF-8 →
    UTF-8-SIG → CP949 → EUC-KR 순으로 시도한다.
    """
    import pandas as pd

    tried, errors = [], []
    for enc in ((encoding,) if encoding else ()) + _ENCODING_CANDIDATES:
        if not enc or enc in tried:
            continue
        tried.append(enc)
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as exc:
            errors.append(f"  - {enc}: {exc}")
    raise ValueError(
        "CSV 인코딩을 판별하지 못했습니다: " + str(path) + "\n" + "\n".join(errors)
    )


def combine_columns(df, input_columns, log=print):
    """설정된 입력 컬럼들을 공백으로 이어붙여 문서 리스트를 만든다.

    없는 컬럼은 경고만 남기고 건너뛴다. 하나도 없으면 예외를 발생시킨다.
    """
    present = [c for c in input_columns if c in df.columns]
    missing = [c for c in input_columns if c not in df.columns]
    if missing:
        log(f"[경고] CSV 에 없는 입력 컬럼을 건너뜁니다: {missing}")
        log(f"       CSV 의 실제 컬럼: {list(df.columns)}")
    if not present:
        raise KeyError(
            "입력 컬럼을 하나도 찾을 수 없습니다.\n"
            f"  features.json 의 input_columns: {list(input_columns)}\n"
            f"  CSV 의 실제 컬럼: {list(df.columns)}"
        )
    block = df[present]
    return block.where(block.notna(), "").astype(str).agg(" ".join, axis=1).tolist()


def data_fingerprint(path: str) -> dict:
    """CSV 파일의 지문(크기 + SHA-256).

    evaluate_model 은 학습 때와 동일한 분할을 재현해 검증 구간을 골라낸다.
    파일이 한 줄이라도 바뀌면 그 재현은 성립하지 않으므로, 학습 시점의 지문을
    남겨 두고 평가 시 비교한다. 지문이 다르면 "검증 구간" 이라고 부를 수 없다.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": os.path.getsize(path)}


def resolve_classes(config: dict, model) -> list:
    """출력할 클래스 번호 목록을 결정한다.

    features.json 의 class_labels 를 우선하되, 모델이 실제로 알고 있는 클래스가
    설정에 빠져 있으면 함께 포함한다. 클래스 수를 3개나 5개로 바꿔도 확률 컬럼
    구성이 따라가도록 하기 위함이다.
    """
    labels = (config or {}).get("class_labels") or {}
    classes = set()
    for key in labels:
        try:
            classes.add(int(key))
        except (TypeError, ValueError):
            continue
    classes.update(int(c) for c in getattr(model, "classes_", []))
    return sorted(classes)


def save_meta(meta: dict, filename: str = META_FILENAME) -> str:
    path = resource_path(filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def load_meta(filename: str = META_FILENAME):
    path = resource_path(filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def check_tokenizer_match(meta, tokenizer, log=print) -> bool:
    """학습에 사용한 토크나이저와 현재 토크나이저가 같은지 확인한다.

    다르면 벡터라이저의 어휘와 토큰이 어긋나 예측 품질이 크게 떨어지므로
    눈에 띄는 경고를 출력한다.
    """
    if not meta:
        log("[알림] model_meta.json 이 없어 토크나이저 일치 여부를 확인할 수 없습니다.")
        return True

    trained = meta.get("tokenizer") or {}
    current = tokenizer.describe()
    diffs = [
        k for k in ("backend", "pos_tags", "min_length", "keep_latin")
        if trained.get(k) != current.get(k)
    ]
    if not diffs:
        return True

    log("")
    log("!" * 62)
    log("[경고] 학습 시점과 다른 토크나이저 설정으로 추론하고 있습니다.")
    for k in diffs:
        log(f"       - {k}: 학습={trained.get(k)!r} / 현재={current.get(k)!r}")
    log("       예측 정확도가 크게 떨어질 수 있습니다. features.json 의 tokenizer")
    log("       설정을 학습 때와 동일하게 맞추거나 train_model 로 재학습하세요.")
    log("!" * 62)
    log("")
    return False


def run_cli(main_func, argv=None) -> int:
    """세 모듈이 공유하는 CLI 진입 래퍼.

    사용자 실수(파일 없음, 설정 오류)는 짧은 한 줄 메시지로, 예기치 못한 오류는
    전체 트레이스백으로 보고한다. 종료 코드: 0 정상 / 1 내부 오류 / 2 사용자 오류.
    """
    from korean_tokenizer import TokenizerError

    args = list(sys.argv[1:] if argv is None else argv)
    no_pause = "--no-pause" in args
    exit_code = 0
    try:
        exit_code = main_func(argv) or 0
    except TokenizerError as exc:
        print(f"\n[오류] 토크나이저를 준비할 수 없습니다.\n{exc}")
        exit_code = 2
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"\n[오류] {exc}")
        exit_code = 2
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 실행을 취소했습니다.")
        exit_code = 130
    except Exception:
        import traceback

        print("\n!!! 예기치 못한 오류가 발생했습니다 !!!")
        print(traceback.format_exc())
        exit_code = 1
    finally:
        print("\n" + "=" * 62)
        pause(no_pause)
    return exit_code


def pause(no_pause: bool = False) -> None:
    """탐색기에서 더블클릭 실행했을 때 창이 즉시 닫히지 않도록 대기한다.

    CI·배치 등 비대화형 환경이거나 ``--no-pause`` / ``MC_NO_PAUSE`` 지정 시에는
    대기하지 않으므로 자동화 파이프라인에서도 그대로 쓸 수 있다.
    """
    if no_pause or os.environ.get("MC_NO_PAUSE"):
        return
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return
    except (AttributeError, ValueError):
        return
    try:
        input("프로그램을 종료하려면 Enter 키를 누르세요...")
    except (EOFError, KeyboardInterrupt):
        pass
