"""Java 없이 동작하는 한국어 토크나이저 (v2.0.0).

v1.8 까지는 KoNLPy 의 ``Okt`` 를 사용했기 때문에 JVM(Java 런타임)이 반드시
설치되어 있어야 했습니다. v2.0.0 은 JVM 없이 동작하는 두 가지 백엔드를 제공합니다.

- ``kiwi``  : `kiwipiepy` (C++ 로 구현된 한국어 형태소 분석기). 기본값이며 권장.
- ``regex`` : 외부 의존성이 전혀 없는 순수 파이썬 규칙 기반 토크나이저.
              `kiwipiepy` 설치가 불가능한 환경을 위한 대체 수단입니다.

학습(train_model.py)과 추론(batch_predict.py)은 **반드시 동일한 백엔드**로
동작해야 하므로, 학습 시 사용한 백엔드를 ``model_meta.json`` 에 기록하고
추론 시 이를 검증합니다.
"""

from __future__ import annotations

import re
import sys

# 기본 품사 필터: 일반명사, 고유명사, 의존명사, 외국어/한자, 어근
DEFAULT_POS_TAGS = ("NNG", "NNP", "NNB", "SL", "SH", "XR")
DEFAULT_MIN_LENGTH = 2

# 한글/영문/숫자만 남기고 나머지는 공백으로 치환한다.
# v1.8 은 한글만 남겼으나, 시스템명에 포함된 영문 약어(HI-sPortal, DB 등)가
# 분류에 유효한 신호이므로 v2.0.0 에서는 영문·숫자도 보존한다.
_CLEAN_RE = re.compile(r"[^0-9A-Za-z가-힣\s]+")
_KOREAN_ONLY_RE = re.compile(r"[^가-힣\s]+")

# regex 백엔드에서 어절 끝에 붙은 조사/어미를 잘라내기 위한 목록.
# 긴 것부터 검사해야 하므로 길이 내림차순으로 정렬해 둔다.
_PARTICLES = sorted(
    (
        "에서부터", "으로부터", "이라고는", "이라고", "라고는", "에게서", "께서는",
        "에서는", "으로는", "이라는", "라는", "에서도", "으로도", "만으로", "까지도",
        "부터는", "에게는", "이라도", "하고는", "이나마",
        "께서", "에서", "에게", "으로", "이라", "까지", "부터", "조차", "마저",
        "처럼", "보다", "이나", "이며", "이고", "이다", "인데", "한테", "더러",
        "이란", "이야", "커녕", "밖에", "대로", "만큼", "라도", "든지", "이든",
        "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만",
        "로", "야", "라", "며", "고", "나", "든",
    ),
    key=len,
    reverse=True,
)

# regex 백엔드 불용어(1~2글자 기능어 위주)
_STOPWORDS = frozenset(
    {
        "및", "등", "관련", "대한", "위한", "통한", "따른", "있는", "없는", "하는",
        "되는", "이번", "해당", "각각", "모든", "일부", "기타", "그리고", "또한",
        "경우", "대하여", "위하여",
    }
)


class TokenizerError(RuntimeError):
    """토크나이저 초기화 또는 사용 중 발생한 오류."""


class KoreanTokenizer:
    """백엔드에 관계없이 동일한 인터페이스를 제공하는 토크나이저.

    Parameters
    ----------
    backend:
        ``"kiwi"`` 또는 ``"regex"``.
    pos_tags:
        ``kiwi`` 백엔드에서 남길 품사 태그 목록.
    min_length:
        이 길이 미만의 토큰은 버린다 (v1.8 과 동일하게 기본 2).
    keep_latin:
        영문/숫자를 보존할지 여부. ``False`` 이면 v1.8 처럼 한글만 남긴다.
    """

    def __init__(
        self,
        backend: str = "kiwi",
        pos_tags=DEFAULT_POS_TAGS,
        min_length: int = DEFAULT_MIN_LENGTH,
        keep_latin: bool = True,
    ):
        backend = (backend or "kiwi").lower()
        if backend not in ("kiwi", "regex"):
            raise TokenizerError(
                f"지원하지 않는 토크나이저 백엔드입니다: {backend!r} (kiwi 또는 regex)"
            )

        self.backend = backend
        self.pos_tags = frozenset(pos_tags)
        self.min_length = int(min_length)
        self.keep_latin = bool(keep_latin)
        self._kiwi = None

        if backend == "kiwi":
            self._kiwi = self._load_kiwi()

    @staticmethod
    def _load_kiwi():
        try:
            from kiwipiepy import Kiwi
        except ImportError as exc:  # pragma: no cover - 설치 환경 의존
            raise TokenizerError(
                "kiwipiepy 를 불러올 수 없습니다. `pip install kiwipiepy` 로 설치하거나 "
                "features.json 의 tokenizer.backend 를 \"regex\" 로 변경하세요."
            ) from exc
        return Kiwi()

    # ------------------------------------------------------------------
    # 전처리
    # ------------------------------------------------------------------
    def clean(self, text) -> str:
        if text is None:
            return ""
        text = str(text)
        pattern = _CLEAN_RE if self.keep_latin else _KOREAN_ONLY_RE
        return pattern.sub(" ", text)

    # ------------------------------------------------------------------
    # 토큰화
    # ------------------------------------------------------------------
    def tokenize(self, text) -> list:
        """텍스트 하나를 토큰 리스트로 변환한다."""
        cleaned = self.clean(text)
        if not cleaned.strip():
            return []
        if self.backend == "kiwi":
            return self._tokenize_kiwi(cleaned)
        return self._tokenize_regex(cleaned)

    def transform(self, text) -> str:
        """토큰을 공백으로 이어붙인 문자열을 반환한다 (TF-IDF 입력용)."""
        return " ".join(self.tokenize(text))

    def transform_many(self, texts) -> list:
        """여러 문서를 한 번에 처리한다.

        ``kiwi`` 백엔드는 배치 API 를 사용하므로 문서 수가 많을 때 훨씬 빠르다.
        """
        texts = list(texts)
        if self.backend != "kiwi":
            return [self.transform(t) for t in texts]

        cleaned = [self.clean(t) for t in texts]
        results = []
        for tokens in self._kiwi.tokenize(cleaned):
            results.append(" ".join(self._filter_kiwi(tokens)))
        return results

    # ------------------------------------------------------------------
    def _filter_kiwi(self, tokens) -> list:
        out = []
        for tok in tokens:
            if tok.tag not in self.pos_tags:
                continue
            form = tok.form
            if len(form) < self.min_length:
                continue
            out.append(form.lower() if form.isascii() else form)
        return out

    def _tokenize_kiwi(self, cleaned: str) -> list:
        return self._filter_kiwi(self._kiwi.tokenize(cleaned))

    def _tokenize_regex(self, cleaned: str) -> list:
        out = []
        for word in cleaned.lower().split():
            if not word.isascii():
                word = self._strip_particle(word)
            if len(word) < self.min_length or word in _STOPWORDS:
                continue
            out.append(word)
        return out

    @staticmethod
    def _strip_particle(word: str) -> str:
        for particle in _PARTICLES:
            if word.endswith(particle) and len(word) - len(particle) >= 2:
                return word[: -len(particle)]
        return word

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        """학습 산출물에 기록할 토크나이저 설정."""
        info = {
            "backend": self.backend,
            "pos_tags": sorted(self.pos_tags),
            "min_length": self.min_length,
            "keep_latin": self.keep_latin,
        }
        if self.backend == "kiwi":
            try:
                import kiwipiepy

                info["kiwipiepy_version"] = kiwipiepy.__version__
            except Exception:  # pragma: no cover
                pass
        return info


def from_config(config: dict) -> KoreanTokenizer:
    """``features.json`` 의 ``tokenizer`` 섹션으로 토크나이저를 만든다."""
    section = (config or {}).get("tokenizer", {}) or {}
    return KoreanTokenizer(
        backend=section.get("backend", "kiwi"),
        pos_tags=section.get("pos_tags", DEFAULT_POS_TAGS),
        min_length=section.get("min_length", DEFAULT_MIN_LENGTH),
        keep_latin=section.get("keep_latin", True),
    )


def build_with_fallback(config: dict, log=print) -> KoreanTokenizer:
    """설정된 백엔드를 시도하고, 실패하면 ``regex`` 로 자동 대체한다.

    추론 경로에서는 학습 때와 다른 백엔드를 쓰면 정확도가 떨어지므로 사용하지
    않는다. 사용자가 직접 실험할 때를 위한 편의 함수다.
    """
    try:
        return from_config(config)
    except TokenizerError as exc:
        log(f"[경고] {exc}")
        log("[경고] regex 백엔드로 대체합니다. 정확도가 낮아질 수 있습니다.")
        section = dict((config or {}).get("tokenizer", {}) or {})
        section["backend"] = "regex"
        return from_config({"tokenizer": section})


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "S포탈 PUSH정보 및 발송정보 DB테이블 데이터 정비 배치 개발"
    for be in ("kiwi", "regex"):
        try:
            tk = KoreanTokenizer(backend=be)
        except TokenizerError as e:
            print(f"[{be}] 사용 불가: {e}")
            continue
        print(f"[{be}] {tk.transform(sample)}")
