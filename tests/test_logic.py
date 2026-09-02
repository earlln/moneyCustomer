"""핵심 판정 로직에 대한 단위 검증.

모델이나 대용량 데이터 없이 순수 로직만 확인하므로 빠르게 돌아간다.

    python tests/test_logic.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import batch_predict as bp
import common

fails = []
def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  '+extra if extra and not cond else ''}")
    if not cond: fails.append(name)

cols = ['작업의뢰명','작업의뢰내용','시스템명']
def frame(rows):
    return pd.DataFrame(rows, columns=cols)

blank = ['', '', '']
data  = ['제목', '내용', '시스템']

# --- trim_blank_rows -------------------------------------------------
quiet = lambda *a, **k: None

# 1) 끝의 빈 행만 제거
df = frame([data, data, blank, blank])
out = bp.trim_blank_rows(df, cols, 'trailing', log=quiet)
check("끝의 빈 행 2건 제거", len(out) == 2, f"got {len(out)}")

# 2) 중간 빈 행은 데이터를 버리지 않는다 (v1.8 의 데이터 손실 버그)
df = frame([data, blank, data, data])
out = bp.trim_blank_rows(df, cols, 'trailing', log=quiet)
check("중간 빈 행이 있어도 뒤 데이터 보존", len(out) == 4, f"got {len(out)} (v1.8은 1)")

# 3) v1.8 호환 정책은 그대로 첫 빈 행에서 중단
out = bp.trim_blank_rows(df, cols, 'stop_at_first', log=quiet)
check("stop_at_first 는 v1.8 과 동일", len(out) == 1, f"got {len(out)}")

# 4) 중간 + 끝 빈 행 혼합
df = frame([data, blank, data, blank, blank])
out = bp.trim_blank_rows(df, cols, 'trailing', log=quiet)
check("중간 보존 + 끝만 제거", len(out) == 3, f"got {len(out)}")

# 5) 빈 행 없음
df = frame([data, data])
check("빈 행 없으면 그대로", len(bp.trim_blank_rows(df, cols, 'trailing', log=quiet)) == 2)

# 6) 첫 컬럼만 비고 나머지는 값이 있으면 빈 행이 아니다
df = frame([['', '내용있음', '시스템'], data])
out = bp.trim_blank_rows(df, cols, 'trailing', log=quiet)
check("일부 컬럼만 빈 행은 유지", len(out) == 2, f"got {len(out)}")

# --- predict_batch: major_rule ---------------------------------------
class FakeVec:
    def transform(self, texts): return texts
class FakeModel:
    classes_ = [0,1,2,3]
    def __init__(self, probs): self._p = probs
    def predict(self, X): return [max(range(4), key=lambda i: p[i]) for p in self._p]
    def predict_proba(self, X): return self._p

cfg = {"threshold":0.5, "major_classes":[1,2,3],
       "class_labels":{"0":"비주요작업","1":"금전작업","2":"대고객작업","3":"금전+대고객"}}

# 확률 (0.40, 0.25, 0.20, 0.15): 주요작업 확률 합 0.60 이지만 개별로는 모두 0.5 미만
probs = [[0.40,0.25,0.20,0.15]]
r_any,_,dis = bp.predict_batch(FakeVec(), FakeModel(probs), {**cfg,"major_rule":"any"}, ["x"])
r_sum,_,_   = bp.predict_batch(FakeVec(), FakeModel(probs), {**cfg,"major_rule":"sum"}, ["x"])
check("any 규칙은 비주요작업으로 판정", r_any[0]["is_major"] is False)
check("sum 규칙은 주요작업으로 판정",   r_sum[0]["is_major"] is True)
check("prob_major = 주요클래스 확률의 합", abs(r_any[0]["prob_major"] - 0.60) < 1e-9,
      str(r_any[0]["prob_major"]))
check("규칙이 갈리는 건수를 보고", dis == 1, f"got {dis}")

# 명확한 케이스는 두 규칙이 일치
probs = [[0.05,0.90,0.03,0.02],[0.95,0.02,0.02,0.01]]
r,_,dis = bp.predict_batch(FakeVec(), FakeModel(probs), {**cfg,"major_rule":"any"}, ["x","y"])
check("명확한 케이스는 규칙 불일치 없음", dis == 0, f"got {dis}")
check("확실한 금전작업은 주요작업", r[0]["is_major"] is True)
check("확실한 비주요작업", r[1]["is_major"] is False)

# 잘못된 major_rule 은 명확히 거부
try:
    bp.predict_batch(FakeVec(), FakeModel([[0.25]*4]), {**cfg,"major_rule":"maybe"}, ["x"])
    check("잘못된 major_rule 거부", False)
except ValueError:
    check("잘못된 major_rule 거부", True)

# --- resolve_classes: 클래스 수 변경 대응 -----------------------------
class M3: classes_ = [0,1,2]
check("class_labels 기준으로 클래스 도출",
      common.resolve_classes({"class_labels":{"0":"a","1":"b","2":"c"}}, M3()) == [0,1,2])
check("설정에 없는 모델 클래스도 포함",
      common.resolve_classes({"class_labels":{"0":"a"}}, M3()) == [0,1,2])

# --- 배포되는 features.json 기본값 ------------------------------------
import json

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(root, 'features.json'), encoding='utf-8'))

check("features.json 의 major_rule 은 sum (v2.1.0 기본값)",
      cfg.get('major_rule') == 'sum', repr(cfg.get('major_rule')))
check("features.json 의 blank_row_policy 는 trailing",
      cfg.get('blank_row_policy') == 'trailing', repr(cfg.get('blank_row_policy')))
check("features.json 의 tokenizer 백엔드는 Java 를 쓰지 않는다",
      cfg.get('tokenizer', {}).get('backend') in ('kiwi', 'regex'),
      repr(cfg.get('tokenizer', {}).get('backend')))
check("major_classes 가 class_labels 안에 있다",
      all(str(c) in cfg.get('class_labels', {}) for c in cfg.get('major_classes', [])))

print()
print(f"=== {len(fails)} 실패 ===" if fails else "=== 전체 통과 ===")
sys.exit(1 if fails else 0)
