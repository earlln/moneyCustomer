"""PyInstaller 번들의 공용 진입점 (v2.0.0).

세 실행 파일(batch_predict / train_model / evaluate_model)은 파이썬 런타임과
scikit-learn·LightGBM·kiwipiepy 모델 등 350MB 가 넘는 동일한 의존성을 공유한다.
각각을 따로 묶으면 배포 용량이 1GB 를 넘기 때문에, 하나의 번들에 세 개의
실행 파일을 만들고 **실행 파일 이름**으로 어떤 모듈을 실행할지 결정한다.

소스에서 직접 실행할 때는 각 스크립트를 그대로 실행하면 되며 이 파일은 필요 없다.
"""

from __future__ import annotations

import os
import sys

import batch_predict
import common
import evaluate_model
import train_model

_COMMANDS = {
    "batch_predict": batch_predict.main,
    "train_model": train_model.main,
    "evaluate_model": evaluate_model.main,
}
_DEFAULT = "batch_predict"


def _command_name() -> str:
    """실행 파일 이름에서 실행할 명령을 알아낸다."""
    argv0 = sys.executable if getattr(sys, "frozen", False) else sys.argv[0]
    stem = os.path.splitext(os.path.basename(argv0 or ""))[0].lower()
    if stem in _COMMANDS:
        return stem
    # mc_entry 등 이름이 맞지 않으면 첫 인자를 명령으로 해석한다.
    if len(sys.argv) > 1 and sys.argv[1] in _COMMANDS:
        return sys.argv.pop(1)
    return _DEFAULT


def main() -> int:
    return common.run_cli(_COMMANDS[_command_name()])


if __name__ == "__main__":
    sys.exit(main())
