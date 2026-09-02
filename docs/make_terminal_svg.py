"""터미널 출력 텍스트를 README 에 넣을 SVG 이미지로 변환한다.

docs/captures/*.txt 의 내용은 실제 실행 출력을 그대로 저장한 것이고,
이 스크립트는 그 텍스트를 터미널 창 모양으로 렌더링하기만 한다.
출력 내용을 손대지 않으므로, 캡처를 갱신하려면 txt 를 새 실행 결과로
바꾼 뒤 이 스크립트를 다시 돌리면 된다.

    python docs/make_terminal_svg.py
"""

from __future__ import annotations

import os
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURE_DIR = os.path.join(ROOT, "docs", "captures")
IMAGE_DIR = os.path.join(ROOT, "docs", "images")

# 렌더링 상수 (등폭 글꼴 기준)
FONT_SIZE = 13
CHAR_W = FONT_SIZE * 0.60      # 등폭 글꼴의 대략적인 글자 폭
LINE_H = 19
PAD_X, PAD_TOP, PAD_BOTTOM = 16, 14, 16
TITLEBAR_H = 30

BG = "#1e1e2e"
TITLEBAR_BG = "#181825"
FG = "#cdd6f4"
TITLE_FG = "#9399b2"
PROMPT_FG = "#a6e3a1"
DOTS = ("#f38ba8", "#f9e2af", "#a6e3a1")

FONT_STACK = (
    "ui-monospace, SFMono-Regular, Menlo, Consolas, "
    "'DejaVu Sans Mono', 'D2Coding', 'Malgun Gothic', monospace"
)


def display_width(text: str) -> int:
    """한글·한자처럼 두 칸을 차지하는 글자를 반영한 표시 폭."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render(title: str, command: str, body_lines: list, description: str) -> str:
    lines = [command] + body_lines
    cols = max(display_width(line) for line in lines) if lines else 40
    width = int(cols * CHAR_W) + PAD_X * 2
    height = TITLEBAR_H + PAD_TOP + len(lines) * LINE_H + PAD_BOTTOM

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(description)}">',
        f"<title>{escape(description)}</title>",
        f'<rect width="{width}" height="{height}" rx="8" fill="{BG}"/>',
        f'<path d="M0 8a8 8 0 0 1 8-8h{width - 16}a8 8 0 0 1 8 8v{TITLEBAR_H - 8}H0z" fill="{TITLEBAR_BG}"/>',
    ]
    for i, color in enumerate(DOTS):
        out.append(f'<circle cx="{18 + i * 18}" cy="{TITLEBAR_H / 2}" r="5.5" fill="{color}"/>')
    out.append(
        f'<text x="{width / 2}" y="{TITLEBAR_H / 2 + 4}" fill="{TITLE_FG}" '
        f'font-family="{FONT_STACK}" font-size="11.5" text-anchor="middle">{escape(title)}</text>'
    )

    y = TITLEBAR_H + PAD_TOP + FONT_SIZE
    # 첫 줄은 실행한 명령(프롬프트), 나머지는 프로그램이 출력한 내용 그대로
    out.append(
        f'<text x="{PAD_X}" y="{y}" fill="{PROMPT_FG}" font-family="{FONT_STACK}" '
        f'font-size="{FONT_SIZE}" xml:space="preserve">{escape(command)}</text>'
    )
    for line in body_lines:
        y += LINE_H
        if not line.strip():
            continue
        out.append(
            f'<text x="{PAD_X}" y="{y}" fill="{FG}" font-family="{FONT_STACK}" '
            f'font-size="{FONT_SIZE}" xml:space="preserve">{escape(line)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


CAPTURES = (
    {
        "name": "batch_predict",
        "title": "명령 프롬프트  —  MoneyCustomer",
        "command": "C:\\MoneyCustomer> batch_predict.exe -i sample_input.csv -o output.csv --show",
        "description": "batch_predict.exe 를 실행해 12건을 분류한 실제 출력",
    },
    {
        "name": "evaluate_model",
        "title": "명령 프롬프트  —  MoneyCustomer",
        "command": "C:\\MoneyCustomer> evaluate_model.exe -d sample_input.csv",
        "description": "evaluate_model.exe 로 모델 성능을 평가한 실제 출력",
    },
    {
        "name": "train_model",
        "title": "명령 프롬프트  —  MoneyCustomer",
        "command": "C:\\MoneyCustomer> train_model.exe",
        "description": "train_model.exe 로 모델을 재학습한 실제 출력",
    },
)


def main() -> int:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    for cap in CAPTURES:
        src = os.path.join(CAPTURE_DIR, f"{cap['name']}.txt")
        with open(src, encoding="utf-8") as f:
            body = f.read().rstrip("\n").split("\n")
        svg = render(cap["title"], cap["command"], body, cap["description"])
        dst = os.path.join(IMAGE_DIR, f"{cap['name']}.svg")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"{os.path.relpath(dst, ROOT)}  ({len(body)}줄, {len(svg):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
