# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 스펙 - 금전 대고객 작업 분류 v2.0.0

세 실행 파일이 하나의 런타임(_internal)을 공유하도록 묶는다.
Java(JVM)는 전혀 필요하지 않다.

    pyinstaller --noconfirm MoneyCustomer.spec
"""

import os

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.getcwd())
EXE_NAMES = ("batch_predict", "train_model", "evaluate_model")

datas, binaries, hiddenimports = [], [], []

# kiwipiepy 는 별도 패키지(kiwipiepy_model)에 형태소 사전을 담고 있고,
# lightgbm 은 네이티브 라이브러리(lib_lightgbm)를 함께 배포해야 한다.
for package in ("kiwipiepy", "kiwipiepy_model", "lightgbm"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# 런타임에 동적으로 import 되어 정적 분석으로는 발견되지 않는 모듈들
hiddenimports += [
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.tree._utils",
    "sklearn.neighbors._partition_nodes",
    "scipy.special.cython_special",
    "joblib",
]

# 쓰이지 않으면서 용량만 키우는 패키지는 제외한다.
# konlpy/jpype 는 v2.0.0 에서 걷어낸 Java 의존 패키지이므로 반드시 빠져야 한다.
excludes = [
    "konlpy",
    "jpype",
    "jpype1",
    "matplotlib",
    "tkinter",
    "IPython",
    "notebook",
    "pytest",
    "sphinx",
]

a = Analysis(
    ["mc_entry.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

executables = [
    EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    for name in EXE_NAMES
]

coll = COLLECT(
    *executables,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MoneyCustomer",
)
