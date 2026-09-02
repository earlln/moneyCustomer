import argparse
import joblib
import json
import os
import sys
import pandas as pd
import traceback
from konlpy.tag import Okt
import re

# PyInstaller 빌드에서 sklearn 및 LightGBM 모듈이 포함되도록 명시적 import
import sklearn
import sklearn.feature_extraction.text
import lightgbm
import sklearn.preprocessing

def preprocess_korean(text, okt):
    """한국어 전처리: 특수문자 제거 및 명사 추출"""
    if pd.isna(text):
        return ""
    text = re.sub(r'[^가-힣\s]', ' ', str(text))
    nouns = okt.nouns(text)
    return ' '.join([word for word in nouns if len(word) > 1])

def get_resource_path(filename):
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)

def load_model_artifacts():
    with open(get_resource_path('features.json'), 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    file_config = features.get('files', {})
    model_file = file_config.get('model', 'request_model.pkl')
    vectorizer_file = file_config.get('vectorizer', 'request_vectorizer.pkl')
    
    vectorizer = joblib.load(get_resource_path(vectorizer_file))
    model = joblib.load(get_resource_path(model_file))
    
    return vectorizer, model, features


def predict_batch(vectorizer, model, features, texts):
    X = vectorizer.transform(texts)
    pred_encoded = model.predict(X)
    prob_encoded = model.predict_proba(X)

    model_classes = list(model.classes_)
    all_classes = [0, 1, 2, 3]
    threshold = features.get('threshold', 0.5)
    major_classes = features.get('major_classes', [1, 2, 3])
    class_labels = features['class_labels']

    results = []
    for i in range(len(texts)):
        pred = int(pred_encoded[i])
        prob_row = prob_encoded[i]
        class_prob_map = {int(cls): float(prob) for cls, prob in zip(model_classes, prob_row)}
        probs = {cls: class_prob_map.get(cls, 0.0) for cls in all_classes}
        is_major = any(probs[cls] >= threshold for cls in major_classes)
        results.append({
            'prediction': pred,
            'label': class_labels.get(str(pred), f'Unknown_{pred}'),
            'prob_0': probs[0],
            'prob_1': probs[1],
            'prob_2': probs[2],
            'prob_3': probs[3],
            'is_major': is_major,
            'major_label': '주요작업' if is_major else '비주요작업'
        })
    return results

def main():
    parser = argparse.ArgumentParser(description='금전 대고객 작업 분류 - 배치 추론 v1.8')
    parser.add_argument('-i', '--input', default=None, help='입력 CSV 파일')
    parser.add_argument('-o', '--output', default=None, help='출력 CSV 파일')
    parser.add_argument('--show', action='store_true', help='결과 콘솔 출력')
    parser.add_argument('--encoding', default='utf-8', help='CSV 인코딩 (기본: utf-8)')
    args = parser.parse_args()

    print("=" * 50)
    print(" 금전 대고객 작업 분류 - 배치 추론 v1.8 (고성능)")
    print("=" * 50)

    print("\n모델 및 형태소 분석기 로딩 중...")
    vectorizer, model, features = load_model_artifacts()
    okt = Okt()
    
    file_config = features.get('files', {})
    input_file = args.input if args.input else file_config.get('input', 'input.csv')
    output_file = args.output if args.output else file_config.get('output', 'output.csv')

    if not os.path.exists(input_file):
        print(f"\n오류: 입력 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)

    print(f"\n입력 파일 읽기: {input_file}")
    try:
        df = pd.read_csv(input_file, encoding=args.encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='cp949')

    input_columns = features.get('input_columns', ['제목', '설명', '개발책임'])
    
    if input_columns:
        first_col = input_columns[0]
        if first_col in df.columns:
            stop_idx = len(df)
            for i in range(len(df)):
                val = df.iloc[i][first_col]
                if pd.isna(val) or str(val).strip() == "":
                    stop_idx = i
                    print(f"\n[알림] {i}행에서 빈 값이 발견되었습니다. 여기서 처리를 중단합니다.")
                    break
            if stop_idx < len(df):
                df = df.iloc[:stop_idx]

    texts = []
    for i in range(len(df)):
        combined_text = ""
        for col in input_columns:
            if col in df.columns and pd.notna(df.iloc[i][col]):
                combined_text += " " + str(df.iloc[i][col])
        texts.append(preprocess_korean(combined_text, okt))

    print("\n고성능 모델 예측 중...")
    results = predict_batch(vectorizer, model, features, texts)

    for key in ['prediction', 'label', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'is_major', 'major_label']:
        df[key] = [r[key] for r in results]

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {output_file}")

    print("\n=== 결과 요약 ===")
    print(f"총 건수: {len(df)}")
    print(f"주요작업: {len(df[df['is_major'] == True])}건")
    print(f"비주요작업: {len(df[df['is_major'] == False])}건")

    if args.show or len(df) <= 10:
        print("\n=== 결과 미리보기 ===")
        display_cols = [c for c in ['label', 'major_label', 'prob_0', 'prob_1', 'prob_2', 'prob_3'] if c in df.columns]
        print(df[display_cols].head(10).to_string(index=True))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("\n!!! 오류 발생 !!!")
        print(traceback.format_exc())
    finally:
        print("\n" + "="*50)
        input("프로그램을 종료하려면 Enter 키를 누르세요...")
