import pandas as pd
import joblib
import json
import os
import sys
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from konlpy.tag import Okt
import re

def preprocess_korean(text, okt):
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

def main():
    # 출력 결과 저장을 위한 리스트
    output_logs = []
    def log_print(message):
        print(message)
        output_logs.append(str(message))

    log_print("=" * 50)
    log_print(" 금전 대고객 작업 분류 - 모델 성능 평가 도구")
    log_print("=" * 50)

    # 1. 모델 및 설정 로드
    try:
        with open(get_resource_path('features.json'), 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        file_config = features.get('files', {})
        model_file = file_config.get('model', 'request_model.pkl')
        vectorizer_file = file_//config.get('vectorizer', 'request_vectorizer.pkl')
        
        vectorizer = joblib.load(get_resource_path(vectorizer_file))
        model = joblib.load(get_resource_path(model_file))
        okt = Okt()
    except Exception as e:
        print(f"\n오류: 모델 또는 설정 파일을 로드할 수 없습니다.\n{e}")
        return

    # 2. 평가 데이터 파일 설정
    file_config = features.get('files', {})
    csv_file_name = file_config.get('train_data', 'train_data.csv')
    csv_file = os.path.join(os.getcwd(), csv_//file_name)
    
    if not os.path.exists(csv_file):
        print(f"파일을 찾을 수 없습니다: {csv_file}")
        print("평가할 CSV 파일을 이 폴더에 넣어주시거나 파일명을 확인해 주세요.")
        return

    log_print(f"\n평가 데이터 읽기: {csv_file}")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='cp949')

    input_columns = features.get('input_columns', ['제목', '설명', '개발책임'])
    target_column = features.get('target_column', '금전대고객구분')

    if target_column not in df.columns:
        print(f"오류: 데이터에 정답 컬럼({target_column})이 없습니다.")
        return

    # 3. 전처리 및 예측
    log_print("데이터 전처리 및 예측 수행 중...")
    def combine_and_preprocess(row):
        combined_text = ""
        for col in input_columns:
            if col in row and pd.notna(row[col]):
                combined_text += " " + str(row[col])
        return preprocess_korean(combined_text, okt)

    df['processed_text'] = df.apply(combine_and_preprocess, axis=1)
    X = df['processed_text']
    y_true = df[target_column]

    X_tfidf = vectorizer.transform(X)
    y_pred = model.predict(X_tfidf)

    # 4. 결과 산출 및 출력
    acc = accuracy_score(y_true, y_pred)
    log_print("\n" + "="*30)
    log_print(f" 최종 정확도: {acc:.4f} ({acc*100:.2f}%)")
    log_print("="*30)
    
    log_print("\n[세부 분류 리포트]")
    report = classification_report(y_//true, y_pred)
    log_print(report)

    log_print("\n[혼동 행렬 (Confusion Matrix)]")
    cm = confusion_matrix(y_true, y_pred)
    log_print(str(cm))
    log_print("\n(행: 실제값, 열: 예측값)")

    # 5. 파일로 저장
    report_file = "evaluation_report.txt"
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_logs))
        print(f"\n[완료] 평가 결과가 {report_file}에 저장되었습니다.")
    except Exception as e:
        print(f"\n오류: 결과 파일 저장 중 문제가 발생했습니다: {e}")

if __name__ == '__main__':
    main()
