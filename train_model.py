import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from konlpy.tag import Okt
import re
import json
import os
import sys
import traceback

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

os.chdir(script_dir)

log_file = os.path.join(script_dir, "train_log.txt")
def log(message):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def preprocess_korean(text, okt):
    """한국어 전처리: 특수문자 제거 및 명사 추출"""
    if pd.isna(text):
        return ""
    # 특수문자 및 숫자 제거
    text = re.sub(r'[^가-힣\s]', ' ', str(text))
    # 명사만 추출
    nouns = okt.nouns(text)
    # 1글자 단어 제거 및 공백 결합
    return ' '.join([word for word in nouns if len(word) > 1])

def main():
    log(f"--- 고성능 학습 시작: {pd.Timestamp.now()} ---")
    log(f"작업 디렉토리: {script_dir}")

    config_file = os.path.join(script_dir, "features.json")
    if not os.path.exists(config_file):
        log(f"에러: 설정 파일이 없습니다: {config_file}")
        raise FileNotFoundError(f"설정 파일이 없습니다: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    input_columns = config.get('input_columns')
    target_column = config.get('target_column')
    
    file_config = config.get('files', {})
    csv_file_name = file_config.get('train_data', config.get('csv_file', 'train_data.csv'))
    csv_file = os.path.join(script_dir, csv_file_name)

    if not input_columns or not target_column:
        raise ValueError("features.json에 input_columns 또는 target_column 설정이 누락되었습니다.")

    log(f"설정: input_columns={input_columns}, target_column={target_column}")
    log(f"학습 데이터 경로: {csv_file}")

    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='cp949')

    log(f"총 {len(df)}개 행 로드")

    # 정답 값이 없는 행 삭제
    initial_len = len(df)
    df = df.dropna(subset=[target_column])
    dropped_len = initial_len - len(df)
    if dropped_len > 0:
        log(f"정답 값이 없는 {dropped_len}개 행을 제거했습니다.")

    # 텍스트 전처리 및 명사 추출 (Okt 사용)
    log("한국어 형태소 분석 및 전처리 중 (시간이 소요될 수 있습니다)...")
    okt = Okt()
    
    def combine_and_preprocess(row):
        combined_text = ""
        for col in input_columns:
            if col in row and pd.notna(row[col]):
                combined_text += " " + str(row[col])
        return preprocess_korean(combined_text, okt)

    df['processed_text'] = df.apply(combine_and_preprocess, axis=1)
    
    X = df['processed_text']
    y = df[target_column]

    log("데이터 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log("TF-IDF 벡터화 중...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    log("LightGBM 모델 훈련 중...")
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        importance_type='gain'
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    log("\n=== 고성능 모델 성능 ===")
    log(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
    log("\n분류 리포트:")
    log(classification_report(y_test, y_pred))

    log("모델 저장 중...")
    model_path = os.path.join(script_dir, file_config.get('model', 'request_model.pkl'))
    vectorizer_path = os.path.join(script_dir, file_config.get('vectorizer', 'request_vectorizer.pkl'))

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    log("\n고성능 모델과 벡터라이저가 저장되었습니다.")
    log(f"--- 학습 완료: {pd.Timestamp.now()} ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("\n!!! 오류 발생 !!!")
        log(traceback.format_exc())
    finally:
        print("\n" + "="*50)
        input("프로그램을 종료하려면 Enter 키를 누르세요...")
