import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import pandas as pd

def load_and_preprocess_data():
    df = pd.read_csv("csvdata.csv")
    df = df.drop(["first_funding_at", "last_funding_at"], axis=1)
    df = df.drop(columns=df.columns[0])

    industry_dummies_df = pd.get_dummies(df["industry"], drop_first=True)
    country_dummies_df = pd.get_dummies(df["country_code"], drop_first=True)
    region_dummies_df = pd.get_dummies(df["region"], drop_first=True)

    X_col_num = ['homepage_url', "funding_total_usd", 'funding_rounds', 'founded_quarter']
    X_df_num = df[X_col_num]
    X_df = X_df_num.merge(industry_dummies_df, left_index=True, right_index=True).merge(country_dummies_df, left_index=True, right_index=True).merge(region_dummies_df, left_index=True, right_index=True)
    X_df['intercept'] = 1
    y_df = df.label

    return X_df, y_df, df

def load_rf_model(X_df, y_df):
    scaler = StandardScaler()
    X_temp_df, _, y_temp_df, _ = train_test_split(X_df, y_df, test_size=0.1, random_state=40, stratify=y_df)
    X_train_df, _, y_train_df, _ = train_test_split(X_temp_df, y_temp_df, test_size=1/9, random_state=40, stratify=y_temp_df)

    X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train_df.values), columns=X_df.columns)
    model_path = "random_forest_model.joblib"

    if os.path.exists(model_path):
        rf = load(model_path)
    else:
        rf = RandomForestClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=1234, n_jobs=-1)
        rf.fit(X_train_scaled_df, y_train_df)
        dump(rf, model_path)

    return rf, scaler

def get_user_input(df):
    # 사용자에게 홈페이지 URL 유무 체크박스 제공
    has_homepage_url = st.checkbox("**Has Homepage URL?**")
    # 체크박스의 선택 여부에 따라 homepage_url 값 설정
    homepage_url = 1 if has_homepage_url else 0
    funding_total_usd = st.number_input("Total Funding in USD", 0)
    funding_rounds = st.number_input("Number of Funding Rounds", 0)
    founded_quarter = st.number_input("Founded Quarter", 1)

    # 'Unknown'과 '0_other_cat'을 제외한 산업 분야 목록 표시
    industry_options_display = [industry for industry in df["industry"].unique() if industry != "Unknown" and industry != "0_other_cat"]
    industry_options_display.append("Others")

    selected_industry = st.selectbox("Industry", industry_options_display)

    if selected_industry == "Others":
        selected_industry = "0_other_cat"

    country_code = st.selectbox("Country Code", df["country_code"].unique())
    region = st.selectbox("Region", df["region"].unique())

    user_data = {
        'homepage_url': [homepage_url],
        'funding_total_usd': [funding_total_usd],
        'funding_rounds': [funding_rounds],
        'founded_quarter': [founded_quarter],
        'industry': [selected_industry],
        'country_code': [country_code],
        'region': [region]
    }

    return pd.DataFrame(user_data)


def preprocess_user_input(user_input, df, X_df, scaler):
    # Dummy variable creation for user input
    industry_dummies = pd.get_dummies(user_input["industry"], prefix='industry')
    country_dummies = pd.get_dummies(user_input["country_code"], prefix='country_code')
    region_dummies = pd.get_dummies(user_input["region"], prefix='region')

    # Concatenate the user input numerical data with the dummies
    user_input_dummies = pd.concat([user_input[['homepage_url', "funding_total_usd", 'funding_rounds', 'founded_quarter']],
                                    industry_dummies, country_dummies, region_dummies], axis=1)

    # Ensure that all columns from the original data are present in the user input
    # For missing columns, fill with zeros
    for column in X_df.columns:
        if column not in user_input_dummies.columns:
            user_input_dummies[column] = 0

    # Order columns to match the original structure
    user_input_dummies = user_input_dummies[X_df.columns]
    user_input_dummies_scaled = scaler.transform(user_input_dummies)

    return user_input_dummies_scaled

# 라벨링 점수 계산을 위한 함수 정의
def calculate_score(row, class_probabilities):
    scores = {
        1: [0, 20],
        2: [20, 40],
        3: [40, 60],
        4: [60, 80],
        5: [80, 100]
    }

    label = row["label"]
    score_range = scores[label]
    min_score = score_range[0]
    max_score_diff = score_range[1] - score_range[0]

    # 확률에 따라서 점수 계산
    if label in [1, 2]:
        return min_score + max_score_diff * (1 - class_probabilities[label - 1])
    else:
        return min_score + max_score_diff * class_probabilities[label - 1]


def main():
    st.title("Startup Credit Scoring with RandomForest")

    X_df, y_df, df = load_and_preprocess_data()
    rf, scaler = load_rf_model(X_df, y_df)
    user_input = get_user_input(df)

    processed_data = preprocess_user_input(user_input,df, X_df, scaler)
    
    # 사용자가 입력한 스타트업 정보도 앱에 표시한다.
    st.write("Startup Details:")
    st.table(user_input)
    
    # 버튼 추가
    if st.button("Calculate Credit Score"):
        # Class probabilities를 얻는다.
        class_probabilities = rf.predict_proba(processed_data)
        predicted_class = rf.predict(processed_data)[0]

        # 점수 계산을 위해 DataFrame에 사용자 입력과 클래스 확률을 추가한다.
        df_input = pd.DataFrame(processed_data, columns=X_df.columns)
        df_input["label"] = predicted_class
        df_input["class_probabilities"] = list(class_probabilities)
        
        # 점수 계산
        df_input["score"] = df_input.apply(lambda row: calculate_score(row, row["class_probabilities"]), axis=1)

        st.subheader("Startup's Predicted Credit Score:")
        st.markdown(f"**Predicted Credit Score: {round(df_input['score'].values[0], 2)}**")
        
        st.write("Scoring Breakdown:")
        st.write(df_input[["label", "score", "class_probabilities"]])

if __name__ == "__main__":
    main()
