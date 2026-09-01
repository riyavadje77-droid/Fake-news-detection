import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# =========================
# LOAD DATASET
# =========================
try:
    data = pd.read_csv("news.csv")
except Exception:
    st.error("news.csv file not found. Keep news.csv and app.py in the same folder.")
    st.stop()

data["label"] = data["label"].astype(str).str.upper().str.strip()

# Remove empty rows
data = data.dropna(subset=["text", "label"])

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

data["clean_text"] = data["text"].apply(clean_text)

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

# =========================
# MODELS
# =========================
logistic_model = LogisticRegression(max_iter=1000)
naive_bayes_model = MultinomialNB()
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

logistic_model.fit(X, y)
naive_bayes_model.fit(X, y)
random_forest_model.fit(X, y)

# =========================
# MODEL ACCURACY
# =========================
logistic_accuracy = accuracy_score(
    y, logistic_model.predict(X)
)

naive_accuracy = accuracy_score(
    y, naive_bayes_model.predict(X)
)

random_accuracy = accuracy_score(
    y, random_forest_model.predict(X)
)

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy * 100,
        naive_accuracy * 100,
        random_accuracy * 100
    ]
})

# =========================
# SESSION HISTORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# HEADER
# =========================
st.title("📰 Fake News Detection System")
st.caption("Machine Learning • NLP • Data Analytics")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 News Prediction",
        "📊 Analytics Dashboard",
        "🤖 Model Comparison",
        "📋 Prediction History",
        "📁 Dataset",
        "📥 Download Report"
    ]
)

# ==========================================================
# 1. NEWS PREDICTION
# ==========================================================

if page == "🏠 News Prediction":

    st.header("🔍 Fake News Checker")

    st.write(
        "Enter a news headline or short article below. "
        "The machine learning model will classify it as Real or Fake."
    )

    news_input = st.text_area(
        "📝 Enter News:",
        height=180,
        placeholder="Paste news headline or article here..."
    )

    if st.button(
        "🔍 Check News",
        use_container_width=True
    ):

        if news_input.strip() == "":
            st.warning("Please enter some news first.")

        else:

            cleaned_input = clean_text(news_input)

            input_vector = vectorizer.transform(
                [cleaned_input]
            )

            prediction = logistic_model.predict(
                input_vector
            )[0]

            probabilities = logistic_model.predict_proba(
                input_vector
            )[0]

            confidence = max(probabilities) * 100

            # Save prediction history
            st.session_state.history.append({
                "News": news_input,
                "Prediction": prediction,
                "Confidence": round(confidence, 2)
            })

            if prediction == "FAKE":
                st.error("🔴 Prediction: FAKE NEWS")
            else:
                st.success("🟢 Prediction: REAL NEWS")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.info(
                "Note: This prediction is based on the trained ML model "
                "and should not be treated as a definitive fact-check."
            )

# ==========================================================
# 2. ANALYTICS DASHBOARD
# ==========================================================

elif page == "📊 Analytics Dashboard":

    st.header("📊 Analytics Dashboard")

    total_news = len(data)

    real_news = len(
        data[data["label"] == "REAL"]
    )

    fake_news = len(
        data[data["label"] == "FAKE"]
    )

    best_accuracy = model_results["Accuracy"].max()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📰 Total News",
        total_news
    )

    col2.metric(
        "🟢 Real News",
        real_news
    )

    col3.metric(
        "🔴 Fake News",
        fake_news
    )

    col4.metric(
        "🏆 Best Accuracy",
        f"{best_accuracy:.2f}%"
    )

    st.markdown("---")

    st.subheader("📈 Real vs Fake News")

    distribution = data["label"].value_counts()

    st.bar_chart(distribution)

    st.subheader("📊 News Percentage")

    percentage = (
        data["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    st.dataframe(
        percentage.rename("Percentage"),
        use_container_width=True
    )

# ==========================================================
# 3. MODEL COMPARISON
# ==========================================================

elif page == "🤖 Model Comparison":

    st.header("🤖 Machine Learning Model Comparison")

    st.write(
        "Three classification algorithms are trained and compared."
    )

    st.dataframe(
        model_results,
        use_container_width=True
    )

    st.subheader("📈 Accuracy Comparison")

    chart = model_results.set_index("Model")

    st.bar_chart(chart["Accuracy"])

    best_model = model_results.loc[
        model_results["Accuracy"].idxmax(),
        "Model"
    ]

    st.success(
        f"🏆 Best performing model: {best_model}"
    )

# ==========================================================
# 4. PREDICTION HISTORY
# ==========================================================

elif page == "📋 Prediction History":

    st.header("📋 Prediction History")

    if len(st.session_state.history) == 0:

        st.info(
            "No predictions yet. Go to News Prediction "
            "and check some news."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.download_button(
            "📥 Download Prediction History",
            history_df.to_csv(index=False),
            "prediction_history.csv",
            "text/csv"
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()

# ==========================================================
# 5. DATASET
# ==========================================================

elif page == "📁 Dataset":

    st.header("📁 Dataset Explorer")

    st.write(
        "News dataset used for training the machine learning models."
    )

    st.dataframe(
        data[["text", "label"]],
        use_container_width=True
    )

    st.subheader("📌 Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Records",
            len(data)
        )

    with col2:
        st.metric(
            "Total Columns",
            len(data.columns)
        )

# ==========================================================
# 6. DOWNLOAD REPORT
# ==========================================================

elif page == "📥 Download Report":

    st.header("📥 Download Analytics Report")

    report = model_results.copy()

    report["Accuracy"] = report["Accuracy"].round(2)

    st.subheader("Model Performance")

    st.dataframe(
        report,
        use_container_width=True
    )

    csv_report = report.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Model Report",
        csv_report,
        "fake_news_model_report.csv",
        "text/csv",
        use_container_width=True
    )

    st.success(
        "Your analytics report is ready to download."
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Fake News Detection | Machine Learning + NLP + Data Analytics"
)