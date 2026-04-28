import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# 🎨 CUSTOM UI CSS
# -------------------------------
st.markdown("""
<style>

/* App background */
body {
    background-color: #0e1117;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
}

/* Cards */
.card {
    padding: 18px;
    border-radius: 12px;
    background: #1e293b;
    color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    margin-bottom: 12px;
}

/* Sentiment colors */
.positive {
    color: #22c55e;
    font-weight: bold;
}
.negative {
    color: #ef4444;
    font-weight: bold;
}
.neutral {
    color: #f59e0b;
    font-weight: bold;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

</style>
""", unsafe_allow_html=True)
# -------------------------------
# 🤖 LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_model()

# -------------------------------
# 🧠 AGENTS
# -------------------------------
def sentiment_agent(state):
    text = state.get("text", "")
    result = sentiment_model(text)[0]["label"]

    if result == "POSITIVE":
        sentiment = "Positive"
    elif result == "NEGATIVE":
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {**state, "sentiment": sentiment}


def topic_agent(state):
    text = state.get("text", "").lower()

    if "hostel" in text:
        topic = "Hostel"
    elif "exam" in text:
        topic = "Exams"
    elif "placement" in text:
        topic = "Placement"
    elif "teacher" in text or "teaching" in text:
        topic = "Teaching"
    elif "lab" in text or "facility" in text:
        topic = "Facilities"
    else:
        topic = "General"

    return {**state, "topic": topic}


def suggestion_agent(state):
    sentiment = state.get("sentiment")
    topic = state.get("topic")

    if sentiment == "Negative":
        suggestions = {
            "Hostel": "Improve food quality and hygiene",
            "Exams": "Make exam schedule clearer",
            "Placement": "Provide better training",
            "Teaching": "Improve teaching methods",
            "Facilities": "Upgrade infrastructure"
        }
        suggestion = suggestions.get(topic, "Take corrective action")
    else:
        suggestion = "No action needed"

    return {**state, "suggestion": suggestion}


def summary_agent(state):
    text = state.get("text", "")
    return {**state, "summary": text[:60] + "..." if len(text) > 60 else text}


def run_pipeline(text):
    state = {"text": text}
    state = sentiment_agent(state)
    state = topic_agent(state)
    state = suggestion_agent(state)
    state = summary_agent(state)
    return state

# -------------------------------
# 📄 PDF GENERATOR
# -------------------------------
def generate_pdf(df, file_path="feedback_report.pdf"):
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Student Feedback Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    total = len(df)
    pos = len(df[df["Sentiment"] == "Positive"])
    neg = len(df[df["Sentiment"] == "Negative"])
    neu = len(df[df["Sentiment"] == "Neutral"])

    summary = f"""
    Total Feedback: {total}<br/>
    Positive: {pos}<br/>
    Negative: {neg}<br/>
    Neutral: {neu}
    """

    elements.append(Paragraph(summary, styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Feedback", "Sentiment", "Topic", "Suggestion"]]

    for _, row in df.iterrows():
        table_data.append([
            str(row["Feedback"])[:50],
            row["Sentiment"],
            row["Topic"],
            str(row["Suggestion"])[:50]
        ])

    table = Table(table_data)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    elements.append(table)
    doc.build(elements)

    return file_path

# -------------------------------
# 🌐 UI START
# -------------------------------
st.set_page_config(layout="wide")

st.markdown('<div class="title">🎓 Student Feedback Analytics</div>', unsafe_allow_html=True)
st.write("AI-powered system for analyzing student feedback and generating insights.")

# -------------------------------
# ✍️ SINGLE INPUT
# -------------------------------
st.subheader("✍️ Analyze Feedback")

user_input = st.text_input("Enter feedback", placeholder="Type student feedback here...")

if st.button("Analyze"):
    if user_input.strip():
        result = run_pipeline(user_input)
        sentiment = result["sentiment"]

        if sentiment == "Positive":
            color_class = "positive"
        elif sentiment == "Negative":
            color_class = "negative"
        else:
            color_class = "neutral"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f'<div class="card">Sentiment: <span class="{color_class}">{sentiment}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card">Topic: {result["topic"]}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="card">Suggestion: {result["suggestion"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card">Summary: {result["summary"]}</div>', unsafe_allow_html=True)

    else:
        st.error("Enter feedback")

# -------------------------------
# 📂 CSV UPLOAD
# -------------------------------
st.divider()
st.subheader("📂 Upload CSV for Bulk Analysis")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    df.columns = df.columns.str.strip().str.lower()
    st.write("Detected columns:", df.columns.tolist())

    if "feedback" not in df.columns:
        st.error("CSV must contain 'feedback' column")
        st.stop()

    results = []

    for text in df["feedback"]:
        if pd.isna(text):
            continue

        res = run_pipeline(str(text))

        results.append({
            "Feedback": text,
            "Sentiment": res["sentiment"],
            "Topic": res["topic"],
            "Suggestion": res["suggestion"],
            "Summary": res["summary"]
        })

    result_df = pd.DataFrame(results)

    st.success("CSV Analysis Done")
    st.dataframe(result_df)

    # -------------------------------
    # 📊 CHART
    # -------------------------------
    st.subheader("📊 Sentiment Distribution")

    counts = result_df["Sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(counts.index, counts.values)
    ax.set_title("Sentiment Overview")

    st.pyplot(fig)

    # -------------------------------
    # 📄 PDF
    # -------------------------------
    st.subheader("📄 Report Generation")

    if st.button("Generate PDF Report"):
        pdf_path = generate_pdf(result_df)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇ Download Report",
                data=f,
                file_name="feedback_report.pdf",
                mime="application/pdf"
            )
