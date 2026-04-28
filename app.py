import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# LOGIN
import json
import hashlib
import os

# -------------------------------
# 🔐 AUTH SYSTEM
# -------------------------------
USERS_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# -------------------------------
# 🎨 UI STYLE
# -------------------------------
st.markdown("""
<style>
body { background-color: #0e1117; }

.title {
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.card {
    padding: 18px;
    border-radius: 12px;
    background: #1e293b;
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    margin-bottom: 12px;
}

.positive { color: #22c55e; font-weight: bold; }
.negative { color: #ef4444; font-weight: bold; }
.neutral { color: #f59e0b; font-weight: bold; }

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
# 🔐 LOGIN PAGE
# -------------------------------
def login_page():
    st.title("🔐 Login System")

    tab1, tab2 = st.tabs(["Login", "Signup"])
    users = load_users()

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u in users and users[u] == hash_password(p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_u = st.text_input("Create Username")
        new_p = st.text_input("Create Password", type="password")

        if st.button("Signup"):
            if new_u in users:
                st.warning("User exists")
            elif new_u and new_p:
                users[new_u] = hash_password(new_p)
                save_users(users)
                st.success("Account created! Login now")
            else:
                st.error("Enter valid details")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# -------------------------------
# 🤖 MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_model()

# -------------------------------
# 🧠 AI PIPELINE
# -------------------------------
def sentiment_agent(state):
    text = state["text"]
    result = sentiment_model(text)[0]["label"]

    if result == "POSITIVE":
        sentiment = "Positive"
    elif result == "NEGATIVE":
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {**state, "sentiment": sentiment}

def topic_agent(state):
    text = state["text"].lower()

    if "hostel" in text:
        topic = "Hostel"
    elif "exam" in text:
        topic = "Exams"
    elif "placement" in text:
        topic = "Placement"
    elif "teacher" in text or "class" in text:
        topic = "Teaching"
    elif "lab" in text:
        topic = "Facilities"
    else:
        topic = "General"

    return {**state, "topic": topic}

def suggestion_agent(state):
    if state["sentiment"] == "Negative":
        suggestions = {
            "Hostel": "Improve food quality",
            "Exams": "Improve exam planning",
            "Placement": "Provide better training",
            "Teaching": "Improve teaching quality",
            "Facilities": "Upgrade infrastructure"
        }
        suggestion = suggestions.get(state["topic"], "Take corrective action")
    else:
        suggestion = "No action needed"

    return {**state, "suggestion": suggestion}

def summary_agent(state):
    text = state["text"]
    return {**state, "summary": text[:60] + "..." if len(text) > 60 else text}

def run_pipeline(text):
    state = {"text": text}
    state = sentiment_agent(state)
    state = topic_agent(state)
    state = suggestion_agent(state)
    state = summary_agent(state)
    return state

# -------------------------------
# 📄 PDF
# -------------------------------
def generate_pdf(df):
    file = "report.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Student Feedback Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    data = [["Feedback", "Sentiment", "Topic"]]

    for _, row in df.iterrows():
        data.append([
            str(row["Feedback"])[:40],
            row["Sentiment"],
            row["Topic"]
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(table)
    doc.build(elements)

    return file

# -------------------------------
# 🚀 MAIN APP
# -------------------------------
if not st.session_state.logged_in:
    login_page()

else:
    st.sidebar.write(f"👤 {st.session_state.username}")
    if st.sidebar.button("Logout"):
        logout()

    st.markdown('<div class="title">🎓 Student Feedback Analytics</div>', unsafe_allow_html=True)

    # SINGLE INPUT
    st.subheader("✍️ Analyze Feedback")
    text = st.text_input("Enter feedback")

    if st.button("Analyze"):
        if text:
            res = run_pipeline(text)

            col1, col2 = st.columns(2)

            color = "positive" if res["sentiment"]=="Positive" else "negative" if res["sentiment"]=="Negative" else "neutral"

            with col1:
                st.markdown(f'<div class="card">Sentiment: <span class="{color}">{res["sentiment"]}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card">Topic: {res["topic"]}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown(f'<div class="card">Suggestion: {res["suggestion"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card">Summary: {res["summary"]}</div>', unsafe_allow_html=True)

    # CSV
    st.divider()
    st.subheader("📂 Upload CSV")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()

        if "feedback" not in df.columns:
            st.error("CSV must contain 'feedback'")
        else:
            results = []

            for t in df["feedback"]:
                res = run_pipeline(str(t))
                results.append({
                    "Feedback": t,
                    "Sentiment": res["sentiment"],
                    "Topic": res["topic"]
                })

            result_df = pd.DataFrame(results)
            st.dataframe(result_df)

            # Chart
            counts = result_df["Sentiment"].value_counts()
            fig, ax = plt.subplots(figsize=(5,3))
            ax.bar(counts.index, counts.values)
            st.pyplot(fig)

            # PDF
            if st.button("Generate PDF"):
                pdf = generate_pdf(result_df)
                with open(pdf, "rb") as f:
                    st.download_button("Download PDF", f, "report.pdf")
