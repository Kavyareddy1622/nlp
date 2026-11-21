import random

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="CollegeBot AI", page_icon="🎓", layout="wide")

# ======================================================
# NLTK SETUP
# ======================================================
for pkg in ["punkt", "punkt_tab", "wordnet", "stopwords"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english")) - {
    "how", "what", "when", "where", "can", "which", "who", "why"
}

# ======================================================
# DATASET
# ======================================================
data = {
    "greeting": [
        "hello", "hi", "hey", "good morning", "good evening", "greetings",
        "howdy", "how are you", "how are you doing", "nice to meet you",
        "hi there", "hello bot",
    ],
    "farewell": ["bye", "goodbye", "see you", "take care", "quit", "exit", "bye bye"],
    "admission": [
        "how to apply", "admission process", "application form",
        "enrollment process", "registration process", "how can i join",
        "admission date", "how can i get admission", "admission enquiry",
        "admissions open",
    ],
    "fees": [
        "fees", "fee structure", "tuition fee", "semester fee", "annual fee",
        "college fee", "payment details", "fee details", "how much are the fees",
        "hostel fees", "course fee",
    ],
    "courses": [
        "courses", "degrees", "available programs", "departments",
        "computer science", "mba", "mca", "what can you do",
        "college information", "tell me about college", "which courses are offered",
    ],
    "hostel": [
        "hostel", "accommodation", "dormitory", "room facility",
        "where can i stay", "girls hostel", "boys hostel", "hostel facility",
        "hostel rooms",
    ],
    "placements": [
        "placement", "salary package", "job opportunities", "campus recruitment",
        "highest package", "average package", "internships", "placement record",
        "companies visiting",
    ],
    "scholarships": [
        "scholarship", "scholarships", "financial aid", "fee waiver",
        "merit scholarship", "how to get a scholarship", "education loan",
    ],
    "thanks": ["thanks", "thank you", "awesome", "great", "helpful", "thanks a lot"],
    "complaint": [
        "bad service", "poor service", "not satisfied", "unhappy", "issue",
        "problem", "disappointed",
    ],
}

responses = {
    "greeting": ["Hello! Welcome to CollegeBot.", "Hi! How can I help you today?"],
    "farewell": ["Goodbye! Have a great day."],
    "admission": ["Admissions are currently open. Please check the college admission portal."],
    "fees": ["Fee details are available on the college website."],
    "courses": ["We offer B.Tech, BCA, MCA, MBA and several other programs."],
    "hostel": ["Separate hostel facilities are available for boys and girls."],
    "placements": ["The college provides placement assistance and internship opportunities."],
    "scholarships": ["Merit and need-based scholarships are available. Please contact the admissions office for eligibility."],
    "thanks": ["You're welcome!", "Glad I could help."],
    "complaint": ["Sorry for the inconvenience. Please explain your issue."],
    "unknown": [
        "I can help with admissions, fees, courses, hostel, placements and scholarships.",
        "Please ask a college-related question.",
    ],
}

# ======================================================
# PREPROCESSING
# ======================================================
def advanced_preprocess(text: str) -> str:
    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha()]
    filtered = [t for t in tokens if t not in STOP_WORDS]
    # If every word was a stopword (e.g. "you"), keep the originals
    tokens = filtered if filtered else tokens
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)


# ======================================================
# TRAIN MODEL (cached so it runs once)
# ======================================================
@st.cache_resource
def load_model():
    X, y = [], []
    for tag, patterns in data.items():
        for p in patterns:
            X.append(advanced_preprocess(p))
            y.append(tag)

    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train on all samples (the dataset is tiny); higher C gives sharper confidence
    model = LogisticRegression(max_iter=1000, C=10)
    model.fit(X_tfidf, y_enc)
    return tfidf, le, model


tfidf, le, model = load_model()


def predict_intent(user_input: str, threshold: float = 0.30):
    processed = advanced_preprocess(user_input)
    if processed.strip() == "":
        return "unknown", 0.0

    vector = tfidf.transform([processed])
    probs = model.predict_proba(vector)[0]
    confidence = float(probs.max())
    tag = le.inverse_transform([probs.argmax()])[0]

    if confidence < threshold:
        return "unknown", confidence
    return tag, confidence


# ======================================================
# STYLING
# ======================================================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center; font-size: 2.8rem; font-weight: 800;
        color: #1f77d0; margin-top: 2rem; margin-bottom: 0;
    }
    .sub-title {
        text-align: center; color: #8b8f99; font-size: 1.1rem; margin-bottom: 2rem;
    }
    .feature-card {
        background: #1e4d36; color: #4ade80; padding: 14px 16px;
        border-radius: 10px; margin-bottom: 12px; font-size: 0.95rem;
    }
    .ask-box {
        background: #1e3350; color: #3b9cff; padding: 16px;
        border-radius: 10px; margin-top: 8px; line-height: 2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("### Project Features")
    for feat in [
        "NLP Text Processing",
        "TF-IDF Feature Extraction",
        "Machine Learning Intent Classification",
        "Confidence Prediction",
    ]:
        st.markdown(f'<div class="feature-card">✓ &nbsp;{feat}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ask-box">
        Ask about:<br>Admission<br>Fees<br>Courses<br>Hostel<br>Placements<br>Scholarships
        </div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================
# MAIN PAGE
# ======================================================
st.markdown('<div class="main-title">CollegeBot AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent College Enquiry Assistant using NLP &amp; Machine Learning</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            st.caption(f"Intent: {msg['intent']} | Confidence: {msg['confidence']:.2f}")

# Input
if prompt := st.chat_input("Ask your college question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    tag, confidence = predict_intent(prompt)
    reply = random.choice(responses[tag])

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "intent": tag, "confidence": confidence}
    )
    with st.chat_message("assistant"):
        st.write(reply)
        st.caption(f"Intent: {tag} | Confidence: {confidence:.2f}")
