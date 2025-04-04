import streamlit as st
import pickle
import PyPDF2
import re
import spacy

# Load model and NLP
model_path = "resume_classifier.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

nlp = spacy.load("en_core_web_sm")

# Clean text
def clean_text(text):
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.lower().strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text.strip()

# Extract key details using spaCy
def extract_details(text):
    doc = nlp(text)
    name = ""
    skills = []
    education = []
    experience = []
    languages = []

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
        elif ent.label_ == "ORG":
            education.append(ent.text)
        elif ent.label_ == "DATE":
            experience.append(ent.text)
        elif ent.label_ == "LANGUAGE":
            languages.append(ent.text)

    # Basic skill keywords
    keywords = ["python", "java", "sql", "machine learning", "excel", "communication", "leadership"]
    skills = [kw for kw in keywords if kw in text.lower()]

    return {
        "Name": name,
        "Skills": list(set(skills)),
        "Education": list(set(education)),
        "Experience": list(set(experience)),
        "Languages": list(set(languages))
    }

# UI
st.title("🔍 AI Resume Screener")
st.write("Upload a PDF Resume, choose a job category, and get the match result!")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_categories = [
    "Accountant", "Advocate", "Arts", "Automobiles", "Aviation", "Banking", "BPO",
    "Business Development", "Chef", "Construction", "Consultant", "Designer",
    "Digital-Media", "Engineering", "Finance", "Healthcare", "HR",
    "Information-Technology", "Public-Relation", "Sales", "Teacher"
]
selected_category = st.selectbox("💼 Select Job Category", job_categories)

if uploaded_file and selected_category:
    # Extract and clean resume text
    resume_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(resume_text)

    # Predict
    predicted_role = model.predict([cleaned_text])[0]
    confidence = model.predict_proba([cleaned_text]).max()

    # Matching logic
    match_percentage = round(confidence * 100)
    if predicted_role == selected_category:
        match_percentage = max(match_percentage, 85)
    else:
        match_percentage = min(match_percentage, 70)

    # Extract details
    extracted = extract_details(resume_text)

    # Output
    st.subheader("🔍 Extracted Resume Details")
    st.json(extracted)

    st.subheader("📊 Matching Result")
    st.write(f"**Predicted Role:** {predicted_role}")
    st.progress(match_percentage / 100)
    st.success(f"✅ Match Percentage: {match_percentage}%")
