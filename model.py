import os
import pickle
import re
import PyPDF2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Path to the main data folder
resume_folder = r"C:\Users\PC\OneDrive\Desktop\shafin\Project_ind\data"

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
        return text.strip() if text else None
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

# Load resumes and labels
data = []
labels = []

# Loop through job role folders
for job_role in os.listdir(resume_folder):
    job_path = os.path.join(resume_folder, job_role)

    # Check if it's a directory
    if os.path.isdir(job_path):
        print(f"Processing resumes for: {job_role}")

        for filename in os.listdir(job_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(job_path, filename)
                text = extract_text_from_pdf(file_path)
                if text:
                    data.append(text)
                    labels.append(job_role)  # Use folder name as job role

# Convert to DataFrame
resume_df = pd.DataFrame({'resume_text': data, 'category': labels})

# Check if data is loaded properly
if resume_df.empty:
    print("No resumes found! Check your file paths.")
    exit()

# Clean text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

resume_df['cleaned_resume'] = resume_df['resume_text'].apply(cleanResume)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    resume_df['cleaned_resume'], resume_df['category'], test_size=0.2, random_state=42)

# Create ML pipeline (TF-IDF + Naive Bayes)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Evaluate model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model_path = "resume_classifier.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")

# --- New Section: Parse a PDF and Recommend a Job Role ---
def recommend_job_role(pdf_path, model):
    # Extract text from the new PDF
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text:
        return "Unable to extract text from the provided PDF."

    # Clean the extracted text
    cleaned_resume = cleanResume(resume_text)

    # Predict the job role using the trained model
    predicted_role = model.predict([cleaned_resume])[0]
    confidence = model.predict_proba([cleaned_resume]).max()  # Get the highest probability

    return f"Recommended Job Role: {predicted_role} (Confidence: {confidence:.2f})"

# Example usage: Replace with the path to a new resume PDF
new_resume_path = r"C:\Users\PC\OneDrive\Desktop\shafin\Project_ind\data\ACCOUNTANT\13701259.pdf"
recommendation = recommend_job_role(new_resume_path, model)
print(recommendation)