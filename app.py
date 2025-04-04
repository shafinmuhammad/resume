import spacy
import re

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract structured details
def extract_details(text):
    doc = nlp(text)

    # Initialize fields
    extracted_data = {
        "Name": "Not Found",
        "Professional Summary": "Not Found",
        "Skills": [],
        "Education": [],
        "Work Experience": [],
        "Projects": [],
        "Certificates": [],
        "Languages": [],
        "Additional Details": []
    }

    # Extract name (First detected PERSON entity)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            extracted_data["Name"] = ent.text
            break  # Stop after the first name found

    # Extract education (ORG entities assumed as institutions)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            extracted_data["Education"].append(ent.text)

    # Extract work experience (DATE entities assumed as experience timeline)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            extracted_data["Work Experience"].append(ent.text)

    # Extract languages (direct LANGUAGE entity detection)
    for ent in doc.ents:
        if ent.label_ == "LANGUAGE":
            extracted_data["Languages"].append(ent.text)

    # Extract professional summary (first 2-3 sentences)
    sentences = text.split(".")
    extracted_data["Professional Summary"] = ". ".join(sentences[:3]) if sentences else "Not Found"

    # Extract projects & certificates using regex (based on keywords)
    projects_pattern = re.findall(r"(?i)(Project[s]?:[\s\S]+?)(?:\n\n|\Z)", text)
    extracted_data["Projects"] = projects_pattern[0] if projects_pattern else "Not Found"

    certificates_pattern = re.findall(r"(?i)(Certification[s]?:[\s\S]+?)(?:\n\n|\Z)", text)
    extracted_data["Certificates"] = certificates_pattern[0] if certificates_pattern else "Not Found"

    # Extract skills based on a predefined skill list
    skill_keywords = ["python", "java", "sql", "machine learning", "excel", "communication", "leadership", "AI", "deep learning", "cloud computing"]
    extracted_data["Skills"] = [kw for kw in skill_keywords if kw in text.lower()]

    # Additional details (detects sections with unknown labels)
    additional_sections = re.findall(r"(?i)([A-Za-z\s]+:)\s*([\s\S]+?)(?:\n\n|\Z)", text)
    for section, content in additional_sections:
        if section.strip() not in extracted_data.keys():
            extracted_data["Additional Details"].append({section.strip(): content.strip()})

    return extracted_data
