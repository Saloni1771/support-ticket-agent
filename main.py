import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===== LOAD DATA =====
tickets = pd.read_csv("data/support_tickets.csv")

# Load corpus (each doc separated by blank line)
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    raw_docs = [doc.strip() for doc in f.read().split("\n\n") if doc.strip()]


# ===== UTILITY FUNCTIONS =====
def preprocess(text):
    text = str(text).lower()

    # normalization (VERY IMPORTANT)
    replacements = {
        "can't": "cannot",
        "cant": "cannot",
        "sign in": "login",
        "log in": "login",
        "not working": "error",
        "failed": "fail",
        "submission not working": "submission error",
        "mock interview": "interview",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


# Preprocess corpus separately (DO NOT overwrite raw_docs)
processed_docs = [preprocess(doc) for doc in raw_docs]


# ===== BUILD TF-IDF VECTORIZER =====
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

doc_vectors = vectorizer.fit_transform(processed_docs)


# ===== CLASSIFICATION FUNCTIONS =====
def get_request_type(text):
    text = text.lower()

    if any(k in text for k in ["error", "bug", "not working", "fail"]):
        return "bug"
    elif any(k in text for k in ["feature", "add", "request"]):
        return "feature_request"
    elif any(k in text for k in ["how", "help", "issue", "can't", "unable"]):
        return "product_issue"
    else:
        return "invalid"


def get_product_area(text):
    text = text.lower()

    if any(k in text for k in ["payment", "refund", "bill", "visa"]):
        return "Payments"
    elif any(k in text for k in ["login", "password", "account"]):
        return "Account"
    elif any(k in text for k in ["assessment", "test", "exam", "interview"]):
        return "Assessment"
    elif any(k in text for k in ["fraud", "unauthorized", "stolen"]):
        return "Fraud"
    elif any(k in text for k in ["api", "integration"]):
        return "API"
    elif "claude" in text:
        return "Claude"
    elif "hackerrank" in text:
        return "HackerRank"
    else:
        return "General"


# ===== ESCALATION LOGIC =====
def should_escalate(text, similarity_score):
    text = text.lower()

    high_risk_keywords = [
        "fraud", "unauthorized", "stolen",
        "payment failed", "refund issue",
        "account locked", "suspended",
        "security", "legal", "urgent"
    ]

    # High-risk always escalate
    if any(word in text for word in high_risk_keywords):
        return True

    # Low confidence escalate (tuned)
    if similarity_score < 0.15:
        return True

    return False


# ===== RETRIEVAL FUNCTION =====
def retrieve_doc(query):
    query_clean = preprocess(query)
    query_vec = vectorizer.transform([query_clean])

    similarity = cosine_similarity(query_vec, doc_vectors)

    best_idx = similarity.argmax()
    best_score = similarity[0][best_idx]

    # Debug (optional)
    # print("QUERY:", query_clean)
    # print("SCORE:", best_score)
    # print("MATCH:", raw_docs[best_idx][:100])

    return raw_docs[best_idx], best_score


# ===== RESPONSE GENERATION =====
def generate_response(doc):
    return f"""
Based on your query, here is relevant information:

{doc[:400]}

If this does not resolve your issue, please let us know.
"""


def escalate_response():
    return "Your issue requires further review. We are escalating this to our support team."


# ===== JUSTIFICATION =====
def get_justification(status, score):
    if status == "escalated":
        return f"Escalated due to high risk or low confidence (score={score:.2f})"
    else:
        return f"Replied using support documentation (score={score:.2f})"


# ===== MAIN LOOP =====
results = []

for _, row in tickets.iterrows():
    issue = str(row.get('Issue', ''))
    subject = str(row.get('Subject', ''))

    full_text = issue + " " + subject

    # Classification
    request_type = get_request_type(full_text)
    product_area = get_product_area(full_text)

    # Retrieval
    doc, score = retrieve_doc(full_text)

    # Escalation
    escalate = should_escalate(full_text, score)
    status = "escalated" if escalate else "replied"

    # Response
    if escalate:
        response = escalate_response()
    else:
        response = generate_response(doc)

    # Justification
    justification = get_justification(status, score)

    # Confidence
    confidence = round(score, 3)

    # Save result
    results.append({
        "status": status,
        "product_area": product_area,
        "response": response.strip(),
        "justification": justification,
        "request_type": request_type,
        "confidence": confidence
    })

    # Logging
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"{issue} | {status} | {confidence} | {response[:80]}\n")


# ===== SAVE OUTPUT =====
output_df = pd.DataFrame(results)
output_df.to_csv("output.csv", index=False)

print("Done! Output saved to output.csv")
print("Saved at:", os.path.abspath("output.csv"))
