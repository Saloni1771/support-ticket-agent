# Support Ticket AI Agent

## 📌 Overview
This project implements a terminal-based AI agent that processes support tickets, retrieves relevant information from a provided support corpus, and generates responses or escalates cases when necessary.

The system strictly uses the given support corpus and avoids generating unsupported or hallucinated responses.

---

## ⚙️ Features

- Request Type Classification (bug, feature_request, product_issue, invalid)
- Product Area Detection (Payments, Account, Assessment, Fraud, API, etc.)
- Information Retrieval using TF-IDF + Cosine Similarity
- Automated Response Generation from support corpus
- Escalation for high-risk or low-confidence cases
- Logging of all interactions (chat transcript)
- Confidence score for each response

---

## 🧠 Approach

### 1. Preprocessing
- Lowercasing
- Text normalization (e.g., "sign in" → "login")
- Removing special characters

### 2. Retrieval
- TF-IDF vectorization with n-grams (1,2)
- Cosine similarity to find best matching document

### 3. Classification
- Rule-based keyword matching for:
  - Request Type
  - Product Area

### 4. Escalation Logic
Cases are escalated if:
- High-risk keywords detected (fraud, stolen, unauthorized, etc.)
- Low similarity score (< 0.15)
- Query is unsupported by corpus

---

## 📂 Project Structure
