<div align="center">
  <h1>🎓 SIT Global Success Hub</h1>
  <p><b>An International MLOps & Wellness Portal for CSBS Students at Srinivas Institute of Technology</b></p>
  
  [![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=streamlit)](https://student-success-dashboard-cpnswmcahqbt6zwqkhwghy.streamlit.app/)
</div>

---

## 🎯 The 'Why': Our Mission
The **SIT Global Success Hub** was engineered to solve the problem of academic fragmentation. Engineering students often juggle multiple platforms for study materials, career guidance, and wellness. This platform unifies predictive AI, generative mentorship, and cognitive wellness tools into one secure, seamless business system designed specifically for the rigorous demands of the CSBS curriculum at Srinivas Institute of Technology.

---

## ✨ Key Features Ecosystem (The 'Big 6')

* **🔒 Secure Auth**: Master-level authentication architecture implementing zero-leak secrets management with `bcrypt`-hashed login and secure session cookies.
* **🔮 ML Outcome Predictor**: A live predictive intelligence engine utilizing a Random Forest Classifier (scikit-learn 1.6.1) to forecast student academic outcomes based on admission metrics.
* **🤖 AI Global Mentor**: Real-time virtual counseling powered by Google Gemini 1.5 Pro, offering specialized advice on GATE 2027 preparation, international MS programs (USA, Germany, Ireland), and core CSBS concepts.
* **🧘 Zen Zone**: A dedicated cognitive focus portal equipped with a customizable Pomodoro timer, live ambient audio streams designed for 'flow state', and interactive exam alarms.
* **🧠 Cognitive Lab**: Embedded brain games including an interactive Chess interface (Lichess) and Sudoku logic puzzles to provide strategic mental breaks.
* **📚 Digital Library**: Instantaneous global textbook and CS journal discovery powered entirely through real-time asynchronous calls to the Open Library API.

---

## 🛠️ Tech Stack & Architecture

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini_AI-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" />
</p>

---

## ⚙️ Local Setup Instructions

Launch the SIT Global Hub in your own development environment in 4 easy steps:

### 1. Clone the Repository
```bash
git clone https://github.com/GANCHU0909/Student-Success-Dashboard.git
cd Student-Success-Dashboard
```

### 2. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure the Secrets (Critical)
Create a `.streamlit` folder at the root directory and add a `secrets.toml` file inside it. Use the template below and replace the placeholder strings with your actual API keys and generated hashes:

```toml
# Template: .streamlit/secrets.toml
HF_TOKEN = "your_hugging_face_token_here"
REPO_ID = "GANCHU0909/Student-Success-Model"
GEMINI_API_KEY = "your_google_gemini_api_key_here"

[credentials]
usernames.student.password = "$2b$12$hashed_bcrypt_password_here"
usernames.student.email = "student@sit.ac.in"
usernames.student.name = "SIT Scholar"

[cookie]
name = "sit_student_portal_auth"
key = "your_random_secure_signature"
expiry_days = 30
```

### 4. Boot the Portal
```bash
streamlit run streamlit_app.py
```

---

## 🛡️ Secret Management & Security

**Cyber Security Awareness Notice:** This project strictly enforces zero-trust compliance. 
**No hardcoded credentials**, API keys, or passwords exist within the source code. All sensitive variables are securely injected at runtime via Streamlit Community Cloud environment variables or the local `.streamlit/secrets.toml` file. The `.gitignore` is strictly configured to exclude sensitive files like `.env` and `secrets.toml` from version control, preventing data leaks and demonstrating elite software engineering practices.

---
<div align="center">
  <b>👨‍🎓 Author: Deeksha G</b> <br>
  <i>Student, Computer Science & Business Systems</i> <br>
  <i>Srinivas Institute of Technology, Mangaluru</i>
</div>
