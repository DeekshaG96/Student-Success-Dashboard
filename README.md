<div align="center">
  <h1>🎓 SIT Global Student Success Hub</h1>
  <p><b>An elite AI platform designed for CSBS students at Srinivas Institute of Technology</b></p>
  
  [![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=streamlit)](https://student-success-dashboard-cpnswmcahqbt6zwqkhwghy.streamlit.app/)
</div>

---

## 🚀 Mission Statement

The SIT Global Student Success Hub is a premier, full-stack intelligence portal tailored exclusively for the Computer Science & Business Systems (CSBS) department at Srinivas Institute of Technology. Built to empower our next generation of engineers, this platform merges predictive analytics, advanced AI mentorship, cognitive resources, and wellness tools into a single, highly secure, and beautifully designed Glassmorphism interface.

## 🛠️ Tech Stack & Automation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini_AI-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" />
</p>

## ✨ Key Features Ecosystem

* **🔒 Enterprise-Grade Secure Login**: Master-level authentication architecture implementing zero-leak secrets management with `bcrypt` password hashing.
* **🎯 MLOps Success Predictor**: A predictive intelligence engine leveraging a meticulously trained scikit-learn model, pulled directly from a Hugging Face repository.
* **🔬 Advanced Ensemble Lab**: Interactive model pipeline allowing students to experiment with Voting Classifiers, Bagging, and AdaBoost implementations live in the browser.
* **🤖 Global AI Mentor**: An elite integration of Google's Gemini-2.5-flash Pro API acting as a virtual academic advisor for GATE 2027 preparation and International Higher Education pathways.
* **📚 International Digital Library**: Direct asynchronous integration with the Open Library API to pull free CS journals, research papers, and global textbooks.
* **🧘 Zen Study Zone**: Cognitive focus portal equipped with Pomodoro timers, interactive exam alarms, and 24/7 ad-free ambient live streams specifically for engineering flow states.
* **🎮 Brain Games Hub**: High-concentration strategic logic games (Interactive Lichess and WebSudoku Logic) designed to keep the analytical mind razor-sharp.
* **🌱 Student Life & Wellness**: A private, downloadable daily engineering journal accompanied by dynamically generated, contextually-aware VTU study hacks and motivation.

## ⚙️ Local Setup & Installation

Follow these steps to set up the SIT Global Hub securely in your local environment.

### 1. Clone the Repository
```bash
git clone https://github.com/GANCHU0909/Student-Success-Dashboard.git
cd Student-Success-Dashboard
```

### 2. Install Project Dependencies
To guarantee compatibility, specifically with our predictive pipelines, install the pinned versions:
```bash
pip install -r requirements.txt
```

### 3. Configure the Secrets
* Create a `.streamlit` folder at the root directory.
* Create a `secrets.toml` file inside the `.streamlit` folder.
* Use the following template, replacing the values with your actual API keys and credentials:

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

## 🛡️ Zero-Leak Security Architecture

**Cyber Security Awareness Notice:** This application strictly adheres to zero-trust compliance standards. 
Absolutely **no secrets, API keys, or passwords** are hardcoded into the source codebase. 
All sensitive variables—including Hugging Face tokens and Gemini keys—are strictly injected at runtime via Streamlit Community Cloud environment variables or the local `.streamlit/secrets.toml` file. The `.gitignore` prevents both `.env` and `secrets.toml` from ever being pushed to the public repository.

---
<p align="center">
  <i>"I would love to change the world, but they won't give me the source code."</i>
</p>
