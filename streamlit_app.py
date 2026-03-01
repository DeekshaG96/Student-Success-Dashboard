import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import time
import os
import joblib
import numpy as np
import pandas as pd
from google import genai
import requests
from huggingface_hub import hf_hub_download
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# --- 1. PAGE CONFIGURATION & CSS ---
def setup_page():
    st.set_page_config(
        page_title="SIT Global Success Hub", 
        page_icon="🎓", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- GLOBAL GLASSMORPHISM & MIDNIGHT CYBER CSS ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* 1. Base Theme - Midnight Cyber */
        .stApp {
            background-color: #0E1117;
            font-family: 'Inter', sans-serif;
            color: #E2E8F0;
        }
        
        /* 2. Glassmorphism for Containers, Forms, and Sidebars */
        [data-testid="stSidebar"] {
            background: rgba(14, 17, 23, 0.7) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        div[data-testid="stForm"], div.stChatFloatingInputContainer {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        }

        /* 3. Hero Section - Animated Gradient */
        .hero-container {
            pointer-events: none;
            background: linear-gradient(-45deg, #00D4FF, #0E1117, #800000, #0E1117);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .hero-container h1 {
            font-weight: 700;
            color: #FFFFFF;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
            margin: 0;
        }
        .hero-container p {
            color: #00D4FF;
            font-weight: 300;
            letter-spacing: 1.5px;
            margin-top: 5px;
            font-size: 1.2rem;
        }

        /* 4. High-Glow Gradient Buttons */
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(90deg, #800000 0%, #00D4FF 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6) !important;
        }

        /* 5. Micro-Interactions (Fade-ins) */
        .stAlert, .stChatMessage {
            animation: fadeIn 0.8s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Hide default Streamlit styling overrides */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)


# --- 2. AUTHENTICATION MODULE ---
def load_secrets():
    """Converts Streamlit secrets to a standard dictionary to prevent mutation errors."""
    def to_dict(secret_obj):
        if hasattr(secret_obj, 'to_dict'):
            return secret_obj.to_dict()
        elif isinstance(secret_obj, dict):
            return {k: to_dict(v) for k, v in secret_obj.items()}
        else:
            return secret_obj
    return to_dict(dict(st.secrets))

def render_login_system(secrets_dict):
    """Handles the SIT Branded Login and Authentication via streamlit-authenticator."""
    if "credentials" not in secrets_dict or "cookie" not in secrets_dict:
        st.error("🚨 Authentication Configuration Missing!")
        st.info("Please ensure you have a properly formatted `.streamlit/secrets.toml` file in your repository root. It must contain the `[credentials]` and `[cookie]` blocks required by `streamlit-authenticator`.")
        st.warning(f"**Diagnostic Info:** Cloud Server currently sees these Secret Keys: `{list(secrets_dict.keys())}`. If this list is empty (`[]`), the Secrets box in the Cloud Settings is empty or failed to save. If you see keys like `'''toml`, remove the markdown backticks from the secrets box!")
        st.stop()

    authenticator = stauth.Authenticate(
        secrets_dict["credentials"],
        secrets_dict["cookie"]["name"],
        secrets_dict["cookie"]["key"],
        secrets_dict["cookie"].get("expiry_days", 30),
        secrets_dict.get("preauthorized", None)
    )

    # SIT Branding
    col1, col2 = st.columns([1, 4])
    with col1:
        if os.path.exists("sit_logo.jpeg"):
            st.image("sit_logo.jpeg", use_container_width=True)
        elif os.path.exists("sit_logo.png"):
            st.image("sit_logo.png", use_container_width=True)
        else:
            st.info("SIT Logo Missing (sit_logo.jpeg or png)")
    with col2:
        st.markdown("""
            <div class="hero-container">
                <h1>SIT Global Hub</h1>
                <p>Department of Computer Science & Business Systems</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    name, authentication_status, username = authenticator.login("main", fields={'Form name': 'Secure Access Portal'})
    
    return authenticator, name, authentication_status


# --- 3. PUBLIC GUEST VIEW ---
def render_public_view(secrets_dict):
    """Renders the Sidebar UI for unauthenticated users."""
    st.info("👋 Welcome to the SIT Global Hub. Please login using **Username:** `student` | **Password:** `SIT_Student_2026`")
    with st.sidebar:
        st.header("Public View")
        st.subheader("📅 2026 Academic Calendar")
        st.markdown(
            "- **March 05:** Tech Fest Inauguration\n"
            "- **April 15:** Internal Assessments Phase II\n"
            "- **May 20:** Even Semester Ends\n"
            "- **June 05:** VTU Practical Exams"
        )
        st.divider()
        st.subheader("🤖 Guest AI Assistant")
        st.caption("Ask general questions about SIT admissions or campus facilities.")
        guest_prompt = st.chat_input("E.g., What are the library hours?")
        if guest_prompt:
            API_KEY = secrets_dict.get("GEMINI_API_KEY")
            if not API_KEY:
                st.warning("Guest AI is currently offline.")
            else:
                try:
                    client = genai.Client(api_key=API_KEY)
                    with st.spinner("AI is thinking..."):
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=guest_prompt,
                            config=genai.types.GenerateContentConfig(
                                system_instruction="You are a concise, helpful guest assistant for Srinivas Institute of Technology (SIT)."
                            )
                        )
                        st.info(response.text)
                except Exception as e:
                    st.error(f"AI Assistant is currently unavailable. Error System details below:")
                    st.exception(e)


# --- 4. SUCCESS PREDICTOR MODULE ---
def render_ml_predictor():
    """Renders the MLOps Success Predictor Tab."""
    st.header("🎯 MLOps Success Predictor")
    st.write("Predict academic outcomes using models configured via Hugging Face Hub.")
    
    col_ml1, col_ml2 = st.columns([1, 1])
    
    with col_ml1:
        with st.form("ml_form"):
            st.subheader("Student Metrics Data")
            prev_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=130.0, step=1.0)
            age = st.number_input("Age at Enrollment", min_value=15, max_value=60, value=18, step=1)
            submitted = st.form_submit_button("Predict Outcome")
            
    with col_ml2:
        st.subheader("Model Inference")
        if submitted:
            with st.spinner("Loading model and predicting outcome..."):
                try:
                    if not os.path.exists("model.joblib"):
                        raise FileNotFoundError("model.joblib not found in the root directory.")
                    
                    model = joblib.load("model.joblib")
                    input_data = np.array([[prev_grade, age]])
                    prediction = model.predict(input_data)
                    prediction_text = str(prediction[0]).title()
                    
                    if prediction_text == "Graduate":
                        st.success(f"Predictive Outcome: The student is strongly projected to **{prediction_text}**.")
                        st.balloons()
                    elif prediction_text == "Dropout":
                        st.error(f"Predictive Outcome: Alert - The student is flagged as a potential **{prediction_text}** risk.")
                    else:
                        st.info(f"Predictive Outcome: The student status is **{prediction_text}**.")
                        
                except Exception as e:
                    st.error(f"Error loading model or making prediction: {str(e)}")
                    st.info("Please ensure that model.joblib is placed in the project root and is compatible with scikit-learn==1.6.1.")


# --- 5. ENSEMBLE LAB MODULE ---
def render_ensemble_lab():
    """Renders the Advanced Ensemble Lab Tab."""
    st.header("Advanced Ensemble Lab")
    st.write("Experiment with interactive model selection and ensemble techniques.")
    
    if "ensemble_models" not in st.session_state:
        st.session_state.ensemble_models = {}
        st.session_state.scaler = None
    
    col_e1, col_e2 = st.columns([1, 2])
    
    with col_e1:
        st.subheader("Model Configuration")
        ensemble_choice = st.selectbox("Select Ensemble Method:", ["Voting Classifier", "Bagging/Pasting", "Boosting (AdaBoost)"])
        if ensemble_choice == "Bagging/Pasting":
            is_bootstrap = st.toggle("Use Bagging (Bootstrap=True)", value=True, help="Toggle OFF to use Pasting")
            
        if st.button("Train Models"):
            with st.spinner("Training ensemble models..."):
                try:
                    if os.path.exists("data.csv"):
                         df = pd.read_csv("data.csv")
                         if 'Previous Qualification Grade' in df.columns and 'Age at Enrollment' in df.columns and 'Target' in df.columns:
                             X = df[['Previous Qualification Grade', 'Age at Enrollment']]
                             y = df['Target']
                         else:
                             raise ValueError("Format of data.csv doesn't match expected columns. Falling back to synthetic.")
                    else:
                         np.random.seed(42)
                         n_samples = 500
                         grades = np.random.normal(130, 20, n_samples)
                         ages = np.random.normal(25, 5, n_samples)
                         X = pd.DataFrame({'Previous Qualification Grade': grades, 'Age at Enrollment': ages})
                         noise = np.random.normal(0, 10, n_samples)
                         y = ((grades - (ages * 2) + noise) > 80).astype(int) 

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # Training Algorithms
                    clf1 = LogisticRegression(random_state=42)
                    clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
                    clf3 = SVC(probability=True, random_state=42)
                    
                    voting_clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='soft')
                    voting_clf.fit(X_scaled, y)
                    st.session_state.ensemble_models["Voting Classifier"] = voting_clf
                    
                    bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=50, max_samples=100, bootstrap=True, random_state=42)
                    bag_clf.fit(X_scaled, y)
                    st.session_state.ensemble_models["Bagging Model"] = bag_clf
                    
                    paste_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=50, max_samples=100, bootstrap=False, random_state=42)
                    paste_clf.fit(X_scaled, y)
                    st.session_state.ensemble_models["Pasting Model"] = paste_clf
                    
                    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=50, algorithm="SAMME", random_state=42)
                    ada_clf.fit(X_scaled, y)
                    st.session_state.ensemble_models["Boosting (AdaBoost)"] = ada_clf
                    
                    st.success("Successfully fitted all ensemble models!")
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    
    with col_e2:
        st.subheader("Dynamic Prediction")
        dyn_grade = st.slider("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=130.0, step=1.0)
        dyn_age = st.slider("Age at Enrollment", min_value=15, max_value=60, value=18, step=1)
        
        if st.button("Generate Ensemble Prediction"):
            if not st.session_state.ensemble_models:
                st.warning("Please click 'Train Models' first to initialize the estimators.")
            else:
                input_df = pd.DataFrame([[dyn_grade, dyn_age]], columns=['Previous Qualification Grade', 'Age at Enrollment'])
                scaled_input = st.session_state.scaler.transform(input_df)
                
                model_key = ensemble_choice
                if ensemble_choice == "Bagging/Pasting":
                    model_key = "Bagging Model" if is_bootstrap else "Pasting Model"
                    
                selected_model = st.session_state.ensemble_models.get(model_key)
                
                try:
                    pred = selected_model.predict(scaled_input)
                    pred_label = "Graduate" if pred[0] == 1 else "Dropout/Enrolled"
                    st.info(f"Using **{model_key}** Engine")
                    
                    if pred[0] == 1:
                        st.success(f"Ensemble Output: The model strongly predicts **{pred_label}**.")
                        st.balloons()
                    else:
                        st.error(f"Ensemble Output: The model warns of potential **{pred_label}** risk.")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")


# --- 6. AI MENTOR MODULE ---
def render_ai_mentor(secrets_dict):
    """Renders the Gemini Pro AI Mentor Tab."""
    st.header("Global AI Mentor")
    st.write("Consult the Gemini Pro expert on GATE 2027, International Higher Ed, and Advanced CSBS Subjects.")
    
    API_KEY = secrets_dict.get("GEMINI_API_KEY")
    if not API_KEY:
        st.error("GEMINI_API_KEY missing from st.secrets. Mentor is offline.")
    else:
        try:
            client = genai.Client(api_key=API_KEY)
            sys_instruct = (
                "You are an elite Principal Global AI Mentor for CSBS students at Srinivas Institute of Technology. "
                "Act as an absolute expert on: "
                "1. GATE 2027 preparation strategies and syllabus. "
                "2. International higher education pathways (specifically USA, Germany, Ireland). "
                "3. Advanced CSBS subjects like Reinforcement Learning, MLOps, and Deep Learning."
            )
            
            if "ai_messages" not in st.session_state:
                st.session_state.ai_messages = []
                
            for msg in st.session_state.ai_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
            prompt = st.chat_input("Ask about GATE 2027, studying in Germany, or RL algorithms...")
            if prompt:
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Mentor is analyzing request..."):
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                            config=genai.types.GenerateContentConfig(system_instruction=sys_instruct)
                        )
                        st.markdown(response.text)
                st.session_state.ai_messages.append({"role": "assistant", "content": response.text})
                
        except Exception as e:
            st.error(f"Error communicating with Gemini API: {str(e)}")


# --- 7. WELLNESS & ZEN MODULE ---
def render_wellness_zen():
    """Renders the Zen Wellness & Focus Radio Tab."""
    st.header("Zen Wellness & Alarm")
    st.write("Enhance focus using the Pomodoro meditation timer and interactive exam alarms.")
    
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        st.subheader("🍅 Pomodoro Timer")
        pomodoro_minutes = st.number_input("Set focus duration (minutes):", min_value=1, max_value=120, value=25, key="pomo")
        if st.button("Start Timer"):
            timer_ph = st.empty()
            total_sec = int(pomodoro_minutes * 60)
            progress = st.progress(0)
            
            for i in range(total_sec, -1, -1):
                m, s = divmod(i, 60)
                timer_ph.markdown(f"<h1 style='text-align: center; color: #ff6347;'>{m:02d}:{s:02d}</h1>", unsafe_allow_html=True)
                progress.progress(1.0 - (i / total_sec))
                time.sleep(1)
                
            st.success("Session complete! Time for a mindful break.")
            st.balloons()
            
    with col_z2:
        st.subheader("⏰ Exam Alarm")
        exam_time = st.time_input("Set focus alarm for next study session:")
        if st.button("Set Alarm"):
            st.info(f"Exam Alarm successfully set for {exam_time.strftime('%I:%M %p')}. Stay concentrated!")
            
    st.divider()
    st.subheader("🎧 24/7 Zen Focus Radio")
    st.write("Live, ad-free ambient streams to keep you in the flow state.")
    sound_choice = st.selectbox("Select soundscape:", [
        "Deep Focus (Groove Salad)", 
        "Coding Hacker (Def Con)", 
        "Deep Space Ambient", 
        "Zen Meditation (Drone Zone)"
    ])
    
    ambient_maps = {
        "Deep Focus (Groove Salad)": "https://ice1.somafm.com/groovesalad-128-mp3",
        "Coding Hacker (Def Con)": "https://ice1.somafm.com/defcon-128-mp3",
        "Deep Space Ambient": "https://ice1.somafm.com/deepspaceone-128-mp3",
        "Zen Meditation (Drone Zone)": "https://ice1.somafm.com/dronezone-128-mp3"
    }
    
    selected_url = ambient_maps.get(sound_choice)
    st.audio(selected_url, format="audio/mp3")
    st.caption(f"Currently streaming: {sound_choice} via SomaFM (No YouTube Blocking Issues).")


# --- 8. BRAIN GAMES MODULE ---
def render_brain_games():
    """Renders the Cognitive Brain Games Tab."""
    st.header("Brain Games Hub")
    st.write("Take a cognitive break to sharpen your mind with logic games.")
    
    game_choice = st.radio("Choose a game strategy:", ["♟️ Chess (Interactive Window)", "🧩 Sudoku Logic Generator"], horizontal=True)
    
    if "Chess" in game_choice:
        st.subheader("Interactive Chess")
        st.write("Train against the computer using a lightweight iframe instance.")
        components.iframe("https://lichess.org/training/frame?theme=brown&bg=dark", height=450)
        
    elif "Sudoku" in game_choice:
        st.subheader("Interactive Sudoku Logic")
        st.write("Challenge your working memory with a live WebSudoku instance.")
        components.iframe("https://websudoku.com/", height=550)


# --- 9. GLOBAL LIBRARY MODULE ---
def render_global_library():
    """Renders the Open Library API Search Tab."""
    st.header("📚 Global Digital Library")
    st.write("Free search for textbooks, CS journals, and global resources via the Open Library API.")
    
    col_lib1, col_lib2 = st.columns([1, 2])
    
    with col_lib1:
        search_query = st.text_input("Enter a subject (e.g., 'Reinforcement Learning', 'Germany')")
        search_btn = st.button("Search Open Library")
        
    with col_lib2:
        if search_btn:
            if search_query.strip() == "":
                st.warning("Please enter a valid search term.")
            elif len(search_query.strip()) < 3:
                st.warning("Your search query is too short (minimum 3 characters).")
            else:
                with st.spinner(f"Querying Open Library for '{search_query}'..."):
                    try:
                        clean_query = search_query.replace(' ', '+')
                        response = requests.get(f"https://openlibrary.org/search.json?q={clean_query}&limit=6", timeout=12)
                        
                        if response.status_code == 200:
                            data = response.json()
                            docs = data.get("docs", [])
                            
                            if not docs:
                                st.info("No matching books found. Try broadening your terms.")
                            else:
                                st.success(f"Retrieved the top {len(docs)} highly relevant results!")
                                for doc in docs:
                                    title = doc.get("title", "Unknown Subject")
                                    authors = ", ".join(doc.get("author_name", ["Unknown Editor"]))
                                    year = doc.get("first_publish_year", "Unknown Year")
                                    preview = doc.get("key", "")
                                    
                                    with st.container(border=True):
                                        st.markdown(f"**📚 {title}** *(Published: {year})*")
                                        st.markdown(f"**Authors:** {authors}")
                                        if preview:
                                            st.markdown(f"[Read Resource on Open Library](https://openlibrary.org{preview})")
                        else:
                            st.error(f"Open Library API responsed with Status Code {response.status_code}")
                    except Exception as e:
                        st.error(f"Failed to connect to Open Library API: {str(e)}")


# --- 10. DAILY LIFE & GROWTH MODULE ---
def render_daily_growth(secrets_dict):
    """Renders the Student Life, Wellbeing, and Study Tips Tab."""
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0,212,255,0.05) 0%, rgba(128,0,0,0.05) 100%); padding: 30px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);">
            <h2 style="margin-top:0;">🌱 Student Life & Wellness</h2>
            <p>Your sanctuary for motivation, humor, and mental clarity during the engineering grind.</p>
        </div>
        <br/>
    """, unsafe_allow_html=True)

    col_growth1, col_growth2 = st.columns([1, 1])

    with col_growth1:
        st.subheader("📝 Daily Engineering Journal")
        if "journal_entry" not in st.session_state:
            st.session_state["journal_entry"] = ""
        
        journal_text = st.text_area("Write down your thoughts, stress, or goals for today:", 
                                    value=st.session_state["journal_entry"], height=200,
                                    placeholder="E.g., Today I finally understood pointers in C...")
        st.session_state["journal_entry"] = journal_text

        if journal_text:
            st.download_button("📥 Download Journal", data=journal_text, file_name="journal_entry.txt", mime="text/plain")

        st.divider()

        st.subheader("✨ Need a Boost?")
        if st.button("Generate AI Core Motivation"):
            API_KEY = secrets_dict.get("GEMINI_API_KEY")
            if not API_KEY:
                st.warning("Google Gemini API is missing. Cannot generate quotes.")
            else:
                with st.spinner("Channeling peace..."):
                    try:
                        client = genai.Client(api_key=API_KEY)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents="Give me exactly one highly motivating, calming, and perspective-shifting quote tailored specifically for an Indian engineering student who is stressed about VTU exams or placements. Keep it under 3 sentences. Do not use quotes; speak directly to them.",
                        )
                        st.success(f"**Wisdom:** {response.text}")
                    except Exception as e:
                        st.error("Could not fetch motivation right now. Remember: You are stronger than your hardest exam!")

    with col_growth2:
        st.subheader("🧠 Smart Study Hacks (VTU 2022/2026)")
        st.info("**🍅 The Pomodoro Technique:** Work for 25 minutes, then take a 5-minute break.")
        st.success("**🧑‍🏫 The Feynman Technique:** Explain the concept to a 5-year-old. Best for DSA.")
        st.warning("**🔁 Active Recall Space Repetition:** Actively quiz yourself using Anki flashcards.")

        st.divider()

        st.subheader("😂 The Engineer's Laugh")
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "There are 10 types of people in the world: those who understand binary, and those who don't.",
            "A SQL query walks into a bar, approaches two tables and asks... 'Can I join you?'",
            "Why did the programmer quit his job? Because he didn't get arrays.",
            "How many programmers does it take to change a light bulb? None, that's a hardware problem.",
            "Engineering Professor: 'You have 3 hours for this open-book VTU exam.'\n*Narrator: None of them found the answers in the book.*",
            "I would love to change the world, but they won't give me the source code."
        ]
        if st.button("Relieve Exam Stress (Joke)"):
            st.info(f"_{random.choice(jokes)}_")


# --- MAIN EXECUTION PIPELINE ---
def main():
    setup_page()
    secrets_dict = load_secrets()
    
    # Render Login System
    authenticator, name, authentication_status = render_login_system(secrets_dict)
    
    # Handle Unauthenticated State
    if authentication_status is False:
        st.error("Username/password is incorrect. Please try again.")
        
    if authentication_status is not True:
        render_public_view(secrets_dict)
        st.stop()
        
    # Handle Authenticated State
    st.sidebar.title(f"Welcome, {name} 🎓")
    authenticator.logout("Logout", "sidebar")
    
    tab_ml, tab_ensemble, tab_ai, tab_wellness, tab_games, tab_library, tab_growth = st.tabs([
        "🎯 AI Predictor", 
        "🔬 Ensemble Lab",
        "🤖 AI Mentor",
        "🧘 Zen Zone", 
        "🎮 Brain Games",
        "📚 Global Library",
        "🌱 Daily Life & Growth"
    ])
    
    with tab_ml: render_ml_predictor()
    with tab_ensemble: render_ensemble_lab()
    with tab_ai: render_ai_mentor(secrets_dict)
    with tab_wellness: render_wellness_zen()
    with tab_games: render_brain_games()
    with tab_library: render_global_library()
    with tab_growth: render_daily_growth(secrets_dict)


if __name__ == "__main__":
    main()
