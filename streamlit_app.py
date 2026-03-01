import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import time
import os
import joblib
import numpy as np
import google.generativeai as genai
import requests
from huggingface_hub import hf_hub_download

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="International Student Success & Wellness Hub", 
    page_icon="🎓", 
    layout="wide"
)

# --- HELPER TO LOAD SECRETS SAFELY ---
@st.cache_data
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

secrets_dict = load_secrets()

# --- FEATURE 1: ENTERPRISE SECURITY ---
if "credentials" not in secrets_dict or "cookie" not in secrets_dict:
    st.error("Authentication secrets missing! Please properly configure 'credentials' and 'cookie' in st.secrets.")
    st.stop()

# Initialize authenticator
authenticator = stauth.Authenticate(
    secrets_dict["credentials"],
    secrets_dict["cookie"]["name"],
    secrets_dict["cookie"]["key"],
    secrets_dict["cookie"].get("expiry_days", 30),
    secrets_dict.get("preauthorized", None)
)

name, authentication_status, username = authenticator.login("main", fields={'Form name': 'Login to CSBS Portal'})

if authentication_status is False:
    st.error("Username/password is incorrect. Please try again.")
    st.stop()
elif authentication_status is None:
    st.warning("Please enter your username and password to access the portal.")
    st.stop()

# --- AUTHENTICATED AREA ---
st.sidebar.title(f"Welcome, {name}")
authenticator.logout("Logout", "sidebar")

# SIT Branding
st.markdown("""
    <style>
    .vtu-header { color: #004d99; font-weight: bold; margin-bottom: 0px; }
    .department-sub { color: #555555; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("sit_logo.png"):
        st.image("sit_logo.png", use_container_width=True)
    else:
        st.info("SIT Logo Missing (sit_logo.png)")
with col2:
    st.markdown("<h1 class='vtu-header'>International Student Success & Wellness Hub</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='department-sub'>Department of Computer Science & Business Systems</h3>", unsafe_allow_html=True)
    st.markdown("**Srinivas Institute of Technology (SIT) — Visvesvaraya Technological University (VTU)**")

st.divider()

# --- TABS CREATION ---
tab_ml, tab_ai, tab_wellness, tab_games, tab_library = st.tabs([
    "📊 MLOps Predictor", 
    "🤖 Global AI Mentor",
    "🧘 Zen Wellness", 
    "♟️ Brain Games",
    "📚 Digital Library"
])

# --- FEATURE 2: MLOps SUCCESS PREDICTOR ---
with tab_ml:
    st.header("MLOps Success Predictor")
    st.write("Predict academic outcomes using models configured via Hugging Face Hub.")
    
    with st.form("ml_form"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            prev_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=130.0, step=1.0)
        with col_m2:
            age = st.number_input("Age at Enrollment", min_value=15, max_value=60, value=18, step=1)
            
        submitted = st.form_submit_button("Predict Outcome")
        
    if submitted:
        with st.spinner("Loading model and predicting outcome..."):
            try:
                # Load the pre-trained model
                if not os.path.exists("model.joblib"):
                    raise FileNotFoundError("model.joblib not found in the root directory.")
                
                model = joblib.load("model.joblib")
                
                # Prepare input data for prediction
                input_data = np.array([[prev_grade, age]])
                
                # Make prediction
                prediction = model.predict(input_data)
                prediction_text = str(prediction[0]).title()
                
                if prediction_text == "Graduate":
                    st.success(f"Predictive Outcome: The student is strongly projected to **{prediction_text}**.")
                elif prediction_text == "Dropout":
                    st.error(f"Predictive Outcome: Alert - The student is flagged as a potential **{prediction_text}** risk.")
                else:
                    st.info(f"Predictive Outcome: The student status is **{prediction_text}**.")
                    
            except Exception as e:
                st.error(f"Error loading model or making prediction: {str(e)}")
                st.info("Please ensure that model.joblib is placed in the project root and is compatible with scikit-learn==1.6.1.")

# --- FEATURE 3: GLOBAL AI MENTOR ---
with tab_ai:
    st.header("Global AI Mentor")
    st.write("Consult the Gemini Pro expert on GATE 2027, International Higher Ed, and Advanced CSBS Subjects.")
    
    API_KEY = secrets_dict.get("GEMINI_API_KEY")
    if not API_KEY:
        st.error("GEMINI_API_KEY missing from st.secrets. Mentor is offline.")
    else:
        try:
            genai.configure(api_key=API_KEY)
            
            # System instructions representing the specialized AI Mentor persona
            sys_instruct = (
                "You are an elite Principal Global AI Mentor for CSBS students at Srinivas Institute of Technology. "
                "Act as an absolute expert on: "
                "1. GATE 2027 preparation strategies and syllabus. "
                "2. International higher education pathways (specifically USA, Germany, Ireland). "
                "3. Advanced CSBS subjects like Reinforcement Learning, MLOps, and Deep Learning."
            )
            
            model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=sys_instruct)
            
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
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                st.session_state.ai_messages.append({"role": "assistant", "content": response.text})
                
        except Exception as e:
            st.error(f"Error communicating with Gemini API: {str(e)}")

# --- FEATURE 4: ZEN WELLNESS & ALARM ---
with tab_wellness:
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
    st.subheader("🎧 Ambient Sounds")
    sound_choice = st.selectbox("Select soundscape:", ["Temple Bells", "Rain", "Deep Space", "Ocean Waves"])
    st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format="audio/mp3")
    st.caption(f"Currently playing placeholder track for '{sound_choice}'.")

# --- FEATURE 5: BRAIN GAMES HUB ---
with tab_games:
    st.header("Brain Games Hub")
    st.write("Take a cognitive break to sharpen your mind with logic games.")
    
    game_choice = st.radio("Choose a game strategy:", ["♟️ Chess (Interactive Window)", "🧩 Sudoku Generator (Placeholder)"], horizontal=True)
    
    if "Chess" in game_choice:
        st.subheader("Interactive Chess")
        st.write("Train against the computer using a lightweight iframe instance.")
        # Using a sleek Lichess embedded iframe designed for interactive play without heavy local libraries
        components.iframe("https://lichess.org/training/frame?theme=brown&bg=dark", height=450)
        
    elif "Sudoku" in game_choice:
        st.subheader("Sudoku Logic Generator")
        st.info("Future enhancement: Integrate a Python-based Sudoku generator payload here.")
        # Visual placeholder displaying the concept cleanly
        st.markdown("""
        ```text
        +-------+-------+-------+
        | 5 3 4 | 6 7 8 | 9 1 2 |
        | 6 7 2 | 1 9 5 | 3 4 8 |
        | 1 9 8 | 3 4 2 | 5 6 7 |
        +-------+-------+-------+
        | 8 5 9 | 7 6 1 | 4 2 3 |
        | 4 2 6 | 8 5 3 | 7 9 1 |
        | 7 1 3 | 9 2 4 | 8 5 6 |
        +-------+-------+-------+
        | 9 6 1 | 5 3 7 | 2 8 4 |
        | 2 8 7 | 4 1 9 | 6 3 5 |
        | 3 4 5 | 2 8 6 | 1 7 9 |
        +-------+-------+-------+
        ```
        """)

# --- FEATURE 6: INTERNATIONAL DIGITAL LIBRARY ---
with tab_library:
    st.header("International Digital Library")
    st.write("Free search for textbooks, CS journals, and global resources via the Open Library API.")
    
    search_query = st.text_input("Enter a subject (e.g., 'Reinforcement Learning', 'Germany Education', 'Data Structures')")
    
    if st.button("Search Open Library"):
        if search_query.strip() == "":
            st.warning("Please enter a valid search term.")
        else:
            with st.spinner(f"Querying Open Library for '{search_query}'..."):
                try:
                    clean_query = search_query.replace(' ', '+')
                    # Secure requests call with a timeout to prevent infinite blocking on Cloud
                    response = requests.get(f"https://openlibrary.org/search.json?q={clean_query}&limit=6", timeout=12)
                    
                    if response.status_code == 200:
                        data = response.json()
                        docs = data.get("docs", [])
                        
                        if not docs:
                            st.info("No matching books or journals found. Try broadening your terms.")
                        else:
                            st.success(f"Retrieved the top {len(docs)} highly relevant results!")
                            for doc in docs:
                                title = doc.get("title", "Unknown Subject")
                                authors = ", ".join(doc.get("author_name", ["Unknown Editor"]))
                                year = doc.get("first_publish_year", "Unknown Year")
                                preview = doc.get("key", "")
                                
                                st.markdown(f"**📚 {title}** *(Published: {year})*")
                                st.markdown(f"**Authors:** {authors}")
                                if preview:
                                    st.markdown(f"[Read Resource on Open Library](https://openlibrary.org{preview})")
                                st.divider()
                    else:
                        st.error(f"API Interface Error: Open Library responded with Status Code {response.status_code}")
                except Exception as e:
                    st.error(f"Failed to connect to Open Library API: {str(e)}")
