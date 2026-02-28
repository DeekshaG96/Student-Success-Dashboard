import streamlit as st
import pandas as pd
import altair as alt
import google.generativeai as genai

st.set_page_config(
    page_title="Student Success Dashboard",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Indian Student Success Portal")
st.markdown("Explore your academic dashboard, scholarships, and daily general knowledge.")

# Authentication & Global State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'student_goals' not in st.session_state:
    st.session_state['student_goals'] = [("Register for JEE Main", "2026-03-15"), ("Complete Physics Mock Test", "2026-03-01")]
if 'xp' not in st.session_state:
    st.session_state['xp'] = 0
if 'level' not in st.session_state:
    st.session_state['level'] = 1
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I am your AI Career Counselor. How can I help you regarding degrees, exams, or scholarships today?"}]

if not st.session_state['logged_in']:
    st.write("### Welcome! Please log in to access your portal.")
    with st.form("login_form"):
        username = st.text_input("Username (Hint: student)")
        password = st.text_input("Password (Hint: india2026)", type="password")
        submit_button = st.form_submit_button("Sign In")
        
        if submit_button:
            if username == "student" and password == "india2026":
                st.session_state['logged_in'] = True
                st.success("Logged in successfully! Reloading...")
                st.rerun()
            else:
                st.error("Invalid credentials. Please use the hints!")
    st.stop() # Halt execution if not logged in

st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False) or st.rerun())

@st.cache_data
def load_data():
    df = pd.read_csv("students_dropout_academic_success.csv")
   
    return df

try:
    df_raw = load_data()
    
    with st.sidebar:
        st.header("🎮 Student Profile")
        st.markdown(f"**Level {st.session_state['level']}** Scholar")
        
        # Gamification XP progression logic
        next_level_xp = st.session_state['level'] * 100
        current_xp = st.session_state['xp']
        progress = min(current_xp / next_level_xp, 1.0)
        st.progress(progress, text=f"{current_xp}/{next_level_xp} XP to Level {st.session_state['level'] + 1}")
        st.markdown("---")
        
    # Sidebar Filters
    st.sidebar.header("Filter Data")
    
    # Define mappings for user-friendly labels
    gender_map = {1: "Male", 0: "Female"}
    gender_inv = {"Male": 1, "Female": 0}
    
    bool_map = {1: "Yes", 0: "No"}
    bool_inv = {"Yes": 1, "No": 0}
    
    # Get unique dataset values and map them to strings
    avail_genders = [gender_map.get(x, str(x)) for x in df_raw["Gender"].unique()]
    avail_scholars = [bool_map.get(x, str(x)) for x in df_raw["Scholarship holder"].unique()]
    avail_displaced = [bool_map.get(x, str(x)) for x in df_raw["Displaced"].unique()]
    
    # Render user-friendly multiselects
    gender_selection = st.sidebar.multiselect("Gender", options=avail_genders, default=avail_genders)
    scholarship_selection = st.sidebar.multiselect("Scholarship holder", options=avail_scholars, default=avail_scholars)
    displaced_selection = st.sidebar.multiselect("Displaced", options=avail_displaced, default=avail_displaced)
    
    # Translate string selections back to integers before filtering
    gender_filter = [gender_inv.get(x, x) for x in gender_selection]
    scholarship_filter = [bool_inv.get(x, x) for x in scholarship_selection]
    displaced_filter = [bool_inv.get(x, x) for x in displaced_selection]
    
    # Apply filters using integer values
    df = df_raw[
        (df_raw["Gender"].isin(gender_filter)) &
        (df_raw["Scholarship holder"].isin(scholarship_filter)) &
        (df_raw["Displaced"].isin(displaced_filter))
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Dashboard", "Predictor Page", "Scholarships", "GK Gauntlet", "Career Guidance", "Top Colleges (NIRF)", "2026 Exam News", "My Goals", "AI Counselor"])
    
    with tab1:
        # Live Metrics
        st.write("### Key Performance Indicators")
        total_students = len(df)
        if total_students > 0:
            dropout_rate = (len(df[df["target"] == "Dropout"]) / total_students) * 100
            graduate_rate = (len(df[df["target"] == "Graduate"]) / total_students) * 100
        else:
            dropout_rate = 0.0
            graduate_rate = 0.0
            
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Students", f"{total_students:,}")
        col_m2.metric("Dropout Rate", f"{dropout_rate:.1f}%")
        col_m3.metric("Graduation Rate", f"{graduate_rate:.1f}%")
        
        st.markdown("---")
        
        # Enhanced Data Preview
        with st.expander("Explore Raw Data & Statistics"):
            st.write("#### Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)
            st.write("#### Descriptive Statistics for Numerical Features")
            st.dataframe(df.describe(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Target Value Distribution")
            target_counts = df["target"].value_counts().reset_index()
            target_counts.columns = ["Target", "Count"]
            
            target_chart = alt.Chart(target_counts).mark_bar().encode(
                x=alt.X("Target:N", title="Outcome"),
                y=alt.Y("Count:Q", title="Number of Students"),
                color=alt.Color("Target:N", legend=None)
            ).properties(
                height=400
            )
            st.altair_chart(target_chart, use_container_width=True)
            
        with col2:
            st.write("### Previous Qualification Grade vs Target")
            if "Previous qualification (grade)" in df.columns:
                grade_chart = alt.Chart(df).mark_boxplot().encode(
                    x=alt.X("target:N", title="Outcome"),
                    y=alt.Y("Previous qualification (grade):Q", title="Previous Grade"),
                    color=alt.Color("target:N", legend=None)
                ).properties(
                    height=400
                )
                st.altair_chart(grade_chart, use_container_width=True)
            else:
                st.info("Grade column not found.")

    with tab2:
        st.write("### Predict Student Success")
        
        @st.cache_resource
        def load_model_from_hf():
            """
            Function to load a model from Hugging Face Hub using Streamlit secrets.
            """
            try:
                from huggingface_hub import hf_hub_download
                import joblib
                
                # Fetch secrets
                REPO_ID = st.secrets["REPO_ID"]
                HF_TOKEN = st.secrets["HF_TOKEN"]
                # The model must be named exactly 'model.joblib'
                MODEL_FILENAME = "model.joblib" 
                
                with st.spinner("Downloading model from Hugging Face..."):
                    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)
                    model = joblib.load(model_path)
                return model
                
            except Exception as model_error:
                st.error("Model could not be loaded. Please ensure your Hugging Face Space is created and contains the model.joblib file.")
                with st.expander("Show specific error"):
                    st.error(f"{model_error}")
                return None
                
        model = load_model_from_hf()
        
        if model is not None:
            st.write("#### Predict & Compare")
            col_in, col_viz = st.columns([1, 2])
            
            with col_in:
                st.write("##### Enter Student Details")
                with st.form("prediction_form"):
                    grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=120.0, step=1.0)
                    age = st.number_input("Age at Enrollment", min_value=15, max_value=60, value=20)
                    submitted = st.form_submit_button("Predict Outcome")
                    
                    if submitted:
                        st.success("Prediction logic would execute here using your inputs!")
                        
            with col_viz:
                st.write("##### Live Scenario Comparison")
                # Create a comparison dataframe
                avg_grade = df_raw["Previous qualification (grade)"].mean() if "Previous qualification (grade)" in df_raw.columns else 0
                avg_age = df_raw["Age at enrollment"].mean() if "Age at enrollment" in df_raw.columns else 0
                
                comp_data = pd.DataFrame({
                    "Metric": ["Previous Grade", "Previous Grade", "Age at Enrollment", "Age at Enrollment"],
                    "Value": [grade, avg_grade, age, avg_age],
                    "Type": ["Your Input", "Historical Average", "Your Input", "Historical Average"]
                })
                
                comp_chart = alt.Chart(comp_data).mark_bar().encode(
                    x=alt.X("Type:N", title=None, axis=alt.Axis(labels=False)),
                    y=alt.Y("Value:Q", title="Value"),
                    color=alt.Color("Type:N", legend=alt.Legend(title="Data Source", orient="bottom")),
                    column=alt.Column("Metric:N", title="")
                ).properties(
                    width=150,
                    height=300
                ).configure_view(
                    stroke='transparent'
                )
                
                st.altair_chart(comp_chart, use_container_width=False)

    with tab3:
        st.write("### 🇮🇳 Top Indian Student Scholarships")
        st.markdown("Explore national and state scholarships available to support your academic success.")
        
        with st.expander("National Scholarship Portal (NSP) - Pre/Post Matric", expanded=True):
            st.write("**Eligibility:** Students from minority communities from Class 1 up to Ph.D. level.")
            st.write("**Benefits:** Admission + Tuition fee and maintenance allowance.")
            st.markdown("[View Details on NSP](https://scholarships.gov.in/)")
            
        with st.expander("AICTE Pragati Scholarship for Girls"):
            st.write("**Eligibility:** Maximum 2 girl children per family entering AICTE approved technical degree/diploma courses.")
            st.write("**Benefits:** Up to ₹50,000 per annum for every year of study.")
            st.markdown("[View AICTE Portal](https://www.aicte-india.org/)")
            
        with st.expander("Prime Minister's Special Scholarship Scheme (PMSSS)"):
            st.write("**Eligibility:** Students from Jammu & Kashmir and Ladakh pursuing undergrad studies outside the UTs.")
            st.write("**Benefits:** Academic fee and maintenance allowance up to ₹3 Lakhs/year.")
            
        with st.expander("Kishore Vaigyanik Protsahan Yojana (KVPY)"):
            st.write("**Eligibility:** Students from Class 11 to 1st year of any undergraduate program in Basic Sciences.")
            st.write("**Benefits:** Monthly fellowship of ₹5,000 to ₹7,000 and an annual contingency grant.")
            st.markdown("[View Details on Department of Science and Technology](https://dst.gov.in/)")
            
        with st.expander("National Talent Search Examination (NTSE)"):
            st.write("**Eligibility:** Class 10 students studying in recognized schools across India.")
            st.write("**Benefits:** Scholarships up to Ph.D. level for sciences and social sciences, and up to second-degree level for professional courses.")
            st.markdown("[View Details on NCERT](https://ncert.nic.in/national-talent-examination.php)")

    with tab4:
        st.write("### 🧠 General Knowledge & Fun Facts")
        
        col_gk1, col_gk2 = st.columns(2)
        with col_gk1:
            st.info("**Fact of the Day:** India established the first university in the world, Takshashila, in 700 BC. More than 10,500 students from all over the world studied more than 60 different subjects there!")
            st.success("**Science Fact:** Aryabhata (born 476 CE) was the first of the major mathematician-astronomers from the classical age of Indian mathematics. He deduced that the Earth is round and rotates on its own axis.")
            
        with col_gk2:
            st.write("#### The 5-Question GK Gauntlet")
            st.write("Test your ultimate Indian General Knowledge!")
            
            with st.form("gk_quiz_form"):
                q1 = st.radio("1. What is the capital of Karnataka?", 
                                       ["Mysuru", "Bengaluru", "Mangaluru", "Hubballi"], 
                                       index=None)
                                       
                q2 = st.radio("2. Which Indian scientist won the 1930 Nobel Prize for Physics for the scattering of light?",
                                       ["Homi J. Bhabha", "Satyendra Nath Bose", "C.V. Raman", "Vikram Sarabhai"],
                                       index=None)
                                       
                q3 = st.radio("3. Who was the first Indian woman of Indian origin to go to space?",
                                       ["Kalpana Chawla", "Sunita Williams", "Sirisha Bandla", "Shawna Pandya"],
                                       index=None)
                                       
                q4 = st.radio("4. Where is the Indian Space Research Organisation (ISRO) headquartered?",
                                       ["New Delhi", "Hyderabad", "Thiruvananthapuram", "Bengaluru"],
                                       index=None)
                                       
                q5 = st.radio("5. Which ancient Indian physician is widely known as the 'Father of Surgery'?",
                                       ["Charaka", "Patanjali", "Sushruta", "Vagbhata"],
                                       index=None)
                                       
                submit_quiz = st.form_submit_button("Submit Answers")
                
                if submit_quiz:
                    score = 0
                    if q1 == "Bengaluru":
                        score += 1
                    if q2 == "C.V. Raman":
                        score += 1
                    if q3 == "Kalpana Chawla":
                        score += 1
                    if q4 == "Bengaluru":
                        score += 1
                    if q5 == "Sushruta":
                        score += 1
                        
                    if score == 5:
                        st.balloons()
                        st.success(f"**Perfect Score! You are a master! {score}/5**")
                        # Gamification Reward
                        st.session_state['xp'] += 100
                        if st.session_state['xp'] >= (st.session_state['level'] * 100):
                            st.session_state['xp'] = st.session_state['xp'] - (st.session_state['level'] * 100)
                            st.session_state['level'] += 1
                            st.sidebar.success(f"Level Up! You are now Level {st.session_state['level']}!")
                    elif score >= 3:
                        st.info(f"**Great Job! {score}/5**. Keep learning!")
                    else:
                        st.warning(f"**Total Score: {score}/5**. Better luck next time!")

    with tab5:
        st.write("### 🧭 Indian Career & Entrance Exam Guidance")
        st.write("Select your 12th-grade stream to see the most popular degree paths and required entrance exams in India.")
        
        stream = st.selectbox("Select Your 12th Grade Stream:", ["Science (PCM)", "Science (PCB)", "Commerce", "Arts/Humanities"])
        
        col_c1, col_c2 = st.columns(2)
        if stream == "Science (PCM)":
            with col_c1:
                st.info("**Top Degree Paths:** \n\n* B.Tech / B.E. (Engineering)\n* B.Arch (Architecture)\n* B.Sc. (Physics/Chemistry/Math)\n* BCA (Computer Applications)")
            with col_c2:
                st.warning("**Major Entrance Exams:** \n\n* JEE Main & Advanced\n* BITSAT\n* VITEEE\n* State CETs (e.g., KCET, MHT-CET)")
        elif stream == "Science (PCB)":
            with col_c1:
                st.info("**Top Degree Paths:** \n\n* MBBS / BDS (Medicine/Dentistry)\n* B.Sc. Nursing\n* B.Pharm (Pharmacy)\n* B.Sc. (Biology/Genetics/Agriculture)")
            with col_c2:
                st.warning("**Major Entrance Exams:** \n\n* NEET-UG\n* AIIMS Nursing\n* ICAR AIEEA\n* State-level Medical/Pharmacy CETs")
        elif stream == "Commerce":
            with col_c1:
                st.info("**Top Degree Paths:** \n\n* B.Com (Hons.)\n* BBA / BMS (Management)\n* CA / CS / CMA (Professional)\n* B.A. Economics (Hons.)")
            with col_c2:
                st.warning("**Major Entrance Exams:** \n\n* CUET-UG (For Central Universities)\n* CA Foundation\n* SET / NPAT / IPMAT (For Management)")
        elif stream == "Arts/Humanities":
            with col_c1:
                st.info("**Top Degree Paths:** \n\n* B.A. (Hons.) in History/Political Science/English\n* B.A. LLB (Law)\n* B.Des (Design)\n* Journalism & Mass Comm.")
            with col_c2:
                st.warning("**Major Entrance Exams:** \n\n* CUET-UG\n* CLAT / AILET (For Law)\n* NID DAT / NIFT (For Design)")

    with tab6:
        st.write("### 🏆 Top Colleges in India (NIRF Rankings 2023-24)")
        st.markdown("Explore the official highest-ranked institutions by the National Institutional Ranking Framework (GoI).")
        
        ranking_category = st.radio("Select Category:", ["Engineering", "Medical", "Management"], horizontal=True)
        
        if ranking_category == "Engineering":
            eng_data = pd.DataFrame({
                "Rank": [1, 2, 3, 4, 5],
                "Institute": ["IIT Madras", "IIT Delhi", "IIT Bombay", "IIT Kanpur", "IIT Roorkee"],
                "City": ["Chennai", "New Delhi", "Mumbai", "Kanpur", "Roorkee"],
                "Score": [89.79, 87.09, 80.74, 80.65, 71.12]
            })
            st.dataframe(eng_data, hide_index=True, use_container_width=True)
            
        elif ranking_category == "Medical":
            med_data = pd.DataFrame({
                "Rank": [1, 2, 3, 4, 5],
                "Institute": ["AIIMS Delhi", "PGIMER", "CMC Vellore", "NIMHANS", "JIPMER"],
                "City": ["New Delhi", "Chandigarh", "Vellore", "Bengaluru", "Puducherry"],
                "Score": [94.32, 81.10, 75.29, 72.46, 72.10]
            })
            st.dataframe(med_data, hide_index=True, use_container_width=True)
            
        elif ranking_category == "Management":
            mgmt_data = pd.DataFrame({
                "Rank": [1, 2, 3, 4, 5],
                "Institute": ["IIM Ahmedabad", "IIM Bangalore", "IIM Kozhikode", "IIM Calcutta", "IIT Delhi"],
                "City": ["Ahmedabad", "Bengaluru", "Kozhikode", "Kolkata", "New Delhi"],
                "Score": [83.20, 80.89, 76.48, 75.53, 74.14]
            })
            st.dataframe(mgmt_data, hide_index=True, use_container_width=True)

    with tab7:
        st.write("### 📰 2026 Exam News & Policy Alerts")
        st.markdown("Stay ahead with expected exam dates and major shifts in Indian education policy.")
        
        with st.container():
            st.error("**🚨 Highly Anticipated 2026 Exam Dates (Expected Calendar)**")
            col_ex1, col_ex2, col_ex3 = st.columns(3)
            col_ex1.metric("JEE Main (Session 1)", "Jan 2026", "Registration: Nov 2025")
            col_ex2.metric("NEET UG 2026", "May 2026", "Registration: Feb 2026")
            col_ex3.metric("CUET UG 2026", "May 2026", "Registration: Mar 2026")
            
        st.markdown("---")
        st.info("**📜 Recent Policy Spotlight: National Education Policy (NEP) 2020 rollout**")
        st.write("""
        * **4-Year UG Programs (FYUGP):** Most central universities now offer 4-year undergraduate degrees with multiple entry and exit points.
        * **Exit Options:** Students can obtain a Certificate after Year 1, a Diploma after Year 2, a Bachelor's Degree after Year 3, and a Bachelor's with Research/Honours after Year 4.
        * **Academic Bank of Credits (ABC):** Your academic credits are now securely stored digitally, enabling easier transfers between recognized institutions.
        """)

    with tab8:
        st.write("### 🎯 My Goals & Study Tracker")
        st.markdown("Keep track of your study targets, form submissions, and daily task deadlines.")
        
        # Add a new goal with a target date
        with st.form("add_goal_form", clear_on_submit=True):
            col_g1, col_g2 = st.columns([3, 1])
            with col_g1:
                new_goal = st.text_input("Add a new study goal or checklist item:")
            with col_g2:
                target_date = st.date_input("Target Date")
                
            submitted_goal = st.form_submit_button("Add Goal")
            if submitted_goal and new_goal:
                st.session_state['student_goals'].append((new_goal, target_date.strftime("%Y-%m-%d")))
                st.rerun()
                
        # Display goals as an interactive checklist
        st.write("#### Active Calendar Goals:")
        if not st.session_state['student_goals']:
            st.success("All caught up! You have no active goals in your calendar.")
        else:
            goals_to_remove = []
            for i, goal_data in enumerate(st.session_state['student_goals']):
                # Maintain backwards compatibility if switching from strings to tuples
                if isinstance(goal_data, tuple):
                    goal_text, goal_date = goal_data
                    completed = st.checkbox(f"**[{goal_date}]** - 📍 {goal_text}", key=f"goal_{i}")
                else:
                    completed = st.checkbox(f"📍 {goal_data}", key=f"goal_{i}")
                    
                if completed:
                    # Gamification Reward
                    st.session_state['xp'] += 50
                    if st.session_state['xp'] >= (st.session_state['level'] * 100):
                        st.session_state['xp'] = st.session_state['xp'] - (st.session_state['level'] * 100)
                        st.session_state['level'] += 1
                        st.sidebar.success(f"Level Up! You are now Level {st.session_state['level']}!")
                    goals_to_remove.append(goal_data)
            
            # Remove checked items and rerun
            if goals_to_remove:
                for completed_goal in goals_to_remove:
                    st.session_state['student_goals'].remove(completed_goal)
                st.rerun()
                
        # Data Export Feature
        if st.session_state['student_goals']:
            st.markdown("---")
            st.write("#### 💾 Export Your Data")
            # Build DataFrame for Download
            goals_export_df = pd.DataFrame(st.session_state['student_goals'], columns=["Goal", "Target Date"])
            csv_data = goals_export_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Study Plan (CSV)",
                data=csv_data,
                file_name="my_study_plan.csv",
                mime="text/csv",
            )

    with tab9:
        st.write("### 🤖 AI Career Counselor")
        st.markdown("Chat with your personalized AI assistant! Ask anything about coding, degree paths, scholarships, or entrance exams.")
        
        # Render existing chat messages
        for msg in st.session_state.messages:
            # Map Gemini's "model" role back to "assistant" for Streamlit UI
            role = "assistant" if msg["role"] == "model" else msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])
                
        # Chat input box
        if prompt := st.chat_input("Ask me about the difference between B.Tech and BCA..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            gemini_key = st.secrets.get("GEMINI_API_KEY")
            if not gemini_key:
                st.error("Missing GEMINI_API_KEY in Streamlit secrets. Please add it to unlock the AI Counselor.")
            else:
                try:
                    # Configure the Google Gemini Client
                    genai.configure(api_key=gemini_key)
                    
                    # System instructions for Gemini 2.0 Flash
                    system_instruction = "You are a helpful Indian academic counselor. Give short, concise, and accurate advice to students regarding degrees, exams, and scholarships in India."
                    model = genai.GenerativeModel(
                        model_name="gemini-2.0-flash",
                        system_instruction=system_instruction
                    )
                    
                    # Gemini requires chat history in a specific format: {"role": "user"|"model", "parts": ["text"]}
                    # We must filter out any legacy "assistant" roles from previous sessions
                    history = []
                    for state_msg in st.session_state.messages[:-1]: # Exclude the prompt we just added
                        role = "model" if state_msg["role"] == "assistant" else state_msg["role"]
                        
                        # Only allow valid Gemini roles
                        if role in ["user", "model"]:
                            history.append({"role": role, "parts": [state_msg["content"]]})
                            
                    # Initialize the chat session
                    chat = model.start_chat(history=history)
                    
                    with st.chat_message("assistant"):
                        # Get AI response
                        response = chat.send_message(prompt)
                        bot_reply = response.text
                        st.markdown(bot_reply)
                        
                    st.session_state.messages.append({"role": "model", "content": bot_reply})
                except Exception as e:
                    st.error(f"Failed to connect to AI server. Error: {e}")

except Exception as e:
    st.error(f"Error loading data: {e}")
