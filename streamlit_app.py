import streamlit as st
import pandas as pd
import altair as alt
import google.generativeai as genai
import streamlit_authenticator as stauth
import os

st.set_page_config(
    page_title="Student Success Dashboard",
    page_icon="🎓",
    layout="wide",
)

# Custom CSS Injection for Premium Dark Mode Aesthetics
st.markdown("""
<style>
    /* Animated Gradient Title */
    .title-glow {
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6, #8b5cf6, #0ea5e9);
        background-size: 300% 300%;
        color: transparent;
        -webkit-background-clip: text;
        animation: gradient_anim 5s ease infinite;
        padding-bottom: 10px;
    }
    
    @keyframes gradient_anim {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Interactive Buttons */
    div.stButton > button {
        transition: all 0.3s ease-in-out;
        border-radius: 8px;
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #e2e8f0;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 15px rgba(14, 165, 233, 0.4);
        border-color: #0ea5e9;
        color: #0ea5e9;
    }
    
    /* Glassmorphism Metric Cards & Expanders */
    div[data-testid="metric-container"], div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #0ea5e9 !important;
    }
    
    div[data-testid="metric-container"]:hover, div[data-testid="stExpander"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.4);
        border-left: 4px solid #8b5cf6 !important;
        background: rgba(30, 41, 59, 0.9) !important;
    }
    
    /* Soften Sidebar Background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title-glow">⚡ Advanced Engineering Student Portal</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Your centralized hub for AI Guidance, Tech Prep, and Advanced Engineering Analytics.</p>", unsafe_allow_html=True)

# Authentication & Global State
if 'student_goals' not in st.session_state:
    st.session_state['student_goals'] = [("Register for JEE Main", "2026-03-15"), ("Complete Physics Mock Test", "2026-03-01")]
if 'xp' not in st.session_state:
    st.session_state['xp'] = 0
if 'level' not in st.session_state:
    st.session_state['level'] = 1
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I am your SIT AI Academic Mentor. How can I help you regarding VTU exams, GATE 2027, or scholarships today?"}]

# Initialize Authenticator with mutable dictionary (st.secrets is immutable)
credentials = st.secrets.get('credentials')
if credentials:
    credentials_dict = {
        'usernames': {
            username: dict(user_info) 
            for username, user_info in credentials['usernames'].items()
        }
    }
else:
    credentials_dict = {'usernames': {}}

authenticator = stauth.Authenticate(
    credentials_dict,
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days']
)

# Login UI
col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
with col_l2:
    if os.path.exists("sit_logo.png"):
        st.image("sit_logo.png", width=150)
    else:
        st.write("### 🎓 Srinivas Institute of Technology")
    
    # Corrected login call for v0.3.1
    authentication_status = authenticator.login(location='main')

if authentication_status == False:
    st.error('Username/password is incorrect')
    st.stop()
elif authentication_status == None:
    st.warning('Please enter your username and password')
    st.stop()

# Successful Login - Branding in Sidebar
st.sidebar.image("sit_logo.png", width=100) if os.path.exists("sit_logo.png") else st.sidebar.write("### SIT Portal")
authenticator.logout(location='sidebar')

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
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["⚡ Smart Dashboard", "🔮 Predictor (AI)", "🇮🇳 Indian Student Resource Center", "🚀 GATE Prep", "🧭 Career Paths", "🏆 Top Colleges", "📰 Tech News", "🎯 My Goals", "🧠 AI Mentor (SIT)", "🧘 Zen Study Hub"])
    
    with tab1:
        # Smart Dashboard Homepage
        st.write("### ⚡ Live Metrics & Daily Overview")
        col_m1, col_m2, col_m3 = st.columns(3)
        total_students = len(df)
        dropout_rate = (len(df[df["target"] == "Dropout"]) / total_students * 100) if total_students else 0.0
        graduate_rate = (len(df[df["target"] == "Graduate"]) / total_students * 100) if total_students else 0.0
        
        col_m1.metric("Total Profiles", f"{total_students:,}")
        col_m2.metric("Dropout Risk", f"{dropout_rate:.1f}%")
        col_m3.metric("Graduation Rate", f"{graduate_rate:.1f}%")
        
        st.markdown("---")
        
        col_dash1, col_dash2 = st.columns([1, 1])
        
        with col_dash1:
            st.info("#### 🚀 Daily GATE Prep Question")
            st.write("**Subject: Computer Networks**")
            st.write("Which protocol uses both TCP and UDP?")
            with st.expander("Show Answer"):
                st.success("**DNS (Domain Name System)** uses UDP for fast queries and TCP for zone transfers.")
                
        with col_dash2:
            st.warning("#### ⏳ Urgent Scholarship Deadlines")
            st.write("National Scholarship Portal (NSP) and State Scholarship Portal (SSP):")
            st.write("- **NSP Post-Matric (Engineering):** Closing in 14 Days")
            st.write("- **SSP Fee Reimbursement:** Closing in 30 Days")
            if st.button("Apply Now (External link)"):
                st.markdown("[Go to NSP Portal](https://scholarships.gov.in/)")
                
        st.markdown("---")
        st.write("### 🧠 Quick AI Assistance")
        st.write("Have an urgent question about your engineering degree? Ask our Gemini AI Assistant directly from the dashboard!")
        quick_prompt = st.text_input("Ask a quick question (or go to the AI Counselor tab for full chat):", placeholder="E.g., What are the best electives for a Software Engineering major?")
        if quick_prompt:
            st.info("Switch to the '🤖 AI Counselor' tab to continue this conversation in depth!")

        # Data Preview & Tools
        with st.expander("Explore Raw Dataset & Distributions"):
            st.dataframe(df.head(), use_container_width=True)
            if "Previous qualification (grade)" in df.columns:
                st.write("### Grade vs Outcome")
                grade_chart = alt.Chart(df).mark_boxplot(color="#0ea5e9").encode(
                    x=alt.X("target:N", title="Outcome"),
                    y=alt.Y("Previous qualification (grade):Q", title="Previous Grade"),
                ).properties(height=300)
                st.altair_chart(grade_chart, use_container_width=True)

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
                    grade = st.slider("Previous Qualification Grade", 0.0, 200.0, 120.0, 1.0)
                    age = st.slider("Age at Enrollment", 15, 60, 20)
                    submitted = st.form_submit_button("Predict Outcome")
                    
                    if submitted:
                        st.success("Prediction logic would execute here using your inputs!")
                        
            with col_viz:
                st.write("##### Dynamic Probability Gauge")
                # Visualizing the input dynamically as a pseudo-gauge
                base_prob = min(80, max(20, (grade / 200) * 100)) # Simple simulated prob
                gauge_data = pd.DataFrame({
                    "Outcome": ["Graduation Probability", "Risk Factor"],
                    "Percentage": [base_prob, 100 - base_prob]
                })
                
                gauge_chart = alt.Chart(gauge_data).mark_arc(innerRadius=50).encode(
                    theta="Percentage:Q",
                    color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Graduation Probability", "Risk Factor"], range=["#0ea5e9", "#ef4444"]))
                ).properties(height=300)
                
                st.altair_chart(gauge_chart, use_container_width=True)

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
        st.write("### 🚀 GATE Prep & Technical Gauntlet")
        
        col_gk1, col_gk2 = st.columns(2)
        with col_gk1:
            st.info("**Fact of the Day:** The concept of 'Zero' as a number was first fully developed in India by Brahmagupta around 628 AD, an essential pillar for modern binary logic and Computer Science!")
            st.success("**Tech Fact:** The Indian IT industry is projected to hit $350 billion in revenue by 2030, driven by AI, cloud computing, and massive global demand.")
            
            st.markdown("---")
            st.write("#### 📝 Official GATE Mock Tests")
            st.write("Practice with the official NPTEL simulated tests for GATE 2027.")
            if st.button("Take Sunday Mock Test (External)"):
                st.markdown("[Go to NPTEL GATE Portal](https://gate.nptel.ac.in/)")
            
        with col_gk2:
            st.write("#### The Daily 5-Question GATE Mini-Mock")
            st.write("Test your core engineering concepts!")
            
            with st.form("gate_quiz_form"):
                q1 = st.radio("1. [Databases] Which normal form eliminates transitive dependencies?", 
                                       ["1NF", "2NF", "3NF", "BCNF"], 
                                       index=None)
                                       
                q2 = st.radio("2. [OS] Which scheduling algorithm is optimal for minimizing average waiting time?",
                                       ["FCFS", "SJF (Shortest Job First)", "Round Robin", "Priority Scheduling"],
                                       index=None)
                                       
                q3 = st.radio("3. [Algorithms] What is the average-case time complexity of QuickSort?",
                                       ["O(N)", "O(N log N)", "O(N^2)", "O(log N)"],
                                       index=None)
                                       
                q4 = st.radio("4. [Network] At which OSI layer does an IP Router operate?",
                                       ["Physical Layer", "Data Link Layer", "Network Layer", "Transport Layer"],
                                       index=None)
                                       
                q5 = st.radio("5. [Digital Logic] A NAND gate is equivalent to an OR gate with...",
                                       ["Inverted inputs", "Inverted output", "Both", "None"],
                                       index=None)
                                       
                submit_quiz = st.form_submit_button("Submit Answers")
                
                if submit_quiz:
                    score = 0
                    if q1 == "3NF": score += 1
                    if q2 == "SJF (Shortest Job First)": score += 1
                    if q3 == "O(N log N)": score += 1
                    if q4 == "Network Layer": score += 1
                    if q5 == "Inverted inputs": score += 1
                        
                    if score == 5:
                        st.balloons()
                        st.success(f"**Perfect Score! You are GATE ready! {score}/5**")
                        # Gamification Reward
                        st.session_state['xp'] += 100
                        if st.session_state['xp'] >= (st.session_state['level'] * 100):
                            st.session_state['xp'] = st.session_state['xp'] - (st.session_state['level'] * 100)
                            st.session_state['level'] += 1
                            st.sidebar.success(f"Level Up! You are now Level {st.session_state['level']}!")
                    elif score >= 3:
                        st.info(f"**Great Job! {score}/5**. Keep practicing!")
                    else:
                        st.warning(f"**Total Score: {score}/5**. Review the concepts and try again!")

    with tab5:
        st.write("### 🧭 Indian Career & AI Placement Roadmaps")
        st.write("Select your domain to see the most popular degree paths, entrance exams, and AI-generated roadmaps.")
        
        stream = st.selectbox("Select Your Focus Area:", ["Computer Science & Business Systems (CSBS)", "Science (PCM)", "Science (PCB)", "Commerce", "Arts/Humanities"])
        
        col_c1, col_c2 = st.columns(2)
        if stream == "Computer Science & Business Systems (CSBS)":
            with col_c1:
                st.info("**Top Placements (2026):** \n\n* Cloud DevOps Engineer\n* Cyber Security Analyst\n* AI/ML Data Scientist\n* Enterprise Architect")
                st.warning("**High-Weightage VTU Subjects:** \n\n* Discrete Mathematics\n* Operating Systems\n* Data Structures & Algorithms")
            with col_c2:
                st.write("#### 🗺️ AI-Generated Roadmaps")
                with st.expander("🚀 DevOps Engineer Roadmap"):
                    st.write("**Months 1-2:** Linux Administration (Bash/Shell) & Git/GitHub")
                    st.write("**Months 3-4:** Networking Basics & AWS/Azure Cloud Practitioner")
                    st.write("**Months 5-6:** CI/CD (Jenkins, GitHub Actions) & Docker/Kubernetes")
                    st.write("**Months 7-8:** Infrastructure as Code (Terraform) & Monitoring (Prometheus/Grafana)")
                    if st.button("Ask AI about DevOps"):
                        st.info("Switch to the '🤖 AI Counselor' tab and ask: 'How do I start with Terraform for DevOps?'")
                        
                with st.expander("🛡️ Cyber Security Roadmap"):
                    st.write("**Months 1-2:** CompTIA Security+ Basics & Networking Protocols (TCP/IP)")
                    st.write("**Months 3-4:** Python Scripting & Ethical Hacking (Kali Linux, Nmap, Burp Suite)")
                    st.write("**Months 5-6:** Web Application Security (OWASP Top 10) & Penetration Testing")
                    st.write("**Months 7-8:** Incident Response & Security Information and Event Management (SIEM)")
        elif stream == "Science (PCM)":
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
                
        st.markdown("---")
        st.error("#### 📢 VTU Vice Chancellor Grievance Portal")
        st.write("Facing issues with VTU administration or exams? Reach out directly via the e-Vidhyarthi Mithra portal.")
        if st.button("Open e-Vidhyarthi Mithra"):
            st.markdown("[Go to Grievance Portal](https://vtu.ac.in/en/e-vidhyarthi-mithra/)")

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
        st.write("### 🧠 SIT AI Academic Mentor")
        st.markdown("Your specialized assistant for VTU academic updates, SIT campus resources, and GATE 2027 preparation strategy.")
        
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
                    
                    # System instructions for Gemini 2.5 Flash
                    system_instruction = "You are a specialized academic mentor for students at Srinivas Institute of Technology (SIT), Mangaluru. Provide expert advice on VTU 2022/2026 schemes, GATE 2027 preparation (focus on CSBS/CS subjects), and Karnataka scholarship deadlines (SSP/NSP). Keep responses professional, student-centric, and highly actionable."
                    model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
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

    with tab10:
        st.write("### 🧘 Zen Study Hub")
        st.markdown("Take a deep breath. Filter out distractions and focus on your academic goals.")
        
        col_zen1, col_zen2 = st.columns([2, 1])
        with col_zen1:
            st.write("#### ⏳ Pomodoro Focus Timer")
            st.write("Select an ambient soundtrack and start your session.")
            
            # Simple JS-based Pomodoro Timer with Audio
            pomodoro_html = '''
            <div style="background-color: #0077B6; padding: 20px; border-radius: 10px; text-align: center; color: white; font-family: sans-serif;">
                <h1 id="timer" style="font-size: 4rem; margin: 10px;">25:00</h1>
                <button onclick="startTimer()" style="background-color: #00B4D8; border: none; padding: 10px 20px; color: white; border-radius: 5px; cursor: pointer; font-size: 1.1rem; margin-right: 10px;">Start</button>
                <button onclick="resetTimer()" style="background-color: #03045E; border: none; padding: 10px 20px; color: white; border-radius: 5px; cursor: pointer; font-size: 1.1rem;">Reset</button>
                <div style="margin-top: 20px;">
                    <label>Ambient Sound: </label>
                    <select id="sound" onchange="changeSound()" style="padding: 5px; border-radius: 5px;">
                        <option value="none">None</option>
                        <option value="https://cdn.pixabay.com/download/audio/2021/08/04/audio_0625c1539c.mp3">Rain</option>
                        <option value="https://cdn.pixabay.com/download/audio/2021/08/09/audio_f5f6244ab7.mp3">Forest</option>
                        <option value="https://cdn.pixabay.com/download/audio/2022/03/17/audio_44ee90dabc.mp3">Temple Bells</option>
                    </select>
                </div>
                <audio id="ambientPlayer" loop></audio>
                
                <script>
                    let timeLeft = 1500; // 25 minutes
                    let timerInterval;
                    
                    function updateDisplay() {
                        let minutes = Math.floor(timeLeft / 60);
                        let seconds = timeLeft % 60;
                        document.getElementById('timer').innerText = 
                            (minutes < 10 ? '0' : '') + minutes + ':' + 
                            (seconds < 10 ? '0' : '') + seconds;
                    }
                    
                    function startTimer() {
                        if(timerInterval) clearInterval(timerInterval);
                        let player = document.getElementById('ambientPlayer');
                        if(player.src) player.play();
                        
                        timerInterval = setInterval(() => {
                            timeLeft--;
                            updateDisplay();
                            if(timeLeft <= 0) {
                                clearInterval(timerInterval);
                                alert("Time for a 5 minute break!");
                            }
                        }, 1000);
                    }
                    
                    function resetTimer() {
                        clearInterval(timerInterval);
                        timeLeft = 1500;
                        updateDisplay();
                        document.getElementById('ambientPlayer').pause();
                    }
                    
                    function changeSound() {
                        let player = document.getElementById('ambientPlayer');
                        let sound = document.getElementById('sound').value;
                        if(sound !== 'none') {
                            player.src = sound;
                            if(timerInterval) player.play();
                        } else {
                            player.src = '';
                            player.pause();
                        }
                    }
                </script>
            </div>
            '''
            st.components.v1.html(pomodoro_html, height=250)
            
            st.markdown("---")
            st.write("#### 🧠 AI Stress Relief Mentor")
            st.write("Feeling overwhelmed by operating systems or discrete math? Ask for a quick mental reset.")
            if st.button("Generate 5-Minute Stress Relief Exercise"):
                gemini_key = st.secrets.get("GEMINI_API_KEY")
                if gemini_key:
                    with st.spinner("Generating exercise..."):
                        try:
                            genai.configure(api_key=gemini_key)
                            model = genai.GenerativeModel("gemini-2.5-flash")
                            response = model.generate_content("Provide a calming, highly specific 5-minute stress relief exercise (breathing, stretching, or mental visualization) designed specifically for an overwhelmed engineering student. Keep it short, compassionate, and actionable.")
                            st.success(response.text)
                        except Exception as e:
                            st.error(f"Failed to fetch exercise: {e}")
                else:
                    st.error("Missing GEMINI_API_KEY. Add it to `.streamlit/secrets.toml`!")

        with col_zen2:
            st.warning("#### ⏰ Exam Alarm")
            st.write("Set an upcoming study block.")
            alarm_time = st.time_input("Target Focus Time")
            if st.button("Save Alarm"):
                st.session_state['exam_alarm'] = alarm_time
                st.success(f"Alarm locked for {alarm_time.strftime('%I:%M %p')}!")
                
            if 'exam_alarm' in st.session_state:
                st.info(f"**Active Schedule:** Next focus block at {st.session_state['exam_alarm'].strftime('%I:%M %p')}.")

            st.markdown("---")
            st.info("#### 🙏 Daily Gratitude")
            st.write("What are you thankful for today?")
            
            if 'gratitude' not in st.session_state:
                st.session_state['gratitude'] = ""
                
            gratitude_input = st.text_area("Log your thoughts:", value=st.session_state['gratitude'], height=100)
            if st.button("Save Gratitude"):
                st.session_state['gratitude'] = gratitude_input
                st.success("Your log has been saved safely!")
                
    # Professional Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.9em; padding: 20px;">
            <p><strong>SIT Student Success & Wellness Hub v1.0</strong> | Built for Srinivas Institute of Technology</p>
            <p>Powered by Streamlit & Google Gemini Pro</p>
        </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading data: {e}")
