import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Student Success Dashboard",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Success Dashboard")
st.markdown("Explore the academic success and dropout dataset.")

@st.cache_data
def load_data():
    df = pd.read_csv("students_dropout_academic_success.csv")
   
    return df

try:
    df_raw = load_data()
    
    # Sidebar Filters
    st.sidebar.header("Filter Data")
    
    # Map binary values for user-friendliness (assuming 1=Yes/Male, 0=No/Female based on typical encodings, though exact mappings might vary in this dataset)
    gender_filter = st.sidebar.multiselect("Gender", options=df_raw["Gender"].unique(), default=df_raw["Gender"].unique())
    scholarship_filter = st.sidebar.multiselect("Scholarship holder", options=df_raw["Scholarship holder"].unique(), default=df_raw["Scholarship holder"].unique())
    displaced_filter = st.sidebar.multiselect("Displaced", options=df_raw["Displaced"].unique(), default=df_raw["Displaced"].unique())
    
    # Apply filters
    df = df_raw[
        (df_raw["Gender"].isin(gender_filter)) &
        (df_raw["Scholarship holder"].isin(scholarship_filter)) &
        (df_raw["Displaced"].isin(displaced_filter))
    ]
    
    tab1, tab2 = st.tabs(["Dashboard", "Predictor Page"])
    
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

except Exception as e:
    st.error(f"Error loading data: {e}")
