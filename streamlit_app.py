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
    csv_path = r"c:\Users\DEEKSHA\Downloads\archive (2)\students_dropout_academic_success.csv"
    df = pd.read_csv(csv_path)
    return df

try:
    df = load_data()
    
    tab1, tab2 = st.tabs(["Dashboard", "Predictor Page"])
    
    with tab1:
        st.write("### Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
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
        st.info("The model connection is currently stubbed. You can plug in your Hugging Face space URL later.")
        
        @st.cache_resource
        def load_model_from_hf():
            """
            Stub function to load a model from Hugging Face Hub.
            Replace the 'REPO_ID' and 'MODEL_FILENAME' with your actual values once the space is created.
            """
            try:
                from huggingface_hub import hf_hub_download
                import joblib
                
                # NOTE: Replace with actual Hugging Face Space details
                REPO_ID = "YOUR-USERNAME/YOUR-SPACE-NAME"  
                MODEL_FILENAME = "model.joblib"
                
                if REPO_ID == "YOUR-USERNAME/YOUR-SPACE-NAME":
                    st.warning("Running with stubbed model connection. Returning a dummy predictor.")
                    # Return a dummy model/function for the layout
                    def dummy_predict(*args):
                        return ["Graduate"]
                    return dummy_predict
                
                with st.spinner("Downloading model from Hugging Face..."):
                    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
                    model = joblib.load(model_path)
                return model
                
            except Exception as model_error:
                st.error(f"Error loading model from Hugging Face: {model_error}")
                return None
                
        model = load_model_from_hf()
        
        if model is not None:
            st.write("#### Enter Student Details")
            with st.form("prediction_form"):
                grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=20.0, value=12.0)
                age = st.number_input("Age at Enrollment", min_value=15, max_value=60, value=20)
                submitted = st.form_submit_button("Predict Outcome")
                
                if submitted:
                    st.success("Prediction logic goes here.")

except Exception as e:
    st.error(f"Error loading data: {e}")
