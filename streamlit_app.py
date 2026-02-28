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

except Exception as e:
    st.error(f"Error loading data: {e}")
