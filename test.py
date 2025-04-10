# app.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
import ast

# ======================
# DATA VALIDATION & PREPROCESSING
# ======================

REQUIRED_COLUMNS = {
    'Student_Name': 'object',
    'Previous Scores': 'object',
    'Current Percentage': 'float64',
    'Attendance': 'float64',
    'Reading_Score': 'float64',
    'Writing_Score': 'float64',
    'Grade': 'category'
}

def safe_convert_to_list(value):
    """Safely convert string representation of list to actual list"""
    try:
        if isinstance(value, str):
            return [float(x) for x in ast.literal_eval(value)]
        elif isinstance(value, list):
            return [float(x) for x in value]
        return [0.0]
    except:
        return [0.0]

def load_and_validate_data(filepath):
    try:
        data = pd.read_csv(filepath)

        # Check for missing columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns in CSV: {missing_cols}")
            return None

        # Convert and validate data types
        for col, dtype in REQUIRED_COLUMNS.items():
            if col == 'Previous Scores':
                data[col] = data[col].apply(safe_convert_to_list)
            else:
                data[col] = data[col].astype(dtype, errors='ignore')

        return data.dropna()

    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None

# ======================
# TREND ANALYSIS & PREDICTION
# ======================

def calculate_trend(scores):
    if len(scores) < 2:
        return 0.0
    x = np.arange(len(scores)).reshape(-1, 1)
    y = np.array(scores)
    return LinearRegression().fit(x, y).coef_[0]

def predict_future_score(student_data, current_score):
    full_history = student_data.get('Previous Scores', []) + [current_score]

    # Calculate trend-based prediction
    trend = calculate_trend(full_history)
    trend_prediction = current_score + trend * len(full_history)

    # Calculate weighted average
    weights = np.linspace(0.1, 1, len(full_history))
    weighted_avg = np.average(full_history, weights=weights)

    # Combine features
    reading_score = student_data.get('Reading_Score', 0)
    writing_score = student_data.get('Writing_Score', 0)

    return 0.5 * trend_prediction + 0.3 * weighted_avg + 0.2 * (reading_score + writing_score)

# ======================
# STREAMLIT UI
# ======================

def main():
    st.title("📈 Student Performance Analyzer Pro")

    # Data loading section
    st.sidebar.header("1. Upload Student Database")
    uploaded_file = st.sidebar.file_uploader("CSV with student records", type="csv")

    student_data = None
    if uploaded_file:
        student_db = load_and_validate_data(uploaded_file)
        if student_db is not None:
            student_names = student_db['Student_Name'].unique().tolist()
            student_names.append("New Student")

            # Prediction section
            st.header("2. Student Analysis")
            selected_student = st.selectbox("Select Student", student_names)

            if selected_student != "New Student":
                student_data = student_db[student_db['Student_Name'] == selected_student].iloc[0].to_dict()

                # Display historical data
                with st.expander("View Student History"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Previous Scores", f"{np.mean(student_data['Previous Scores']):.1f}%")
                        st.metric("Best Previous Score", f"{max(student_data['Previous Scores']):.1f}%")
                    with col2:
                        st.metric("Attendance Rate", f"{student_data['Attendance']}%")
                        literacy_score = (student_data['Reading_Score'] + student_data['Writing_Score']) / 2
                        st.metric("Literacy Score", f"{literacy_score:.1f}/10")

            # Input current performance
            with st.form("prediction_form"):
                current_perc = st.number_input(
                    "Current Term Percentage (%)", 0.0, 100.0,
                    value=student_data['Current Percentage'] if student_data else 85.0
                )

                if st.form_submit_button("Analyze Performance"):
                    try:
                        historical = student_data['Previous Scores'] if student_data else []

                        # Make prediction
                        prediction = predict_future_score(student_data if student_data else {}, current_perc)

                        # Display results
                        st.subheader("📊 Performance Analysis Report")

                        # Plotting
                        fig, ax = plt.subplots(figsize=(10, 6))
                        history = historical + [current_perc]
                        x_labels = [f"Term {i + 1}" for i in range(len(history))] + ["Prediction"]

                        ax.plot(x_labels[:-1], history, marker='o', label='Historical Performance')
                        ax.plot(x_labels[-2:], [current_perc, prediction], 'ro--', markersize=8, label='Current & Prediction')
                        ax.set_title(f'Performance Trend for {selected_student}')
                        ax.set_ylabel('Score (%)')
                        ax.legend()
                        st.pyplot(fig)

                        # Feedback analysis
                        st.subheader("📝 Personalized Recommendations")

                        # Attendance analysis
                        if student_data and student_data.get('Attendance', 100) < 75:
                            st.error("🚨 Critical Attendance: Below 75% attendance detected")
                        elif student_data and student_data.get('Attendance', 100) < 85:
                            st.warning("📉 Attendance Warning: Below optimal attendance level")

                        # Literacy analysis
                        literacy_score = (student_data.get('Reading_Score', 0) + student_data.get('Writing_Score', 0)) / 2
                        if literacy_score < 6:
                            st.error("📚 Literacy Deficiency: Immediate remediation needed")
                        elif literacy_score < 8:
                            st.warning("📖 Literacy Improvement: Focus on language skills")

                        # Trend analysis
                        avg_history = np.mean(historical) if historical else current_perc
                        trend = current_perc - avg_history

                        if trend > 5:
                            st.success(f"🌟 Outstanding Progress! +{trend:.1f}% improvement from average")
                        elif trend > 0:
                            st.info(f"📈 Positive Trend: +{trend:.1f}% improvement from average")
                        else:
                            st.warning(f"📉 Concerning Trend: {abs(trend):.1f}% drop from historical average")

                        st.markdown(f"""
                        **Predicted Next Term Score:**  
                        <span style="font-size:28px; color:green">{prediction:.1f}%</span>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()