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
        return [float(x) for x in value]
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
                data[col] = data[col].astype(dtype)
                
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

def predict_future_score(historical_scores, current_score, attendance, reading, writing, grade):
    # Create extended history
    full_history = historical_scores + [current_score]
    
    # Calculate trend-based prediction
    trend = calculate_trend(full_history)
    trend_prediction = current_score + trend
    
    # Calculate average-based prediction
    avg_prediction = np.mean(full_history[-3:] + [current_score])
    
    # Create feature vector
    features = [
        np.mean(full_history),
        trend,
        current_score,
        attendance,
        reading,
        writing,
        ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F'].index(grade)
    ]
    
    # Blend predictions (you can adjust these weights)
    return 0.6 * trend_prediction + 0.3 * avg_prediction + 0.1 * (reading + writing)

# ======================
# STREAMLIT UI
# ======================

def main():
    st.title("📈 Student Performance Predictor")
    
    # Data loading section
    st.sidebar.header("1. Upload Historical Data (Optional)")
    uploaded_file = st.sidebar.file_uploader("CSV with student patterns", type="csv")
    
    # Prediction section
    st.header("2. Student Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            prev_scores = st.text_input("Previous Scores (comma-separated)", "78, 80, 82, 85")
            current_perc = st.number_input("Current Percentage (%)", 0.0, 100.0, 85.0)
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 90.0)
            
        with col2:
            reading = st.number_input("Reading Score (0-10)", 0.0, 10.0, 8.0)
            writing = st.number_input("Writing Score (0-10)", 0.0, 10.0, 9.0)
            grade = st.selectbox("Current Grade", ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F'])
        
        if st.form_submit_button("Predict"):
            try:
                # Process inputs
                historical = [float(x.strip()) for x in prev_scores.split(',')]
                
                # Make prediction
                prediction = predict_future_score(
                    historical_scores=historical,
                    current_score=current_perc,
                    attendance=attendance,
                    reading=reading,
                    writing=writing,
                    grade=grade
                )
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                # Plotting
                fig, ax = plt.subplots()
                history = historical + [current_perc]
                ax.plot(history, marker='o', label='Historical Performance')
                ax.plot(len(history), prediction, 'ro', markersize=8, label='Predicted Score')
                ax.set_title('Academic Performance Trend')
                ax.set_xlabel('Term')
                ax.set_ylabel('Score (%)')
                ax.legend()
                st.pyplot(fig)
                
                # Feedback
                st.subheader("📝 Improvement Suggestions")
                
                if attendance < 75:
                    st.error("⚠️ Attendance Alert: Regular attendance below 75% significantly impacts learning outcomes.")
                elif attendance < 85:
                    st.warning("🗓️ Attendance Note: Try to maintain at least 85% attendance for better results")
                
                if reading < 7:
                    st.warning("📚 Reading Recommendation: Practice daily reading comprehension exercises")
                if writing < 7:
                    st.warning("✍️ Writing Suggestion: Focus on structured writing practice weekly")
                
                performance_change = prediction - current_perc
                if performance_change > 2:
                    st.success(f"🚀 Positive Trend! Predicted improvement of {performance_change:.1f}%")
                elif performance_change > 0:
                    st.info(f"📈 Steady Progress: Expected small improvement of {performance_change:.1f}%")
                else:
                    st.warning("🔍 Attention Needed: Consider additional academic support")
                
                st.markdown(f"""
                **Predicted Next Term Score:**  
                <span style="font-size:24px; color:green">{prediction:.1f}%</span>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing input: {str(e)}")

if __name__ == "__main__":
    main()