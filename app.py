import streamlit as st
import pandas as pd
import numpy
import joblib

def load_model():
    return joblib.load('performance_model.pkl')  # Replace with your actual model filename

def main():
    st.title("Employee Performance Prediction")

    model = load_model()

    # Input fields for the three factors
    department = st.selectbox("Select Department", ["Sales", "HR", "R&D", "Development", "Data Science", "Finance"])  # Add your departments
    satisfaction = st.slider("Employee Satisfaction (1-5)", 1, 5, 3)
    last_salary_hike = st.number_input("Last Salary Hike Percentage", min_value=0.0, value=5.0)

    if st.button("Predict Performance"):
        try:
            # Create a DataFrame from the input data
            input_data = pd.DataFrame({
                "Department": [department],
                "Satisfaction": [satisfaction],
                "LastSalaryHike": [last_salary_hike]
            })

            # One-hot encode the 'Department' column
            input_data = pd.get_dummies(input_data, columns=['Department'], drop_first=True)

            # Ensure the input data has the same columns as the model was trained on
            # This is critical.
            model_columns = model.feature_names_in_.tolist() # Assuming your model has feature_names_in_ attribute
            for col in model_columns:
                if col not in input_data.columns:
                    input_data[col] = 0 # Add missing columns with 0 values.
            input_data = input_data[model_columns] # reorder columns to match model

            # Make the prediction
            prediction = model.predict(input_data)

            st.write(f"Predicted Performance Rating: {prediction[0]}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
