import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model
loaded_model = pickle.load(open("D:/ML PROJECTS/heart dis prediction/heartdismodel.pkl", 'rb'))

# Streamlit app title
st.title('Heart Disease Prediction App')

# User inputs for each feature
age = st.number_input("Age", min_value=0, max_value=120, value=62)
sex = st.number_input("Sex (1 = Male, 0 = Female)", min_value=0, max_value=1, value=0)
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, value=140)
chol = st.number_input("Cholesterol (chol)", min_value=0, value=268)
fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", min_value=0, max_value=1, value=0)
restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, value=160)
exang = st.number_input("Exercise Induced Angina (1 = Yes, 0 = No)", min_value=0, max_value=1, value=0)
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, value=3.6, format="%.2f")
slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=0)
ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=2)
thal = st.number_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", min_value=1, max_value=3,
                       value=2)

# Combine user input into a single array
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

# Convert to numpy array and reshape to the model input format
input_data_as_numpy_arr = np.asarray(input_data)
input_reshaped = input_data_as_numpy_arr.reshape(1, -1)

# Prediction button
if st.button('Predict'):
    prediction = loaded_model.predict(input_reshaped)

    # Display prediction result
    if prediction[0] == 0:
        st.success('The person is not a heart disease patient')
    else:
        st.error('The person is a heart disease patient')
