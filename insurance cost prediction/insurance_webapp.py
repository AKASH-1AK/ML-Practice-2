import numpy as np
import pickle
import streamlit as st

# Load the pre-trained insurance model
loaded_model = pickle.load(open("D:/ML PROJECTS/insurance cost prediction/insurance_trained_model.pkl", 'rb'))

# Streamlit app title
st.title('Insurance Charge Prediction App')

# User inputs for each feature
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sex = st.selectbox("Sex", options=['Male', 'Female'])
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=50.0, value=25.0, format="%.2f")
children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
smoker = st.selectbox("Smoker", options=['Yes', 'No'])
region = st.selectbox("Region", options=['northeast', 'northwest', 'southeast', 'southwest'])

# Encode categorical features
sex_encoded = 1 if sex == 'Male' else 0  # 1 = Male, 0 = Female
smoker_encoded = 1 if smoker == 'Yes' else 0  # 1 = Smoker, 0 = Non-Smoker

# Encode region feature (you can adjust based on how your model was trained)
region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
region_encoded = region_mapping[region]

# Combine user input into a single array
input_data = (age, sex_encoded, bmi, children, smoker_encoded, region_encoded)

# Convert to numpy array and reshape to the model input format
input_data_as_numpy_arr = np.asarray(input_data)
input_reshaped = input_data_as_numpy_arr.reshape(1, -1)

# Prediction button
if st.button('Predict Insurance Charges'):
    prediction = loaded_model.predict(input_reshaped)

    # Display prediction result
    st.write(f"The predicted insurance charges are: ${prediction[0]:,.2f}")
