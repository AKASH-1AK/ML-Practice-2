import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model
loaded_model = pickle.load(open("D:/ML PROJECTS/sonar detection/sonar tained model.pkl", 'rb'))

# Streamlit app title
st.title('Rock or Mine Prediction App')

st.write("Please enter the 59 input feature values:")

# Create inputs for 59 features
feature_values = []
for i in range(0, 60):
    value = st.number_input(f"Feature {i}", format="%.5f", value=0.0)  # Default value is 0.0
    feature_values.append(value)

# Convert user inputs into a numpy array
input_data = np.asarray(feature_values)

# Reshape the input to match the model's expected input format
input_reshaped = input_data.reshape(1, -1)

# Debugging: Print the shape of the reshaped input to make sure it's correct
st.write(f"Input shape: {input_reshaped.shape}")  # It should display (1, 59)

# Prediction button
if st.button('Predict'):
    try:
        # Make prediction
        prediction = loaded_model.predict(input_reshaped)

        # Debugging: Check the prediction result
        st.write(f"Prediction result: {prediction}")

        # Display the prediction result
        if prediction[0] == 0:
            st.success('The object is predicted to be a Rock')
        else:
            st.error('The object is predicted to be a Mine')
    except ValueError as e:
        # Catch and display any errors related to input shape
        st.error(f"Error: {e}")
