import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open("D:/ML PROJECTS/wine quality/trained_model_wine.pkl", 'rb'))

# Creating a prediction function
def wine_quality(inp_data):
    # Convert the tuple to np array
    inp_data_as_nparray = np.asarray(inp_data)

    # Reshape the data
    reshaped_inp_npdata = inp_data_as_nparray.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(reshaped_inp_npdata)

    # Return result
    if prediction[0] == 1:
        return "Good quality wine"
    else:
        return "Bad quality wine"


# Main function for the Streamlit app
def main():
    st.title("Wine Quality Prediction")

    # Collect inputs from the user
    try:
        fixed_acidity = float(st.text_input("Enter the fixed acidity", ""))
        volatile_acidity = float(st.text_input("Enter the volatile acidity", ""))
        citric_acid = float(st.text_input("Enter the citric acid amount", ""))
        residual_sugar = float(st.text_input("Enter the residual sugar", ""))
        chlorides = float(st.text_input("Enter the chlorides", ""))
        free_sulfur_dioxide = float(st.text_input("Enter the free sulfur dioxide amount", ""))
        total_sulfur_dioxide = float(st.text_input("Enter the total sulfur dioxides", ""))
        density = float(st.text_input("Enter the density", ""))
        pH = float(st.text_input("Enter the pH level", ""))
        sulphates = float(st.text_input("Enter the sulphates", ""))
        alcohol = float(st.text_input("Enter the alcohol level", ""))
    except ValueError:
        st.error("Please enter valid numerical values.")
        return

    quality = ""

    # On button click, make prediction
    if st.button("Wine Quality Prediction"):
        quality = wine_quality((fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                               free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol))

    st.success(quality)


if __name__ == '__main__':
    main()