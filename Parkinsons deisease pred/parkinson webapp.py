import numpy as np
import pickle
import streamlit as st

# Load the pre-trained Parkinson's disease model
loaded_model = pickle.load(open("D:/ML PROJECTS/Parkinsons deisease pred/parkinson_trained_model.pkl",'rb'))
scaler_model = pickle.load(open("D:/ML PROJECTS/Parkinsons deisease pred/parkinsontrainedscaler.pkl",'rb'))
# Streamlit app title
st.title('Parkinson’s Disease Prediction App')

# User inputs for each feature
mdvp_fo_hz = st.number_input("MDVP:Fo(Hz)", min_value=0.0, format="%.2f", value=119.99)
mdvp_fhi_hz = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, format="%.2f", value=157.30)
mdvp_flo_hz = st.number_input("MDVP:Flo(Hz)", min_value=0.0, format="%.2f", value=109.40)
mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, format="%.5f", value=0.00284)
mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, format="%.5f", value=0.00003)
mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, format="%.5f", value=0.00155)
mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, format="%.5f", value=0.00182)
jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, format="%.5f", value=0.00465)
mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, format="%.5f", value=0.02758)
mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, format="%.5f", value=0.282)
mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, format="%.5f", value=0.01608)
shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, format="%.5f", value=0.04756)
nhr = st.number_input("NHR", min_value=0.0, format="%.5f", value=0.02211)
hnr = st.number_input("HNR", min_value=0.0, format="%.2f", value=21.033)
rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, format="%.5f", value=0.41888)
dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, format="%.5f", value=0.81527)
spread1 = st.number_input("Spread1", format="%.5f", value=-4.813031)
spread2 = st.number_input("Spread2", format="%.5f", value=0.266482)
d2 = st.number_input("D2", format="%.5f", value=2.301442)
ppe = st.number_input("PPE", format="%.5f", value=0.123654)

# Assuming two additional missing features, add them here as placeholders
# Replace with actual feature names and value ranges
feature1 = st.number_input("Feature1 (Missing Feature 1)", min_value=0.0, format="%.5f", value=0.0)
feature2 = st.number_input("Feature2 (Missing Feature 2)", min_value=0.0, format="%.5f", value=0.0)

# Combine user input into a single array (now with 22 features)
input_data = (mdvp_fo_hz, mdvp_fhi_hz, mdvp_flo_hz, mdvp_jitter_percent, mdvp_jitter_abs,
              mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, mdvp_apq,
              shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe, feature1, feature2)

# Convert to numpy array and reshape to the model input format
input_data_as_numpy_arr = np.asarray(input_data)
input_reshaped = input_data_as_numpy_arr.reshape(1, -1)

# Prediction button
if st.button('Predict'):
    prediction = loaded_model.predict(input_reshaped)

    # Display prediction result
    if prediction[0] == 0:
        st.success('The person is not likely to have Parkinson’s disease')
    else:
        st.error('The person is likely to have Parkinson’s disease')
