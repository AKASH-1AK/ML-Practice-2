import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("D:/ML PROJECTS/heart dis prediction/heartdismodel.pkl",'rb'))

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# changing the input data to array
input_data_as_numpy_arr = np.asarray(input_data)

#reshaping the numpy array:
input_reshaped = input_data_as_numpy_arr.reshape(1,-1)

prediction = loaded_model.predict(input_reshaped)
print(prediction)

if(prediction[0]==0):
    print('The person is not a heart disease patient')
else:
    print('The perosn is a heart disease patient')