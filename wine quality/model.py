import numpy as np
import streamlit as st
import pickle


loaded_model = pickle.load(open("D:/ML PROJECTS/wine quality/trained_model_wine.pkl",'rb'))

inp_data = (7.8,0.57,0.31,1.8,0.069,26.0,120.0,0.99625,3.29,0.53,9.3)

# converting the tuple to nparray
inp_data_as_nparray = np.asarray(inp_data)

# reshaping the data
reshaped_inp_npdata = inp_data_as_nparray.reshape(1,-1)

prediction = loaded_model.predict(reshaped_inp_npdata)

print(prediction)

if(prediction[0]==1):
    print("Good quality wine")
else:
    print("Bad quality wine")
