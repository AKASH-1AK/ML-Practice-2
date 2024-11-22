import pickle
import numpy as np

loaded_model = pickle.load(open("D:/ML PROJECTS/Parkinsons deisease pred/parkinson_trained_model.pkl",'rb'))
scaler_model = pickle.load(open("D:/ML PROJECTS/Parkinsons deisease pred/parkinsontrainedscaler.pkl",'rb'))

input_data = (88.33300,112.24000,84.07200,0.00505,0.00006,0.00254,0.00330,0.00763,0.02143,0.19700,0.01079,0.01342,0.01892,0.03237,0.01166,21.11800,0.611137,0.776156,-5.249770,0.391002,2.407313,0.249740)

# Changing input data to a numpy array
input_data_as_nparray = np.asarray(input_data)

# reshaping the numpy array:
input_data_reshaped = input_data_as_nparray.reshape(1,-1)

#Starndardizing the data :
std_data = scaler_model.transform(input_data_reshaped)

# Fitting this in to model:
prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print("The person is not an parkinson patient")

else:
    print("The person has parkinson")