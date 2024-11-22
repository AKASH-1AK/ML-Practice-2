import numpy as np
import pickle

loaded_model = pickle.load(open("D:/ML PROJECTS/insurance cost prediction/insurance_trained_model.pkl",'rb'))

input_data = (31,1,25.74,0,1,0)

# changing input to array
inp_dat_arr = np.asarray(input_data)

# reshaping the data
inp_data_reshape = inp_dat_arr.reshape(1,-1)

pred = loaded_model.predict(inp_data_reshape)
print("The predicted insurance amount is:",pred)