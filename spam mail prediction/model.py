import numpy as np
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

loaded_model = pickle.load(open("D:/ML PROJECTS/spam mail prediction/trained_mailspam.pkl",'rb'))
feature_extraction = pickle.load(open("D:/ML PROJECTS/spam mail prediction/feature_extraction.pkl",'rb'))


input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = loaded_model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')