import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

loaded_model = pickle.load(open("D:/ML PROJECTS/spam mail prediction/trained_mailspam.pkl",'rb'))
feature_extraction = pickle.load(open("D:/ML PROJECTS/spam mail prediction/feature_extraction.pkl",'rb'))


def spam_mail_pred(input_mail):
    # convert text to feature vectors
    input_data_features = feature_extraction.transform([input_mail])

    # making prediction

    prediction = loaded_model.predict(input_data_features)
    print(prediction)

    if (prediction[0] == 1):
        return 'Ham mail'

    else:
        return 'Spam mail'

def main():
    st.title("Spam mail Prediction")

    input_mail = st.text_input("Enter the mail")


    predict_output = ''

    if st.button('Check spam or Ham'):
        predict_output = spam_mail_pred(input_mail)

    st.success(predict_output)

if __name__ == '__main__':
    main()