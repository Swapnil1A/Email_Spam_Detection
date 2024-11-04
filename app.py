import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()




def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text) #It will break the sent. into words..

    y=[]
    for i in text:
        if i.isalnum(): #alphanumeric
            y.append(i)
    
    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
                     


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS spam Classifier")

input_sms=st.text_area("Enter the Message")

if st.button('Predict'):


##1. Preproccessing
    transform_sms=transform_text(input_sms)
    ##2. Vectorize
    vector_input=tfidf.transform([transform_sms])
    ##3. predict
    result=model.predict(vector_input)[0] ## I would get 1 or 0 as a result from which I would extract the zeroth element/item
    #4. display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")