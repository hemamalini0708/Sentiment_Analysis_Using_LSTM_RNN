    import flask
from flask import Flask,render_template,request
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Activation,Flatten,LSTM,Bidirectional,SimpleRNN,Embedding,Masking
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid,relu,softmax,tanh
lem = WordNetLemmatizer()
dic_size = 2000
labels = ['negative','positve','neutral','irrelavent']
import numpy as np
import pickle

with open('review.pkl','rb') as f:
    m = pickle.load(f)


app = Flask(__name__)


@app.route('/')
def fun():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def fun2():
    if request.method == 'POST':
        text = request.form["message"]
        t = text.lower()
        text = ''.join([i for i in text if i not in string.punctuation])
        text = ' '.join([lem.lemmatize(i) for i in text.split() if i not in stopwords.words('english')])
        v = [one_hot(i,dic_size) for i in [text]]
        p = pad_sequences(v,maxlen=163,padding='post')
        result = labels[np.argmax(m.predict(p))]
        return render_template('index.html',prediction_text = result)

if __name__ == '__main__':
    app.run(debug=True)