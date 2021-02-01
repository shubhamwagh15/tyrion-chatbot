#from chatbot import chatbot
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# Libraries needed for Tensorflow processing
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import json
#from keras.models import load_model
data_file = open('tintents.json').read()
intents = json.loads(data_file)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle


from tensorflow.keras.models import load_model
model = load_model('tyrion_model.h5')
import json
import random
intents = json.loads(open('tintents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    def chat(sentence):
        show_details = False
        # sentence = 'sentence'
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)

        res = model.predict(np.array([bag]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

        ints = return_list
        tag = ints[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if (i['tag'] == tag):
                result = random.choice(i['responses'])

        return result

    result = chat(userText)
    return str(result)


if __name__ == "__main__":
    app.run()
