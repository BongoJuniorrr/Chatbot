import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf



lemmatizer = WordNetLemmatizer()
intents = json.loads(open("./intents.json").read())

words = pickle.load(open("./words.pkl", "rb"))
classes = pickle.load(open("./classes.pkl", "rb"))
model = tf.keras.models.load_model('./chatbotmodel.h5', compile=False)
model.compile()

#Make the sentence cleaner again!
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Like function name
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#Predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        
    return return_list

#Get response
def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        DATA = []
        for i in list_of_intents:
            if i['tags'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    except:
        return ""

#Check if the answer is fit with the question
def check_answer(DATA, message):
    message = message + " "
    best_correct = 0
    must_correct = 0
    correct = 0
    for i in message:
        if (i==' '):
            must_correct += 1
    for i in DATA:
        ans = ""
        for j in i:
            if j != ' ':
                ans += j
            else:
                if ans in message:
                    correct += 1
                ans = ""
        if correct > best_correct:
            best_correct = correct
    correct_point = float(best_correct) / float(must_correct)
    #print(correct_point)
    if correct_point >= 0.5:
        return 1
    return 0

def reply(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res

while True:
    you=input("Ask: ")
    rep=reply(you)
    print("Reply: "+rep)