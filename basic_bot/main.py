import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')

# Load intents and model data
intents = json.loads(open('C:/Users/Nikhil/PycharmProjects/AIproject/basic_bot/intents.json').read())
words = pickle.load(open('C:/Users/Nikhil/PycharmProjects/AIproject/basic_bot/words.pkl', 'rb'))
classes = pickle.load(open('C:/Users/Nikhil/PycharmProjects/AIproject/basic_bot/classes.pkl', 'rb'))
model = tf.keras.models.load_model('C:/Users/Nikhil/PycharmProjects/AIproject/basic_bot/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_classes(sentence):
    bag = bag_of_words(sentence)
    results = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    identified_intents = []
    for r in results:
        identified_intents.append(classes[r[0]])

    return identified_intents

def get_responses(intent_list, intents_json):
    responses = []
    list_of_intents = intents_json['intents']

    for intent in list_of_intents:
        if intent['tag'] in intent_list:
            responses.append(intent['tag'])

    return responses


def output(message):
    identified_symptoms = predict_classes(message)
    if identified_symptoms:
        return "".join(identified_symptoms)
    else:
        return message