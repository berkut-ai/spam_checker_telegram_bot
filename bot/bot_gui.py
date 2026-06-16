from pathlib import Path
import joblib
import telebot

with open('API.txt', 'r', encoding='utf-8') as f:
    API = f.read()
bot = telebot.TeleBot(API)

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'model' / 'model.pkl'

model, vectorizer = joblib.load(MODEL_DIR)

@bot.message_handler(commands=['start'])
def start_message(mes):
    bot.send_message(mes.chat.id, "Hello, user! Send me the text you want to check.")

@bot.message_handler(content_types=['text'])
def text(mes):
    data = vectorizer.transform([mes.text])

    proba = model.predict_proba(data)[0][1]


    bot.send_message(mes.chat.id, f"I think it's{' ' if proba > 0.7 else ' not '}spam.")



bot.polling(none_stop=True)
