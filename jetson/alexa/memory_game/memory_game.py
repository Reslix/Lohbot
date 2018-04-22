import json
import logging
import math
import random
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session

"""
A simple memory game

Alexa will say a word, and you (the user) respond if you've heard it before or not
50% chance of saying word already heard before, 50% chance of saying old word

Flask session attributes:
words_to_say -- list of words to say, randomized for each session
words_said_count -- number of words in words_to_say that have been spoken (0:words_said_count)
this_word -- previous word spoken
"""


app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)


@ask.launch
def new_game():
    """Starts memory game.

    Generates words_to_say from JSON file of categories.
    """

    words_to_say = []

    with open('wordlist.json', "r") as file:
        data = json.load(file)
    wordlist = data["wordlist"]
    for category in wordlist:
        words_in_category = category["words"]
        words_to_say.extend(words_in_category)

    # Randomize order of words
    random.shuffle(words_to_say)
    # print(words_to_say)
    session.attributes['words_to_say'] = words_to_say
    session.attributes['words_said_count'] = 0

    welcome_msg = render_template('welcome')
    return question(welcome_msg)


@ask.intent("AMAZON.YesIntent")
def said_yes():
    return next_round(True)


@ask.intent("AMAZON.NoIntent")
def said_no():
    return next_round(False)


def next_round(user_said_yes):
    """Returns a response either ending the game or saying the next word

    user_said_yes -- False if user said No, True if user said Yes
    """

    words_said = session.attributes['words_said_count']
    words_to_say_count = len(session.attributes['words_to_say'])

    # Check if user said correct yes/no response
    if words_said > 0:
        already_said = session.attributes['this_word'] in session.attributes['words_to_say'][0:words_said]
        if (already_said and not user_said_yes) or (not already_said and user_said_yes):
            msg = render_template('lose', count=words_to_say_count)
            return statement(msg)

    # Check if we've gone through all the words
    if words_said == words_to_say_count:
        msg = render_template('win', count=words_to_say_count)
        return statement(msg)

    # Say next word
    r = random.randint(0, 2) # 0 or 1
    if not r or words_said == 0 or words_said == words_to_say_count:
        # Say word not heard before
        word = session.attributes['words_to_say'][words_said]
        session.attributes['words_said_count'] = words_said + 1
    else:
        # Say word heard before
        r = random.randint(0, words_said)
        word = session.attributes['words_to_say'][r]
    session.attributes['this_word'] = word
    return question(word)


@ask.intent('AMAZON.StartOverIntent')
def start_over():
    return new_game()


@ask.intent('AMAZON.StopIntent')
def stop():
    words_to_say_count = len(session.attributes['words_to_say']) - 1
    bye_msg = render_template('bye', count=words_to_say_count)
    return statement(bye_msg)


@ask.intent('AMAZON.CancelIntent')
def cancel():
    words_to_say_count = len(session.attributes['words_to_say']) - 1
    bye_msg = render_template('bye', count=words_to_say_count)
    return statement(bye_msg)


@ask.session_ended
def session_ended():
    return "{}", 200
