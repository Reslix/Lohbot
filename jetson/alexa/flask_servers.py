import argparse
import datetime
import json
import logging
import math
import random
import time

import dateutil.parser
import fasteners
from flask import Flask, redirect, render_template, url_for
from flask_ask import Ask, statement, question, session
import gunicorn.app.base
from gunicorn.six import iteritems

lock_file_name = 'ALEXA_COMMAND.txt.lock'
file_name = 'ALEXA_COMMAND.txt'


#############################################################################
# Flask server to handle request for movement from the Alexa
#############################################################################

movement_app = Flask(__name__)
movement_ask = Ask(movement_app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

@movement_ask.intent("TurnIntent", mapping={'direction': 'Direction'})
def turn(direction):
    with fasteners.InterProcessLock(lock_file_name):
        with open(file_name, "w+") as file:
            file.write(direction.lower())
    return statement(render_template('turn', direction=direction))

@movement_ask.intent("StartMovingIntent")
@fasteners.interprocess_locked(lock_file_name)
def start_moving():
    with open(file_name, "w+") as file:
        file.write("forward")
    return statement(render_template('move_forward'))

@movement_ask.intent("StopIntent")
@fasteners.interprocess_locked(lock_file_name)
def stop_moving():
    with open(file_name, "w+") as file:
        file.write("stop")
    return statement(render_template('stop'))

@movement_ask.intent("FollowIntent")
@fasteners.interprocess_locked(lock_file_name)
def follow():
    with open(file_name, "w+") as file:
        file.write("follow")
    return statement(render_template('follow'))


#############################################################################
# A simple memory game

# Alexa will say a word, and you (the user) respond if you've heard it before
# or not
# 1/3 chance of saying word already said before, 2/3 chance of saying new word

# Flask session attributes:
# words_to_say -- list of words to say, randomized for each session
# words_already_said -- list of words already said (check for duplicates before adding)
# this_word -- previous word spoken
#############################################################################


results_file = 'alexa/memory_game_results.json'

memory_game_app = Flask(__name__)
memory_game_ask = Ask(memory_game_app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)


# From https://stackoverflow.com/a/27058505
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


def record_results():
    with open(results_file, "r") as file:
        try:
            data = json.load(file)
        except ValueError:
            print("Filling in file for results")
            data = {}
    if not "results" in data:
        data["results"] = []
    results = data["results"]
    words_said_count = len(session.attributes['words_already_said'])
    results.append({"time": datetime.datetime.now(), "correct": words_said_count})
    with open(results_file, 'w') as outfile:
        json.dump(data, outfile, cls=DateTimeEncoder)

@memory_game_app.template_filter('format_datetime')
def format_datetime(s):
    """
    :param s: string datetime in ISO format
    """
    return dateutil.parser.parse(s).strftime("%B %d, %Y %H:%M")

@memory_game_app.route("/results")
def show_results():
    with open(results_file, "r") as file:
        try:
            data = json.load(file)
        except ValueError:
            print("Data file is empty")
            data = {}
        if not "results" in data:
            results = []
        else:
            results = data["results"]
        return render_template('memory_game_results.html', results=results)

@memory_game_app.route("/clear_results", methods=['POST'])
def clear_results():
    with open(results_file, "w+") as file:
        file.write("{}")
    return redirect(url_for("show_results"))

@memory_game_ask.launch
def new_game():
    """Starts memory game.

    Generates words_to_say from JSON file of categories
    (selects half the words in each category).
    """

    words_to_say = []

    with open('alexa/wordlist.json', "r") as file:
        data = json.load(file)
    wordlist = data["wordlist"]
    for category in wordlist:
        words_in_category = category["words"]
        length = int(math.ceil(len(words_in_category)/2.0))
        words_to_say.extend(random.sample(words_in_category, length))

    # Randomize order of words
    random.shuffle(words_to_say)
    # print(words_to_say)
    session.attributes['words_to_say'] = words_to_say
    session.attributes['words_already_said'] = []

    welcome_msg = render_template('welcome')
    return question(welcome_msg)


@memory_game_ask.intent("AMAZON.YesIntent")
def said_yes():
    return next_round(True)


@memory_game_ask.intent("AMAZON.NoIntent")
def said_no():
    return next_round(False)


def next_round(user_said_yes):
    """Returns a response either ending the game or saying the next word

    :param user_said_yes: False if user said No, True if user said Yes
    """

    words_said_count = len(session.attributes['words_already_said'])
    words_to_say_count = len(session.attributes['words_to_say'])

    # Check if user said correct yes/no response
    if 'this_word' in session.attributes:
        already_said = session.attributes['this_word'] in session.attributes['words_already_said']
        if (already_said and not user_said_yes) or (not already_said and user_said_yes):
            msg = render_template('lose', count=words_said_count)
            record_results()
            return statement(msg)
        # User answered correctly; add to list of words already said
        if session.attributes['this_word'] not in session.attributes['words_already_said']:
            session.attributes['words_already_said'].insert(0, session.attributes['this_word'])

    words_said_count = len(session.attributes['words_already_said'])
    # Check if we've gone through all the words
    if words_to_say_count == 0:
        msg = render_template('win', count=words_said_count)
        record_results()
        return statement(msg)

    # Say next word
    r = random.randint(0, 2) # 0, 1, or 2
    print(r)
    if r != 0 or words_said_count == 0:
        # Say word not heard before
        word = session.attributes['words_to_say'].pop()
    else:
        # Say word heard before
        word = random.sample(session.attributes['words_already_said'], 1)[0]
    session.attributes['this_word'] = word
    return question(word)


@memory_game_ask.intent('AMAZON.StartOverIntent')
def start_over():
    return new_game()


@memory_game_ask.intent('AMAZON.StopIntent')
def stop():
    return cancel()


@memory_game_ask.intent('AMAZON.CancelIntent')
def cancel():
    words_said_count = len(session.attributes['words_already_said'])
    bye_msg = render_template('bye', count=words_said_count)
    record_results()
    return statement(bye_msg)


@memory_game_ask.session_ended
def session_ended():
    return "{}", 200


#############################################################################
# Boilerplate code for starting Gunicorn
#############################################################################

def start(s):
    """
    Starts a Gunicorn server for a Flask Ask application
    :param s: integer (server number)
    """

    if s == 0:
        options = {
            'bind': '%s:%s' % ('localhost', '34443'),
            'certfile': '%s' % ('certificate.pem'),
            'keyfile': '%s' % ('private-key.pem'),
            'workers': number_of_workers(),
        }
        StandaloneApplication(movement_app, options).run()
    elif s == 1:
        options = {
            'bind': '%s:%s' % ('localhost', '11577'),
            'certfile': '%s' % ('certificate.pem'),
            'keyfile': '%s' % ('private-key.pem'),
            'workers': number_of_workers(),
        }
        StandaloneApplication(memory_game_app, options).run()
    else:
        print("Invalid server arg")


class StandaloneApplication(gunicorn.app.base.BaseApplication):
    """Class for starting custom Gunicorn WSGI application"""
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(StandaloneApplication, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])

        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def number_of_workers():
    return 1

