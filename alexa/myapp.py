import logging
from random import randint
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session


app = Flask(__name__)
ask = Ask(app, "/robot/")

logging.getLogger("flask_ask").setLevel(logging.DEBUG)

@ask.intent("TurnIntent", mapping={'direction': 'Direction'})
def turn(direction):
    return statement(render_template('turn', direction=direction))

@ask.intent("StartMovingIntent")
def start_moving():
    return statement(render_template('move_forward'))

@ask.intent("StopIntent")
def stop_moving():
    return statement(render_template('stop'))