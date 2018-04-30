import logging
from random import randint
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session
import gunicorn.app.base
from gunicorn.six import iteritems


app = Flask(__name__)
ask = Ask(app, "/")

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


# Boilerplate code for starting Gunicorn

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


if __name__ == '__main__':
    options = {
        'bind': '%s:%s' % ('localhost', '34443'),
        'certfile': '%s' % ('certificate.pem'),
        'keyfile': '%s' % ('private-key.pem'),
        'workers': number_of_workers(),
    }
    StandaloneApplication(app, options).run()

