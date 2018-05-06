import multiprocessing
import time

from flask import Blueprint, Flask, render_template, Response, send_file
import gunicorn.app.base
from gunicorn.six import iteritems

"""
Starts a Flask server to listen to streaming requests
"""

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    if camera == None:
        return
    while True:
        frame = camera.get_jpg()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(app.config['CAMERA']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def send_status():
    return send_file("ALEXA_COMMAND.txt")

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

def start_streaming_server(c):
    """
    c -- Camera object (from camera.py)
    """
    app.config['CAMERA'] = c

    options = {
        'bind': '%s:%s' % ('localhost', '11578'),
        'certfile': '%s' % ('certificate.pem'),
        'keyfile': '%s' % ('private-key.pem'),
        'workers': number_of_workers(),
    }
    StandaloneApplication(app, options).run()

def number_of_workers():
    return (multiprocessing.cpu_count() * 2) + 1

if __name__ == "__main__":
    p = multiprocessing.Process(target = start_streaming_server, args=(None,))
    p.start()
    p.join()
