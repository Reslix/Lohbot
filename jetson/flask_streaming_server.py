import multiprocessing
import time

import fasteners
from flask import Blueprint, Flask, render_template, Response, send_file
import gunicorn.app.base
from gunicorn.six import iteritems

from manager import ImageManager


"""
Starts a Flask server to listen to streaming requests
"""

app = Flask(__name__)

lock_file_name = 'ALEXA_COMMAND.txt.lock'
file_name = 'ALEXA_COMMAND.txt'

@app.route('/')
def index():
    return render_template('index.html')

def gen(manager):
    if manager == None:
        return
    while True:
        image_dictionary = manager.get_dict()
        # if 'encoded' in image_dictionary.keys():
        #     frame = image_dictionary.get('encoded')
        if 'camera' in image_dictionary.keys():
            camera = image_dictionary.get('camera')
            frame = camera.get_jpg()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + b'\0' + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(app.config['MANAGER']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def send_status():
    output = "Alexa command:<br> "
    with fasteners.InterProcessLock(lock_file_name):
        with open(file_name, "r") as file:
            output = output + file.read().replace('\n', '')
    manager = app.config['MANAGER']
    if manager == None:
        return output

    image_dictionary = manager.get_dict()
    if 'state' in image_dictionary.keys():
        state = image_dictionary.get('state')
        output = output + "<br> Tracker state:<br> " + state

    return output


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

def start_streaming_server():
    # SyncManager with member get_dict that is a shared Proxy Object
    # See https://docs.python.org/3.6/library/multiprocessing.html#multiprocessing.managers.SyncManager
    try:
        manager = ImageManager(address=('', 11579), authkey=b'password')
        ImageManager.register('get_dict')
        manager.connect()
        print("Connected to manager.")
        app.config['MANAGER'] = manager
    except ConnectionRefusedError:
        print("No connection to  manager.")


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
    p = multiprocessing.Process(target = start_streaming_server)
    p.start()
    p.join()
