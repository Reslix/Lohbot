import multiprocessing
from multiprocessing.managers import SyncManager
import time

import fasteners
from flask import Flask, render_template, Response, send_file


class ImageManager(SyncManager):
    """
    Controls access to shared dict object

    get_dict() returns the Manager.dict()
    Key     Value
    camera  Camera object (webcam image)
    state   status of tracker (string)
    encoded encoded camera image with overlay (string)
    """
    pass


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


def start_streaming_server(manager):
    """
    :param c: SyncManager with member get_dict that is a shared Proxy Object
    See https://docs.python.org/3.6/library/multiprocessing.html#multiprocessing.managers.SyncManager
    """
    app.config['MANAGER'] = manager

    app.run(host='0.0.0.0', port=11578, ssl_context = 'certificate.pem', 'private-key.pem')

if __name__ == "__main__":
    p = multiprocessing.Process(target = start_streaming_server, args=(None,))
    p.start()
    p.join()
