
class SerialIO():

    def __init__(self, com=None, baud=9600, delay=10):
        pass
        # initialize data structure to keep shizz
        # put lockfile, write buffer

    def start(self):
        pass
        # start the thread that constantly does serial reading

    def read(self):
        pass
        # returns the data structure, be sure to check lockfile

    def write(self):
        pass
        # sends the write buffer, be sure to check lockfile. Returns delay between when the write was sent and when
        # this was called.

    def update(self):
        while self.running:
            pass
            # do the handshakes to read, write if necessary, then delay