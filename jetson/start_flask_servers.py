import argparse
from flask_servers import FlaskServerStarter


parser = argparse.ArgumentParser(description='Start a Flask server to listen to Alexa commands.')
parser.add_argument('server_number', type=int,
            help='a number for which server to start')
args = parser.parse_args()
s = args.server_number
print(args.server_number)

FlaskServerStarter.start(s)
