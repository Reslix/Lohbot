# Install Instructions

How to set up flask-ask

## Install python and set up virtualenv on Jetson

1. Install Python: `sudo apt-get install python`
1. Install dependencies for cryptography module: `sudo apt-get install build-essential libssl-dev libffi-dev python-dev`
1. Install pipenv for all users: `sudo -H pip3 install -U pipenv`
1. Install Python packages to this virtualenv: `pipenv install`

## Set up service on Jetson

1. Activate virualenv: `pipenv shell`
1. Start the web server: `gunicorn -b 0.0.0.0:8443 myapp:app`. Ctrl+C will terminate it.
1. Download ngrok from https://ngrok.com/download
1. In another terminal, open up a localhost tunnel: `./ngrok http 8443`. Ctrl+C will terminate it. Copy the https URL into the Alexa skill (Configuration > Endpoint).
1. To exit the shell, deactivate virtualenv with `exit`

## Resources

* [Tunneling to localhost](https://blog.jayway.com/2013/10/17/tunneling-to-localhost-via-ssh/)
* [SSH Reverse Tunneling](https://juntx.wordpress.com/2014/07/28/use-amazon-ec2-and-ssh-reverse-tunneling-to-connect-computers-behind-firewall-or-nat/)