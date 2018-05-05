# Install Instructions

How to set up flask-ask running locally

## Install python and set up virtualenv on Jetson

1. Open a terminal in the folder where Pipfile.lock is (jetson).
1. Install Python: `sudo apt-get install python`
1. Install dependencies for cryptography module: `sudo apt-get install build-essential libssl-dev libffi-dev python-dev python3-pip`
1. Install pipenv for all users: `sudo -H pip3 install -U pipenv`
1. Install Python packages to this virtualenv: `pipenv install`

## Install nginx

1. Update the timezone: `sudo dpkg-reconfigure tzdata`
1. Install nginx: `sudo apt install nginx`
1. Add a user `sudo adduser flaskapp`
1. In `sudo vim /etc/nginx/nginx.conf`, Change `user  www-data;` to `user  flaskapp`. In the http block, uncomment this line: `server_names_hash_bucket_size 64;` and change `64` to `128`
1. Copy nginx-virtual.conf to /etc/nginx/conf.d/virtual.conf
1. Restart nginx: `sudo systemctl restart nginx`

## Set up HTTPS certificate

1. Edit configuration.cnf. Change the IP address.
1. Create a private key: `openssl genrsa -out private-key.pem 2048`
1. Start a server to listen to voice commands for memory game: `python start_flask_servers.py 0`. Ctrl+C will terminate it.
1. Start a server to listen to voice commands for robot movement: `python start_flask_servers.py 1`. Ctrl+C will terminate it.
1. Upload certificate.pem to the Alexa skill under Endpoint > Default Region > Upload self-signed certificate.

## Set up service on Jetson

1. Start virualenv: `pipenv shell`
1. Start a server to listen to voice commands: `python start_flask_servers.py`. Ctrl+C will terminate it.
1. To exit virtualenv and return to plain shell, use `exit`

## Resources

* [Alexa Skills Custom Certificate](https://developer.amazon.com/docs/custom-skills/test-a-custom-skill.html#h2_sslcert)
* [Run on 80 and 443 with iptables](https://wiki.jenkins.io/display/JENKINS/Running+Jenkins+on+Port+80+or+443+using+iptables)
* [Flask over HTTPS](https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https)
