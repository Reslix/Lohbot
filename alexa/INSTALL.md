# Install Instructions

How to set up flask-ask running locally

## Install python and set up virtualenv on Jetson

1. Install Python: `sudo apt-get install python`
1. Install dependencies for cryptography module: `sudo apt-get install build-essential libssl-dev libffi-dev python-dev`
1. Install pipenv for all users: `sudo -H pip3 install -U pipenv`
1. Install Python packages to this virtualenv: `pipenv install`

## Install nginx

1. Update the timezone: `sudo dpkg-reconfigure tzdata`
1. Install nginx: `sudo apt install nginx`
1. Add a user `sudo adduser flaskapp`
1. In `sudo vim /etc/nginx/nginx.conf`, Change `user  www-data;` to `user  flaskapp`. In the http block, uncomment this line: `server_names_hash_bucket_size 64;` and change `64` to `128`
1. In `sudo vi /etc/nginx/conf.d/virtual.conf`, add:

        server {
            listen 443 ssl;
            ssl_certificate /home/nvidia/certificate.pem;
            ssl_certificate_key /home/nvidia/private-key.pem;

            location / {
                proxy_pass https://127.0.0.1:34443;
            }
        }

1. Restart nginx: `sudo systemctl restart nginx`

## Set up HTTPS certificate

1. Edit configuration.cnf. Change the IP address.
1. Create a private key: `openssl genrsa -out private-key.pem 2048`
1. Generate a private key `openssl req -new -x509 -days 365 -key private-key.pem -config configuration.cnf -out certificate.pem`
1. Upload certificate.pem to the Alexa skill under Endpoint > Default Region > Upload self-signed certificate.

## Set up service on Jetson

1. Start virualenv: `pipenv shell`
1. Start the web server: `gunicorn --certfile certificate.pem --keyfile private-key.pem -b localhost:34443 myapp:app`. Ctrl+C will terminate it.
1. To exit the shell, deactivate virtualenv with `exit`

## Resources

* [Alexa Skills Custom Certificate](https://developer.amazon.com/docs/custom-skills/test-a-custom-skill.html#h2_sslcert)
* [Run on 80 and 443 with iptables](https://wiki.jenkins.io/display/JENKINS/Running+Jenkins+on+Port+80+or+443+using+iptables)
* [Flask over HTTPS](https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https)