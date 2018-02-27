# Install Instructions

How to set up flask-ask running locally without Ngrok

## Install python and set up virtualenv on Jetson

1. Install Python: `sudo apt-get install python`
1. Install dependencies for cryptography module: `sudo apt-get install build-essential libssl-dev libffi-dev python-dev`
1. Install pipenv for all users: `sudo -H pip3 install -U pipenv`
1. Install Python packages to this virtualenv: `pipenv install`

## Configure an EC2 server to act as a remote proxy

1. Add ports 34080 and 34443 to the security groups inbound rules on the AWS console.
1. SSH into the EC2 machine: `ssh -i ~/.ssh/aws-key-pair.pem ubuntu@ec2-18-222-97-118.us-east-2.compute.amazonaws.com`
1. Update the machine: `sudo apt update` and `sudo apt upgrade`. Restart: `sudo reboot`
1. Update the timezone: `sudo dpkg-reconfigure tzdata`
1. Edit the sshd configuration: `sudo -H vim /etc/ssh/sshd_config`. Change `GatewayPorts yes` to `GatewayPorts yes`
1. Restart sshd: `sudo systemctl restart ssh`
1. Redirect 80 and 443 to 34080 and 34443 and enable forwarding:

        sudo iptables -I INPUT 1 -p tcp --dport 34443 -j ACCEPT
        sudo iptables -I INPUT 1 -p tcp --dport 34080 -j ACCEPT
        sudo iptables -I INPUT 1 -p tcp --dport 443 -j ACCEPT
        sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT

1. Check iptables rules: `sudo iptables -L -n`
1. Install nginx: `sudo apt install nginx`
1. Add a user `sudo adduser flaskapp`
1. In `sudo vim /etc/nginx/nginx.conf`, Change `user  www-data;` to `user  flaskapp`. In the http block, uncomment this line: `server_names_hash_bucket_size 64;` and change `64` to `128`
1. In `sudo vi /etc/nginx/conf.d/virtual.conf`, add:

        server {
            listen 443 ssl;
            ssl_certificate /home/flaskapp/certificate.pem;
            ssl_certificate_key /home/flaskapp/private-key.pem;

            location / {
                proxy_pass https://127.0.0.1:8443;
            }
        }

1. Restart nginx: `sudo systemctl start nginx`

## Set up HTTPS certificate

1. Edit configuration.cnf. Change the FQDN.
1. Create a private key: `openssl genrsa -out private-key.pem 2048`
1. Generate a private key `openssl req -new -x509 -days 365 -key private-key.pem -config configuration.cnf -out certificate.pem`
1. Copy the contents of certificate.pem to the Alexa skill under SSL Certificate.
1. Copy the certificate and private key to /home/flaskapp/.
1. Restart nginx.

## Set up service on Jetson

1. Start virualenv: `pipenv shell`
1. Start the web server: `gunicorn --certfile certificate.pem --keyfile private-key.pem -b localhost:8443 myapp:app`. Ctrl+C will terminate it.
1. In another terminal, open up a localhost tunnel: `ssh -i ~/.ssh/aws-key-pair.pem -N -R *:34443:localhost:8443 ubuntu@ec2-18-222-97-118.us-east-2.compute.amazonaws.com`. Ctrl+C will terminate it.
1. To exit the shell, deactivate virtualenv with `exit`

## Resources

* [Alexa Skills Custom Certificate](https://developer.amazon.com/docs/custom-skills/test-a-custom-skill.html#h2_sslcert)
* [Run on 80 and 443 with iptables](https://wiki.jenkins.io/display/JENKINS/Running+Jenkins+on+Port+80+or+443+using+iptables)
* [Deploying Flask to Amazon EC2](https://www.matthealy.com.au/blog/post/deploying-flask-to-amazon-web-services-ec2/)
* [Flask over HTTPS](https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https)
* [Tunneling to localhost](https://blog.jayway.com/2013/10/17/tunneling-to-localhost-via-ssh/)
* [SSH Reverse Tunneling](https://juntx.wordpress.com/2014/07/28/use-amazon-ec2-and-ssh-reverse-tunneling-to-connect-computers-behind-firewall-or-nat/)