#/bin/bash 
set -e 
apt-get update 
apt-get upgrade -y 
curl -fsSL https://get.docker.com -o get-docker.sh 
sh get-docker.sh 
usermod -aG docker ubuntu 
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-Linux-x86_64" -o /usr/local/bin/docker-compose 
chmod +x /usr/local/bin/docker-compose 
apt-get install -y nginx git htop 
ufw allow 22 
ufw allow 80 
ufw allow 443 
ufw allow 8000 
ufw --force enable 
mkdir -p /opt/sentiment-ai 
echo "Instance initialisee" > /var/log/init-complete.log 
