@echo off
setlocal enabledelayedexpansion

set INSTANCE_NAME=sentiment-ai-api
set INSTANCE_BLUEPRINT=ubuntu_20_04
set INSTANCE_BUNDLE=medium_2_0
set REGION=us-east-1
set KEY_PAIR_NAME=sentiment-ai-key

for /f %%i in ('powershell -command "Get-Date -Format 'yyyyMMddHHmmss'"') do set TIMESTAMP=%%i
set S3_BUCKET_NAME=sentiment-ai-models-%TIMESTAMP%

echo [92m=== Deploiement Sentiment AI Platform ===[0m
echo.

echo [93mVerification des prerequis...[0m
aws --version >nul 2>&1
if errorlevel 1 (
    echo [91mErreur: AWS CLI non installe[0m
    exit /b 1
)

aws sts get-caller-identity >nul 2>&1
if errorlevel 1 (
    echo [91mErreur: AWS CLI non configure[0m
    exit /b 1
)

docker --version >nul 2>&1
if errorlevel 1 (
    echo [91mErreur: Docker non installe[0m
    exit /b 1
)

echo [92mPrerequis OK[0m

echo [93mCreation du bucket S3...[0m
aws s3 mb s3://%S3_BUCKET_NAME% --region %REGION%
if errorlevel 1 (
    echo [91mErreur creation bucket S3[0m
    exit /b 1
)
echo [92mBucket S3 cree: %S3_BUCKET_NAME%[0m

echo [93mCreation de la paire de cles...[0m
aws lightsail get-key-pair --key-pair-name %KEY_PAIR_NAME% >nul 2>&1
if errorlevel 1 (
    aws lightsail create-key-pair --key-pair-name %KEY_PAIR_NAME% --query "privateKeyBase64" --output text > %KEY_PAIR_NAME%.pem
    echo [92mPaire de cles creee: %KEY_PAIR_NAME%.pem[0m
) else (
    echo [93mPaire de cles existe deja[0m
)

echo [93mCreation de l'instance Lightsail...[0m

echo #!/bin/bash > user_data.sh
echo set -e >> user_data.sh
echo apt-get update >> user_data.sh
echo apt-get upgrade -y >> user_data.sh
echo curl -fsSL https://get.docker.com -o get-docker.sh >> user_data.sh
echo sh get-docker.sh >> user_data.sh
echo usermod -aG docker ubuntu >> user_data.sh
echo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-Linux-x86_64" -o /usr/local/bin/docker-compose >> user_data.sh
echo chmod +x /usr/local/bin/docker-compose >> user_data.sh
echo apt-get install -y nginx git htop >> user_data.sh
echo ufw allow 22 >> user_data.sh
echo ufw allow 80 >> user_data.sh
echo ufw allow 443 >> user_data.sh
echo ufw allow 8000 >> user_data.sh
echo ufw --force enable >> user_data.sh
echo mkdir -p /opt/sentiment-ai >> user_data.sh
echo echo "Instance initialisee" ^> /var/log/init-complete.log >> user_data.sh

aws lightsail create-instances --instance-names %INSTANCE_NAME% --availability-zone %REGION%a --blueprint-id %INSTANCE_BLUEPRINT% --bundle-id %INSTANCE_BUNDLE% --user-data file://user_data.sh --key-pair-name %KEY_PAIR_NAME%

if errorlevel 1 (
    echo [91mErreur creation instance[0m
    exit /b 1
)

echo [92mInstance creee: %INSTANCE_NAME%[0m

echo [93mAttente demarrage instance...[0m
:wait_loop
for /f "tokens=*" %%i in ('aws lightsail get-instance --instance-name %INSTANCE_NAME% --query "instance.state.name" --output text') do set STATE=%%i
if not "%STATE%"=="running" (
    timeout /t 10 /nobreak >nul
    goto wait_loop
)

echo [92mInstance en cours d'execution[0m

echo [93mConfiguration du firewall...[0m
aws lightsail put-instance-public-ports --instance-name %INSTANCE_NAME% --port-infos fromPort=80,toPort=80,protocol=TCP,cidrs=0.0.0.0/0 fromPort=443,toPort=443,protocol=TCP,cidrs=0.0.0.0/0 fromPort=22,toPort=22,protocol=TCP,cidrs=0.0.0.0/0 fromPort=8000,toPort=8000,protocol=TCP,cidrs=0.0.0.0/0

for /f "tokens=*" %%i in ('aws lightsail get-instance --instance-name %INSTANCE_NAME% --query "instance.publicIpAddress" --output text') do set PUBLIC_IP=%%i

echo [93mCreation IP statique...[0m
set STATIC_IP_NAME=%INSTANCE_NAME%-static-ip
aws lightsail allocate-static-ip --static-ip-name %STATIC_IP_NAME%
aws lightsail attach-static-ip --static-ip-name %STATIC_IP_NAME% --instance-name %INSTANCE_NAME%

for /f "tokens=*" %%i in ('aws lightsail get-static-ip --static-ip-name %STATIC_IP_NAME% --query "staticIp.ipAddress" --output text') do set PUBLIC_IP=%%i

echo.
echo [92m=== Deploiement termine ===[0m
echo [92mInstance: %INSTANCE_NAME%[0m
echo [92mIP publique: %PUBLIC_IP%[0m
echo [92mBucket S3: %S3_BUCKET_NAME%[0m
echo [92mCle SSH: %KEY_PAIR_NAME%.pem[0m
echo.
echo [93mProchaines etapes:[0m
echo 1. Attendre 5-10 minutes que l'instance termine son initialisation
echo 2. Modifier le fichier .env avec le nom du bucket S3: %S3_BUCKET_NAME%
echo 3. Executer: deploy_to_instance.bat %PUBLIC_IP%
echo 4. API accessible sur: http://%PUBLIC_IP%:8000
echo.

echo AWS_REGION=%REGION% > .env.production
echo S3_BUCKET_NAME=%S3_BUCKET_NAME% >> .env.production
type .env >> .env.production

del user_data.sh

echo [93mFichier .env.production cree avec les bonnes valeurs[0m