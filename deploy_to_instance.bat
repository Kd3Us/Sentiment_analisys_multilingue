@echo off
setlocal

if "%1"=="" (
    echo Usage: %0 ^<instance_ip^>
    echo Exemple: %0 18.206.12.34
    exit /b 1
)

set INSTANCE_IP=%1
set KEY_FILE=sentiment-ai-key.pem
set ENV_FILE=.env.production

echo [92mDeploiement sur l'instance %INSTANCE_IP%...[0m

echo [93mCopie des fichiers...[0m
scp -i %KEY_FILE% -o StrictHostKeyChecking=no Dockerfile docker-compose.yml %ENV_FILE% requirements.txt ubuntu@%INSTANCE_IP%:/opt/sentiment-ai/

echo [93mCopie du code de l'application...[0m
scp -i %KEY_FILE% -o StrictHostKeyChecking=no -r app main.py ubuntu@%INSTANCE_IP%:/opt/sentiment-ai/

echo [93mDeploiement et demarrage des services...[0m
ssh -i %KEY_FILE% -o StrictHostKeyChecking=no ubuntu@%INSTANCE_IP% "cd /opt/sentiment-ai && mv %ENV_FILE% .env && sudo docker-compose up --build -d"

echo [93mVerification du statut...[0m
timeout /t 30 /nobreak >nul
ssh -i %KEY_FILE% -o StrictHostKeyChecking=no ubuntu@%INSTANCE_IP% "cd /opt/sentiment-ai && sudo docker-compose ps"

echo [93mTest de l'API...[0m
ssh -i %KEY_FILE% -o StrictHostKeyChecking=no ubuntu@%INSTANCE_IP% "curl -f http://localhost:8000/health || echo 'API pas encore prete'"

echo.
echo [92mDeploiement termine![0m
echo [92mAPI accessible sur: http://%INSTANCE_IP%:8000[0m
echo [92mDocumentation: http://%INSTANCE_IP%:8000/docs[0m