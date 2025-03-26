#!/usr/bin/env bash
REPOSITORY=/home/ubuntu/flask
FLASK_APP_DIR=/home/ubuntu/flask
ENV_PATH=$FLASK_APP_DIR/.env

echo "> Checking and creating project directory..."
mkdir -p $REPOSITORY
cd $REPOSITORY

# 기존 Flask 앱 종료
FLASK_PID=$(pgrep -f "flask run")
if [ -z "$FLASK_PID" ]
then
  echo "> 종료할 Flask 애플리케이션이 없습니다."
else
  echo "> kill Flask app with PID: $FLASK_PID"
  kill -15 $FLASK_PID
  sleep 3
fi

# .env 로딩
if [ -f "$ENV_PATH" ]; then
    export $(cat "$ENV_PATH" | xargs)   # flask에서 환경변수 인식하도록 export
fi

# 가상환경 재설정
echo "> Removing existing venv directory"
rm -rf $FLASK_APP_DIR/venv

echo "> Setting up new virtual environment"
python3 -m venv $FLASK_APP_DIR/venv
source $FLASK_APP_DIR/venv/bin/activate

echo "> Installing dependencies"
pip install -r $FLASK_APP_DIR/requirements.txt

# # Flask 앱 실행 (flask run)
# echo "> Starting Flask app with flask run"
# cd $FLASK_APP_DIR
# source $FLASK_APP_DIR/venv/bin/activate

# # FLASK_APP 설정 (main 모듈명에 따라 수정)
# export FLASK_APP=app.py   # 또는 main.py 등 진입점 파일명
# export FLASK_ENV=production  # 또는 development

# # 백그라운드 실행
# nohup flask run --host=0.0.0.0 --port=5000 > /dev/null 2> /dev/null < /dev/null &
