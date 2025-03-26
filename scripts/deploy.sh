#!/usr/bin/env bash

set -e  # 실패 시 즉시 종료
set -x  # 실행되는 명령어 로그 출력

REPOSITORY=/home/ubuntu/flask
FLASK_APP_DIR=$REPOSITORY
ENV_PATH=$FLASK_APP_DIR/.env

echo ">> CD into repo"
mkdir -p $REPOSITORY
cd $REPOSITORY

echo ">> Killing existing flask (if any)"
FLASK_PID=$(pgrep -f "flask run")
[ -n "$FLASK_PID" ] && kill -15 $FLASK_PID && sleep 2

echo ">> Load .env if exists"
[ -f "$ENV_PATH" ] && export $(cat "$ENV_PATH" | xargs)

echo ">> Setting up virtualenv if missing"
if [ ! -d "venv" ]; then
  python3 -m venv venv || echo "❌ venv 생성 실패!"
fi

echo ">> Activating venv"
source venv/bin/activate

echo ">> Installing requirements"
pip install -r requirements.txt