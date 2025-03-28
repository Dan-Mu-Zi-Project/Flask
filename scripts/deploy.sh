#!/usr/bin/env bash

REPOSITORY=/home/ubuntu/flask
FLASK_APP_DIR=$REPOSITORY
ENV_PATH=$FLASK_APP_DIR/.env

echo "✅ CD into repo"
mkdir -p $REPOSITORY
cd $REPOSITORY

echo "🛑 Killing existing flask (if any)"
FLASK_PID=$(pgrep -f "flask run")
if [ -n "$FLASK_PID" ]; then
  echo "🔪 Killing PID: $FLASK_PID"
  kill -15 "$FLASK_PID"
  sleep 3
else
  echo "ℹ️ No running Flask app found"
fi

echo "📦 Loading .env if exists"
if [ -f "$ENV_PATH" ]; then
  export $(cat "$ENV_PATH" | xargs)
fi

echo "🔧 Fixing directory ownership"
sudo chown -R ubuntu:ubuntu $FLASK_APP_DIR

echo "🧱 Setting up virtualenv if missing"
if [ ! -d "venv" ]; then
  echo "🧱 Creating virtual environment..."
  python3 -m venv venv || (echo "❌ venv 생성 실패" && exit 1)
else
  echo "♻️ 기존 가상환경 사용"
fi

echo "✅ Activating virtualenv"
source venv/bin/activate || (echo "❌ 가상환경 활성화 실패" && exit 1)

echo "📦 Installing requirements"
pip install -r requirements.txt || (echo "❌ requirements 설치 실패" && exit 1)

sudo systemctl restart myflask

# 선택적으로 Flask 실행
# echo "🚀 Starting Flask app"
# export FLASK_APP=app.py
# export FLASK_ENV=production
# nohup flask run --host=0.0.0.0 --port=5000 > flask.log 2>&1 &

echo "✅ Deploy script completed successfully"
