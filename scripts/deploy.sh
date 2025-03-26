#!/usr/bin/env bash

REPOSITORY=/home/ubuntu/flask
FLASK_APP_DIR=$REPOSITORY
ENV_PATH=$FLASK_APP_DIR/.env

echo "✅ Checking and creating project directory..."
mkdir -p $REPOSITORY
cd $REPOSITORY

# 기존 Flask 앱 종료
FLASK_PID=$(pgrep -f "flask run")
if [ -z "$FLASK_PID" ]; then
  echo "🛑 종료할 Flask 애플리케이션이 없습니다."
else
  echo "🔪 kill Flask app with PID: $FLASK_PID"
  kill -15 $FLASK_PID
  sleep 3
fi

# .env 로딩
if [ -f "$ENV_PATH" ]; then
    export $(cat "$ENV_PATH" | xargs)
    echo "📦 환경변수 로딩 완료"
fi

# 가상환경이 없으면 새로 생성
if [ ! -d "$FLASK_APP_DIR/venv" ]; then
  echo "🧱 가상환경 생성 중..."
  python3 -m venv $FLASK_APP_DIR/venv
else
  echo "♻️ 기존 가상환경 재사용"
fi

# 가상환경 진입
source $FLASK_APP_DIR/venv/bin/activate

# 의존성 설치 (필요한 것만)
echo "📦 requirements.txt로 의존성 설치 중..."
pip install -r $FLASK_APP_DIR/requirements.txt

# Flask 앱 실행 (원할 경우 주석 해제)
# echo "🚀 Starting Flask app with flask run"
# export FLASK_APP=app.py
# export FLASK_ENV=production
# nohup flask run --host=0.0.0.0 --port=5000 > flask.log 2>&1 &

echo "✅ 배포 스크립트 완료"