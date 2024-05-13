FROM python:3.11

USER root

EXPOSE 8585

WORKDIR /code
# 환경 변수 설정

RUN apt-get update && \
    apt-get install -y wget unzip && \
    apt-get clean


# Install dependencies
RUN apt-get install -f


# # 필요한 Python 패키지 설치
RUN pip install fastapi uvicorn Pillow opencv-python


# 컨테이너 내에서 작업할 디렉토리 생성
WORKDIR /code

# capture.py와 같은 스크립트를 복사하여 컨테이너 내부에 추가
# 필요한 패키지 및 라이브러리 설치

CMD ["bash", "-c", "Xvfb :99 -ac & export DISPLAY=:99"]
