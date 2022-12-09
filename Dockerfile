FROM python:3.9.15-buster
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV TZ=Asia/Taipei
EXPOSE 7010
CMD ["python", "-u", "app.py"]
