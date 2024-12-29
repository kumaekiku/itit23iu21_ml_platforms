FROM python:3.11-slim

WORKDIR /predict_app

COPY ./requirements.txt ./

COPY ./data/train_data.csv ./data/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY ["train.py", "model_wrapper.py","predict.py", "./"]

RUN python3 train.py

EXPOSE 8080

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8080", "predict:app" ]
