FROM python:3.11-slim

WORKDIR /predict_app

COPY ./requirements.txt ./

COPY ./HR_comma_sep.csv ./

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY ["train.py", "model_wrapper.py","predict.py", "./"]

RUN python3 train.py

EXPOSE 8080

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8080", "predict:app" ]
