FROM python:3.7-slim

WORKDIR /
RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]
CMD [ "run.py" ]
