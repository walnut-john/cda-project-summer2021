FROM python:3.7-slim

WORKDIR /
RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT [ "python" ]
CMD [ "jupyter lab" ]
