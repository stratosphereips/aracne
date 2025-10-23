FROM python:3.12-slim

WORKDIR /agent

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ /agent

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["bash"]
