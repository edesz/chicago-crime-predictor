FROM python:3.8.1-slim as builder
ENV PY_VERSION=3.8
RUN apt-get update \
    && apt-get install -y gcc curl \
    python-pip \
    --no-install-recommends \
    && pip install --no-cache-dir --upgrade pip \
    && apt-get clean
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --user -r requirements.txt
COPY /app /app

FROM python:3.8.1-slim as app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
ARG AZURE_STORAGE_ACCOUNT=abc
ARG AZURE_STORAGE_KEY=defg
ENV AZURE_STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT}"
ENV AZURE_STORAGE_KEY="${AZURE_STORAGE_KEY}"
EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:80", "-w4", "app:server"]
