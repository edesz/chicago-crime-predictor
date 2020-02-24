FROM python:3.7.6-stretch

USER root

ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GROUP=jovyan
ENV USER=${NB_USER} \
    NB_UID=${NB_UID} \
    HOME=/home/${NB_USER} \
    TZ_TIMEZONE=/etc/timezone \
    TZ_LOCALTIME=/etc/localtime \
    TZ_ZONEINFO=/usr/share/zoneinfo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PY_VERSION=3.7

WORKDIR ${HOME}
COPY . ${HOME}

RUN apt-get update \
    && apt-get install -y \
    python-pip \
    apt-utils \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    python-apt \
    bc \
    --no-install-recommends \
    && pip install --no-cache-dir --upgrade pip -U tox \
    && find /usr/local/lib/python$PY_VERSION -name '*.c' -delete \
    && find /usr/local/lib/python$PY_VERSION -name '*.pxd' -delete \
    && find /usr/local/lib/python$PY_VERSION -name '*.pyd' -delete \
    && find /usr/local/lib/python$PY_VERSION -name '__pycache__' | xargs rm -r

RUN addgroup ${NB_GROUP} \
    && adduser \
    --disabled-password \
    --gecos "Default user" \
    --home ${HOME} \
    --ingroup ${USER} \
    --no-create-home \
    --uid ${NB_UID} \
    ${NB_USER} && \
    chown -R ${NB_UID} ${HOME}

USER ${NB_USER}

CMD ["tox", "-e", "build"]
