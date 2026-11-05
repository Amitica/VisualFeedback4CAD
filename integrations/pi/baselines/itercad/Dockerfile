FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libglx-mesa0 libglib2.0-0 libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY integrations/pi/baselines/itercad/requirements.txt /opt/adapter/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/adapter/requirements.txt \
    && python -m pip freeze > /opt/runtime-requirements.freeze.txt
ENV XDG_CACHE_HOME=/tmp/cache
COPY external_baselines/itercad/eval /opt/IterCAD/eval
COPY external_baselines/itercad/LICENSE /opt/IterCAD/LICENSE
COPY integrations/pi/baselines/itercad /opt/adapter
RUN python /opt/adapter/native.py source-check
WORKDIR /work
ENTRYPOINT ["python", "/opt/adapter/native.py"]
