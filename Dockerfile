FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PYTHONUNBUFFERED=1 VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
COPY apt-full.lock /locks/apt.lock
RUN apt-get update && xargs -a /locks/apt.lock apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements-full.lock /locks/requirements.lock
RUN python3 -m venv /opt/venv && pip install --no-deps -r /locks/requirements.lock
RUN python -c "import numpy,scipy,numba,cv2,pycolmap,psutil,threadpoolctl,gtsam; from projectaria_tools.core import calibration"
COPY slam_dunk/engine /recipe/slam_dunk/engine
COPY build_native.sh /recipe/build_native.sh
WORKDIR /recipe
RUN ./build_native.sh
COPY . /recipe
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/numba PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/opt/venv/bin/python", "-m", "slam_dunk"]
