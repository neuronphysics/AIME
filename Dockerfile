FROM ed2022/eric-smarts:planet-pytorch

RUN pip install --upgrade pip && \
    pip install --upgrade --no-cache-dir torch torchvision