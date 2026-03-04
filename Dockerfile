# Dockerfile for AL-image-dataset
# builds an image with Python environment and exposes Jupyter

FROM python:3.11-slim

# metadata
LABEL maintainer="Klemen <no-reply@example.com>"

# set working directory
WORKDIR /app

# copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir jupyter \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# copy rest of repo
COPY . /app/

# expose notebook port
EXPOSE 8888

# default command: start jupyter notebook accessible from outside
CMD ["python", "-m", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
