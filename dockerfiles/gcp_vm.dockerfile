# Base image
FROM python:3.10-slim

# Install git and other dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Clone your repository
RUN git clone -b bucket_mounting https://github.com/CristianaLazar/mlops-bird-classification-project.git


# COPY requirements.txt requirements.txt
# COPY requirements_dev.txt requirements_dev.txt
# COPY pyproject.toml pyproject.toml
# COPY src/ src/
# COPY data/ data/

# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir
# RUN pip install . --no-deps --no-cache-dir

# ENTRYPOINT ["python", "-u", "src/train_model.py"]


# Set the working directory to the cloned repository
WORKDIR /mlops-bird-classification-project
# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/train_model.py"]

# CMD [ "override": "experiment:exp1gcs" ]

