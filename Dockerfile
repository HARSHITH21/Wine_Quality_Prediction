# Use an official openjdk base image
FROM openjdk:8-jre-slim

# Install dependencies (curl, wget, etc.)
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    wget \
    unzip \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -s -L --url "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -f -p "/opt/miniconda3" && \
    rm /tmp/miniconda.sh && \
    /opt/miniconda3/bin/conda config --set auto_update_conda true && \
    /opt/miniconda3/bin/conda config --set channel_priority false && \
    /opt/miniconda3/bin/conda update conda -y --force-reinstall && \
    /opt/miniconda3/bin/conda clean -tipy

# Install required Python packages
RUN /opt/miniconda3/bin/pip install --no-cache-dir pyspark==3.4.0 numpy pandas boto3

# Install Spark
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz" \
    && tar -xvf apache-spark.tgz -C /opt \
    && rm apache-spark.tgz

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark-3.4.0-bin-hadoop3
ENV PATH=$SPARK_HOME/bin:$PATH

# Set working directory
WORKDIR /opt

# Copy necessary files
COPY validate_and_evaluate.py /opt/
COPY ValidationDataset.csv /opt/

# Default command to run when the container starts
CMD ["spark-submit", "/opt/validate_and_evaluate.py", "/opt/ValidationDataset.csv"]

