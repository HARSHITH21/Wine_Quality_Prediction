# Use an official OpenJDK runtime as a base image
FROM openjdk:8-jre-slim

# Install required tools and dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    python3 \
    python3-pip \
    procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PySpark and additional Python dependencies
RUN pip3 install pyspark pandas numpy

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Download and install Apache Spark 3.5.3
RUN wget https://archive.apache.org/dist/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz && \
    tar -xvf spark-3.5.3-bin-hadoop3.tgz && \
    mv spark-3.5.3-bin-hadoop3 /opt/spark && \
    rm spark-3.5.3-bin-hadoop3.tgz

# Set the working directory inside the container
WORKDIR /opt

# Copy application files into the container
COPY validate_and_evaluate.py /opt/
COPY ValidationDataset.csv /opt/
COPY ./spark-model /opt/spark-model

# Set the command to run the Spark application
CMD ["spark-submit", "/opt/validate_and_evaluate.py"]

