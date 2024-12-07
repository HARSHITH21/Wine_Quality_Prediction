
# Wine Quality Prediction on AWS

## Overview
This project involves developing a parallel machine learning application on the Amazon AWS cloud platform using Apache Spark. The goal is to train a wine quality prediction model in parallel across multiple EC2 instances, load and save the model for predictions, and deploy the model using Docker.

## Dataset
The project uses the following datasets:
- **TrainingDataset.csv**: Used for training the model in parallel on 4 EC2 instances.
- **ValidationDataset.csv**: Used for model validation and optimization of model parameters.
- **TestDataset.csv**: Used for testing prediction functionality and performance (not provided; validation dataset can be used for testing).

## Goal
The main objective is to build and deploy a machine learning model that predicts wine quality, using parallel computing to train the model on AWS. The project will output the F1 score to measure prediction performance.

## Technologies and Tools
- **Amazon EC2**: Used for running Spark and distributing tasks across multiple instances.
- **Apache Spark**: Distributed computing framework used for parallel training and prediction.
- **Java**: Programming language for implementation.
- **Docker**: Containerization tool for deploying the trained model.
- **Spark MLlib**: Spark's built-in library for machine learning to train and validate models.

---

## Cluster Configuration

### Step 1: Launch the EMR Cluster
- **Go to the AWS Management Console**.
- Navigate to **EMR (Elastic MapReduce)** and click **Create Cluster**.
- **Configure the cluster**:
  - **Applications**: Select **Spark** and **Hadoop**.
  - **Instance Types**: Choose instance types for your master and slave nodes (e.g., m5.xlarge).
  - **Number of Instances**: Set 1 master and 4 slave nodes.
  - **Key Pair**: Select an existing key pair or create a new one for SSH access.
- Launch the cluster and wait for it to start.

### Step 2: SSH into the Master Node
- Obtain the **Master Node DNS** from the EMR cluster details in the AWS Console.
- Open a terminal on your local machine and connect to the master node via SSH:
```bash
ssh -i your-key.pem hadoop@<MasterNodeDNS>
```

### Step 3: Verify PySpark Installation
- Check if **PySpark** is installed by running:
```bash
pyspark
```
- If PySpark starts successfully, you will see the Spark shell prompt.
- To exit the Spark shell, type:
```bash
exit()
```

### Step 4: Configure Environment Variables
- Ensure Spark and Hadoop are properly configured by checking the environment variables.
- View the `~/.bashrc` file to confirm Spark and Hadoop paths:
```bash
cat ~/.bashrc
```
- Add the following lines if they are missing:
```bash
echo "export SPARK_HOME=/usr/lib/spark" >> ~/.bashrc
echo "export PATH=\$SPARK_HOME/bin:\$PATH" >> ~/.bashrc
echo "export HADOOP_HOME=/usr/lib/hadoop" >> ~/.bashrc
echo "export PATH=\$HADOOP_HOME/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Test PySpark with a Simple Script
- Create a test PySpark script:
```bash
nano test_pyspark.py
```
- Add the following code to the file:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestPySpark").getOrCreate()
data = [("Alice", 34), ("Bob", 45), ("Cathy", 29)]
df = spark.createDataFrame(data, ["Name", "Age"])
df.show()
spark.stop()
```
- Run the script:
```bash
spark-submit test_pyspark.py
```

### Step 6: Validate Cluster Configuration
- Ensure Spark can utilize multiple nodes by checking cluster details:
```bash
yarn node -list
```
- This should list the master and slave nodes.

- Check available Spark resources:
```bash
spark-submit --status
```

### Step 7: Upload Datasets to S3
- Open the AWS Console and go to **S3**.
- Upload the datasets **TrainingDataset.csv** and **ValidationDataset.csv** to your S3 bucket.
- Alternatively, you can upload the datasets using the AWS CLI:
```bash
aws s3 cp TrainingDataset.csv s3://aws-logs-065870645303-us-east-1/
aws s3 cp ValidationDataset.csv s3://aws-logs-065870645303-us-east-1/
```
- Test S3 Access:
```bash
aws s3 ls s3://aws-logs-065870645303-us-east-1
```

### Step 8: Train the Model on EC2
- SSH into the master node and run the training script:
```bash
spark-submit wine_quality_training.py
```
- Ensure that the model is trained and saved to a directory.

---

## Docker Setup for Prediction

### Step 1: Create the Dockerfile

Create a **Dockerfile** to define the application environment. The container will include dependencies like Apache Spark, Python, and necessary libraries for the model.

#### Dockerfile

### Step 2: Build the Docker Image
- After creating the Dockerfile, build the Docker image:
```bash
docker build -t wine-quality-app .
```

### Step 3: Run the Docker Container
- Once the Docker image is built, you can run the container:
```bash
docker run -it wine-quality-app
```
- This will execute the prediction process using Spark inside the container.

### Step 4: Running the Prediction on EC2 (Optional)
- Alternatively, you can run the prediction directly on EC2 without Docker:
```bash
spark-submit --master yarn --deploy-mode client /opt/validate_and_evaluate.py
```

---

## Conclusion
This project successfully demonstrates parallel machine learning on AWS using Apache Spark. The training is done on multiple EC2 instances using Spark's distributed computing capabilities, and the trained model is deployed in a Docker container for easy prediction deployment. The entire process leverages AWS's powerful infrastructure and Spark's capabilities to handle large datasets efficiently.

