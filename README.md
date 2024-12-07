# wine-quality-prediction
Cluster Configuration
Steps:
Launch an EMR Cluster:

Navigate to the EMR section in the AWS Console.
Create a cluster with:
1 Master Node and 4 Slave Nodes.
Select applications: Spark, Hadoop.
Choose an EC2 key pair for SSH access (create one if needed).
Once launched, note the Master Node DNS and login details.
Install Required Libraries:

SSH into the Master Node.
Install additional Java dependencies if necessary (e.g., libraries for Spark MLlib).
Ensure that all configurations are synced across nodes.
