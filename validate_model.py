from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.types import StructType, StructField, DoubleType

# Initialize Spark session
spark = SparkSession.builder.appName("VerifyModel").getOrCreate()

# Load the model
model_path = "./model"
model = LogisticRegressionModel.load(model_path)

# Define schema for 11 features
schema = StructType([StructField(f"feature{i}", DoubleType(), True) for i in range(1, 12)])

# Create validation data with 11 features
data = [(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0),
        (11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)]

validation_data = spark.createDataFrame(data, schema)

# Assemble features
assembler = VectorAssembler(inputCols=[f"feature{i}" for i in range(1, 12)], outputCol="features")
validation_data = assembler.transform(validation_data)

# Perform predictions
predictions = model.transform(validation_data)
predictions.select("features", "prediction").show()

