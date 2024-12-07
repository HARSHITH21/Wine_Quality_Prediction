from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Wine Quality Evaluation") \
    .getOrCreate()

# Path to the saved model and validation dataset
model_path = "/home/ubuntu/wine-quality-app/model"
validation_data_path = "/home/ubuntu/wine-quality-app/ValidationDataset.csv"

# Load the Logistic Regression model
print("Loading the model...")
model = LogisticRegressionModel.load(model_path)

# Load the validation dataset
print("Loading validation dataset...")
validation_data = spark.read.csv(
    validation_data_path,
    header=True,
    inferSchema=True,
    sep=";"  # Specify semicolon as the delimiter
)

# Rename columns to remove extra quotes
print("Cleaning column names...")
for col in validation_data.columns:
    validation_data = validation_data.withColumnRenamed(col, col.strip('"'))

# Vectorize the features
print("Vectorizing features...")
feature_columns = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
validation_data = assembler.transform(validation_data)

# Evaluate the model on the validation data
print("Generating predictions...")
predictions = model.transform(validation_data)

# Show predictions
predictions.select("features", "quality", "prediction", "probability").show()

# Evaluate using metrics
print("Evaluating the model...")
evaluator = MulticlassClassificationEvaluator(
    labelCol="quality",
    predictionCol="prediction",
    metricName="f1"
)

f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score:.2f}")

# Stop Spark session
spark.stop()

