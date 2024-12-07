from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Paths to datasets and model save location
training_data_path = "s3://aws-logs-065870645303-us-east-1/elasticmapreduce/winequality/TrainingDataset.csv"
validation_data_path = "s3://aws-logs-065870645303-us-east-1/elasticmapreduce/winequality/ValidationDataset.csv"
model_save_path = "s3://aws-logs-065870645303-us-east-1/elasticmapreduce/winequality/models/wine-quality-model"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Wine Quality Prediction") \
    .getOrCreate()

def load_data(path):
    """Loads data from a CSV file."""
    df = spark.read.csv(path, header=True, inferSchema=True, sep=';')
    print("Data Schema:")
    df.printSchema()
    print(f"Data Row Count: {df.count()}")
    return df

def rename_columns(df):
    """Renames columns to remove unnecessary quotes."""
    expected_columns = [
        "fixed acidity", "volatile acidity", "citric acid", 
        "residual sugar", "chlorides", "free sulfur dioxide", 
        "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"
    ]
    renamed_df = df
    for original, expected in zip(df.columns, expected_columns):
        renamed_df = renamed_df.withColumnRenamed(original, expected)
    print("Schema after renaming columns:")
    renamed_df.printSchema()
    return renamed_df

def preprocess_data(df):
    """Preprocesses the dataset by handling missing values and assembling features."""
    # Handle missing values
    df = df.dropna()
    print(f"Row Count after dropping nulls: {df.count()}")
    
    # Assemble features
    feature_columns = [
        "fixed acidity", "volatile acidity", "citric acid", 
        "residual sugar", "chlorides", "free sulfur dioxide", 
        "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
    ]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)
    
    # Keep only features and label
    df = df.select("features", col("quality").cast("double").alias("quality"))
    print("Schema after preprocessing:")
    df.printSchema()
    print(f"Row Count after preprocessing: {df.count()}")
    df.show(5, truncate=False)
    return df

def train_and_evaluate_model(train_df, validation_df):
    """Trains a Logistic Regression model and evaluates it on validation data."""
    print("Training Dataset Schema:")
    train_df.printSchema()
    print(f"Training Dataset Row Count: {train_df.count()}")
    train_df.show(5, truncate=False)

    if train_df.count() == 0:
        raise ValueError("The training dataset is empty after preprocessing. Please check the input data and preprocessing steps.")

    # Train logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol="quality")
    model = lr.fit(train_df)

    # Save the trained model
    model.write().overwrite().save(model_save_path)

    # Evaluate model
    predictions = model.transform(validation_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    print(f"F1 Score on validation data: {f1_score}")
    return model

# Main execution flow
if __name__ == "__main__":
    # Load datasets
    train_df = load_data(training_data_path)
    validation_df = load_data(validation_data_path)
    
    # Rename columns
    train_df = rename_columns(train_df)
    validation_df = rename_columns(validation_df)
    
    # Preprocess datasets
    train_df = preprocess_data(train_df)
    validation_df = preprocess_data(validation_df)
    
    # Train and evaluate the model
    model = train_and_evaluate_model(train_df, validation_df)

    # Stop Spark session
    spark.stop()

