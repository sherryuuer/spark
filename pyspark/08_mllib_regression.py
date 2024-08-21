# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Imputer

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Load the dataset
print("Step 1: Loading the dataset")
df = spark.read.csv("house_prices.csv", header=True, inferSchema=True)

# Handle missing values
print("Step 2: Handling missing values")
imputer = Imputer(
    inputCols=["total_bedrooms"],
    outputCols=["total_bedrooms_imputed"]
)
df = imputer.fit(df).transform(df)

# Renaming and feature engineering
print("Step 3: Feature engineering")
df = df.withColumnRenamed('median_house_value', 'price')
df = df.withColumn(
    'per_capita_income',
    df['median_income'] * 10000 / df['population']
)

indexer = StringIndexer(inputCol="ocean_proximity",
                        outputCol="ocean_proximity_index")
df = indexer.fit(df).transform(df)
df = df.drop('ocean_proximity')

# Assembling features
print("Step 4: Feature Transformation")
assembler = VectorAssembler(
    inputCols=[
        "total_rooms",
        "housing_median_age",
        "total_bedrooms_imputed",
        "per_capita_income"
    ],
    outputCol="features"
)
df_assembler = assembler.transform(df)

# Normalizing the features
print("Step 5: Normalizing")
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=False
)
scalerModel = scaler.fit(df_assembler)
scaledOutput = scalerModel.transform(df_assembler)

# Select only price and scaled features columns
df_model_final = scaledOutput.select(['price', 'scaledFeatures'])

# Split the data into training and testing sets (e.g., 80% train, 20% test)
print("Step 6: Splitting data into training and testing sets")
trainData, testData = df_model_final.randomSplit([0.8, 0.2], seed=123)

# Create a regression model instance
print("Step 7: Creating Regression instance")
regression = LinearRegression(featuresCol="scaledFeatures", labelCol="price")

# Train the regression model on the training data
print("Step 8: Building and training the regression model")
regressionModel = regression.fit(trainData)

# Make predictions
print("Step 9: Making predictions")
predictions = regressionModel.transform(testData)
predictions.show()
