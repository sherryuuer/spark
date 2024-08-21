# Import the necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Load the Amazon product reviews dataset
print("Step 1: Loading the Amazon Product Reviews dataset")
df = spark.read.csv("Amazon_Reviews.csv", header=True, inferSchema=True)

# Tokenize the text data
print("Step 2: Tokenizing the text")
tokenizer = Tokenizer(inputCol="Review", outputCol="words")
words = tokenizer.transform(df)

# Apply the HashingTF transformation to convert words into features
print("Step 3: Applying HashingTF transformation to convert words into features")
hashingTF = HashingTF(
    inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(words)

# Applying IDF to rescale the raw TF
print("Step 4: Applying IDF to rescale the raw features")
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Split the data into training and testing sets (80% train, 20% test)
print("Step 5: Splitting data into training and testing sets")
trainData, testData = rescaledData.randomSplit([0.8, 0.2], seed=123)

# Train a Logistic Regression classifier
print("Step 6: Define a classifier (e.g., Logistic Regression) and train it")
classifier = LogisticRegression(
    maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol="Rating")

# Create a parameter grid for hyperparameter tuning
print("Step 7: Creating a Parameter grid for tuning")
paramGrid = ParamGridBuilder() \
    .addGrid(classifier.maxIter, [10, 20, 30]) \
    .addGrid(classifier.regParam, [0.1, 0.01, 0.001]) \
    .addGrid(classifier.elasticNetParam, [0.7, 0.8, 0.9]) \
    .build()

# Use BinaryClassificationEvaluator for evaluation
print("Step 8: Creating a BinaryClassification Evaluator")
binaryEvaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction", labelCol="Rating")

# Create a TrainValidationSplit for model selection
print("Step 9: TrainValidationSplit for model selection step")
tvs = TrainValidationSplit(estimator=classifier,
                           estimatorParamMaps=paramGrid,
                           evaluator=binaryEvaluator,
                           trainRatio=0.8)  # 这里不一样！！！

# Fit the CrossValidator to the training data
print("Step 10: Model fitting with the CrossValidator")
tvsModel = tvs.fit(trainData)

# Get the best-performing model
print("Step 11: Print the best performing model")
bestModel = tvsModel.bestModel

# Print the best hyperparameters
print("Best Max Iteration:", bestModel.getOrDefault("maxIter"))
print("Best Regularization Parameter:", bestModel.getOrDefault("regParam"))
print("Best Elastic Net Parameter:", bestModel.getOrDefault("elasticNetParam"))
