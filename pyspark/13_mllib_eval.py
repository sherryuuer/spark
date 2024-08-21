# Import the necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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

# Train a logistic regression classifier
print("Step 6: Define a classifier (e.g., Logistic Regression) and train it")
classifier = LogisticRegression(maxIter=10, regParam=0.01, labelCol="Rating")
model = classifier.fit(trainData)

# Make predictions on the test data
print("Step 7: Make predictions on the test data")
prediction = model.transform(testData)

# Evaluate the model
print("Step 8: Evaluating the model")
binaryEvaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction", labelCol="Rating")

# Calculate accuracy
print("Calculating accuracy")
accuracy = prediction.filter(
    prediction.Rating == prediction.prediction).count() / prediction.count()

# Calculate precision
print("Calculating precision")
precision = (
    prediction.filter((prediction.Rating == 1) & (
        prediction.prediction == 1)).count()
    / prediction.filter(prediction.prediction == 1).count()
)

# Calculate recall
print("Calculating precision")
recall = (
    prediction.filter((prediction.Rating == 1) & (
        prediction.prediction == 1)).count()
    / prediction.filter(prediction.Rating == 1).count()
)

# Calculate F1-score
print("Calculating F1")
f1 = 2 * (precision * recall) / (precision + recall)

# Calculate AUC
print("Calculating AUC")
binaryAUC = binaryEvaluator.evaluate(prediction)

# Print final evaluation results
print(f"Binary AUC: {binaryAUC}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
