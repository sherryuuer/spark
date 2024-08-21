# Import the necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Load the Amazon product reviews dataset
print("Step 1: Loading the Amazon Product Reviews dataset")
df = spark.read.csv("Amazon_Reviews.csv", header=True, inferSchema=True)

# Tokenize the text data and create a new column
print("Step 2: Tokenizing the text")
tokenizer = Tokenizer(inputCol="Review", outputCol="tokenized_words")
words = tokenizer.transform(df)

# Apply the HashingTF transformation to convert words into features
print("Step 3: Applying HashingTF transformation to convert words into features")
hashingTF = HashingTF(
    inputCol="tokenized_words",
    outputCol="rawFeatures",
    numFeatures=20
)
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
print("Step 6: Training a Logistic Regression classifier")
classifier = LogisticRegression(
    maxIter=10, regParam=0.01, elasticNetParam=1, labelCol="Rating"
)

# Fit the classifer to the training data
print("Step 7: Fitting the pipeline to the input training data and print the model")
model = classifier.fit(trainData)

# Make predictions on test data
print("Step 8: Making predictions on test data")
prediction = model.transform(testData)
result = prediction.select(
    "features", "Rating", "probability", "prediction"
).collect()

# Print final results
print("Printing feature, original rating along with probability and predictions")
for row in result:
    print("Features=%s, Label=%s -> Prob=%s, Prediction=%s"
          % (row.features, row.Rating, row.probability, row.prediction))
