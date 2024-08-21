# Import the necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Load the Amazon Product Reviews dataset
print("Step 1: Loading the Amazon Product Reviews dataset")
df = spark.read.csv("Amazon_Reviews.csv", header=True, inferSchema=True)

# Tokenize and vectorize text data
print("Step 2: Tokenizing and vectorizing text data")
tokenizer = Tokenizer(inputCol="Review", outputCol="words")
words = tokenizer.transform(df)

# Apply the HashingTF transformation to convert words into features
hashingTF = HashingTF(
    inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(words)

# Applying IDF to rescale the raw TF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Split the data into training and testing sets
print("Step 3: Splitting data into training and testing sets")
trainData, testData = rescaledData.randomSplit([0.8, 0.2], seed=123)

# Create and fit the K-means model on the training data
print("Step 4: Creating and fitting the K-means model")
k = 3  # Choose the number of clusters
kmeans = KMeans().setK(k).setSeed(1)
model = kmeans.fit(trainData)

# Make predictions on the test data
print("Step 5: Making predictions on the test data")
predictions = model.transform(testData)

# Evaluate the clustering using the Silhouette score
print("Step 6: Evaluating the clustering")
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette}")

# Display cluster centers
centers = model.clusterCenters()
print("Step 7: Display Cluster Centers:")
for center in centers:
    print(center)
