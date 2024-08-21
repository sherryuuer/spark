# Import the necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Load the Amazon product reviews Dataset
print("Step 1: Loading the Amazon Product Reviews dataset")
df = spark.read.csv("Amazon_Reviews.csv", header=True, inferSchema=True)

# Tokenize the text data
print("Step 2: Tokenizing the text")
tokenizer = Tokenizer(inputCol="Review", outputCol="words")
words = tokenizer.transform(df)

# Apply the HashingTF transformation to convert words into features
# HashingTF 是 PySpark 中用于特征提取的一种方法。
# 具体来说，它是一种将文本数据转换为固定长度数值向量的技术，常用于自然语言处理（NLP）和文本分类任务中。
# HashingTF 是基于哈希函数的特征表示方法，属于词袋模型的一种变体。
print("Step 3: Applying HashingTF transformation to convert words into features")
hashingTF = HashingTF(
    inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(words)

# Applying IDF to rescale the raw TF
print("Step 4: Applying IDF to rescale the raw features")
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Train a Logistic Regression classifier
print("Stage 5: Define a classifier (e.g., Logistic Regression)")
classifier = LogisticRegression(maxIter=10, regParam=0.01, labelCol="Rating")

# Create an ML Pipeline by chaining the stages
print("Stage 6: Create an ML pipeline by chaining all the stage")
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, classifier])

# Fit the pipeline to the data
print("Step 7: Fit the pipeline to the input training data and print the model")
model = pipeline.fit(df)

# Load the test Dataset
print("Step 8: Loading the test dataset")
test_df = spark.read.csv("test.csv", header=True)

# Make predictions on test data
print("Step 9: Make predictions on test data")
prediction = model.transform(test_df)
result = prediction.select(
    "features", "Rating", "probability", "prediction").collect()

# Print final results
print("Printing feature, original rating along with probability and predictions")
for row in result:
    print("features=%s, label=%s -> prob=%s, prediction=%s"
          % (row.features, row.Rating, row.probability, row.prediction))
