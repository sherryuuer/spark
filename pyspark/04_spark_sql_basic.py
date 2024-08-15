# spark sql会利用谓词下推和列修建优化查询，说白了就是将where和select col优先执行
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, substring, when, countDistinct, col, avg

spark = SparkSession.builder.appName("pyspark_sql").getOrCreate()

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("file.csv", header=True, inferSchema=True)

print("Concatenating Name and Age columns:")
df.select(concat("name", "age").alias("Name_Age")).show()

print("Extracting the first letter from the City column:")
df.select(substring("city", 1, 1).alias("First Letter")).show()

print("Performing conditional logic on Age column:")
df.select(
    "name",
    "city",
    when(df["age"] < 30, "Young").otherwise("Old").alias("Age Group")
).show()

print("Counting the number of distinct cities:")
df.select(countDistinct("city").alias("Distinct Cities")).show()

print("Selecting the Name column:")
df.select(col("name")).show()

print("Calculating the average age:")
df.agg(avg(col("age")).alias("Average Age")).show()

print("Calculating the average age by city:")
df.groupBy("city").agg(avg("age").alias("Average Age by city")).show()

# UDF
spark = SparkSession.builder.getOrCreate()

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("file.csv", header=True, inferSchema=True)

print("Define a Python function")


def capitalize_string(s):
    return s.upper()


print("Register the UDF")
capitalize_udf = udf(capitalize_string, StringType())

print("Apply the registered UDF to a column in the DataFrame")
df = df.withColumn("Capitalized City", capitalize_udf(df["city"]))

print("Display the final results of the DataFrame")
df.show()

# run sql on Pyspark
spark = SparkSession.builder.getOrCreate()

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("file.csv", header=True, inferSchema=True)

print("Create a Temporary View")
df.createOrReplaceTempView("people")

print("Select the City column from the temp view")
sqlDF = spark.sql("SELECT city FROM people")

print("Display the contents")
sqlDF.show()

# Global temporary view share in all sessions
spark = SparkSession.builder.getOrCreate()

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("file.csv", header=True, inferSchema=True)

print("Create a Global Temporary View")
df.createGlobalTempView("people")

print("Select city column from the temp view")
sqlDF = spark.sql("SELECT city FROM global_temp.people")

print("Display the contents")
sqlDF.show()
