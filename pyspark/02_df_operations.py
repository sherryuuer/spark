from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DataFrameTransformations").getOrCreate()

# transformations
print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("file.csv", header=True, inferSchema=True)

print("Showing the contents of the DataFrame:")
df.select("name", "city").show()

print("Filtering for rows whose age is above 21:")
df.filter(df["age"] > 21).show()

print("Creating a new column by subtracting 100 from the person's age:")
df.withColumn('new_column', 100 - df['age']).show()

print("Grouping by city and then taking an average of age from that grouped city:")
df.groupBy("city").agg({'age': 'sum'}).show()

print("Sorting the DataFrame by age and printing the DataFrame:")
df.orderBy('age').show()

print("Removing email and city columns and printing the rest of the columns:")
df.drop('email', 'city').show()

# actions
spark = SparkSession.builder.getOrCreate()

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("file.csv", header=True, inferSchema=True)

print("Printing the contents of the DataFrame:")
df.show()

print("Showing the number of rows in the DataFrame:")
print(df.count())

print("Retrieving all the data in the DataFrame as a list instead of a table:")
print(df.collect())

print("Showing the first row of the DataFrame:")
print(df.first())

print("Showing the first two rows of the DataFrame:")
print(df.take(2))

print("Filtering for rows whose age is above 21:")
df.filter(df['age'] > 21).show()

print("Print the schema of the df:")
df.printSchema()
