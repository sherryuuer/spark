# create from RDDs
from pyspark.sql import Row
import pandas as pd
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("pyspark_sql").getOrCreate()

print("Create a sample RDD")
rdd = spark.sparkContext.parallelize(
    [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
)

print("Create a PySpark DataFrame from RDD")
df = spark.createDataFrame(rdd, ["id", "name"])

print("Print the contents of the DataFrame")
df.show()

# create from csv file
spark = SparkSession.builder.appName("pyspark_sql").getOrCreate()

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("file.csv", header=True, inferSchema=True)

print("Print the contents of the PySpark DataFrame")
df.show(10)

# create from pandas dataframes
spark = SparkSession.builder.appName("pyspark_sql").getOrCreate()

print("Create a Python dictionary")
data = {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35]
}

print("Create a Pandas DataFrame")
pandas_df = pd.DataFrame(data)

print("Convert Pandas DataFrame to PySpark DataFrame")
# for row in pandas_df.to_dict(orient='records'):
#     print(Row(**row))
# Row(id=1, name='Alice', age=25)
# Row(id=2, name='Bob', age=30)
# Row(id=3, name='Charlie', age=35)
rows = [Row(**row) for row in pandas_df.to_dict(orient='records')]
df = spark.createDataFrame(rows)

print("Print the contents of the PySpark DataFrame")
df.show()
