# Data exploration and analysis
# preview data
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("data_exploration").getOrCreate()

print(f'Import "obesity.csv" into PySpark DataFrame')
df = spark.read.csv("obesity.csv", header=True, inferSchema=True)

print("Create a Temporary View")
df.createOrReplaceTempView("people")

print("Select and display the first 5 rows and all columns from the temp view")
spark.sql("SELECT * FROM people").show(5)

print("Count the total number of rows in the DataFrame")
total_counts = spark.sql("SELECT COUNT(*) FROM people").first()[0]
print(f'Total counts: {total_counts}')

print("Select and display all data points related to people with obesity")
spark.sql("SELECT * FROM people WHERE Label = 'Obese'").show()

print("Group people by label and display them the results")
spark.sql("SELECT Label, COUNT(*) AS count FROM people GROUP BY Label").show()

# 使用pandas的info或者descirbe可以达到同样的效果
# distinct count values
spark = SparkSession.builder.appName("distinct count value").getOrCreate()

print(f'Import "obesity.csv" into PySpark DataFrame')
df = spark.read.csv("obesity.csv", header=True, inferSchema=True)

print("Create a Temporary View")
df.createOrReplaceTempView("people")

print("Create a variable to capture column names")
columns = df.columns

print("Create an empty dictionary")
distinct_counts = {}

print("Iterate over each column in the DataFrame")
for column in columns:
    query = f"SELECT COUNT(DISTINCT `{column}`) FROM people"
    distinct_count = spark.sql(query).collect()[0][0]
    distinct_counts[column] = distinct_count
if distinct_counts:
    pandas_df = pd.DataFrame.from_dict(
        distinct_counts, orient="index", columns=["Distinct Count"])
    pandas_df.index.name = "Column"
    pandas_df.reset_index(inplace=True)
    print(pandas_df)
else:
    print("No distinct counts found")

# summary statistics
spark = SparkSession.builder.appName("pyspark_sql").getOrCreate()

print(f'Import "obesity.csv" into PySpark DataFrame')
df = spark.read.csv("obesity.csv", header=True, inferSchema=True)

print("Create a Temporary View")
df.createOrReplaceTempView("people")

print("Create a list containing new column names")
numerical_columns = ["age", "weight", "height"]

print("Create a SQL query")
sql_query = f"SELECT 'summary' AS summary, {', '.join(
    [f'min({column}), max({column}), avg({column})' for column in numerical_columns])} FROM people"

print("Convert the output to Pandas")
summary_stats = spark.sql(sql_query).toPandas()

print("Summary stats")
print(summary_stats)
