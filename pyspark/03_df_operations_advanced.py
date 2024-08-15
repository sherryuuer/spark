from pyspark.sql.functions import lead, lag, ntile
from pyspark.sql.functions import rank, dense_rank, row_number
from pyspark.sql.functions import avg
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

# join
spark = SparkSession.builder.appName("DataFrameActions").getOrCreate()

print(f'Import "file.csv" into PySpark DataFrame')
df1 = spark.read.csv("file.csv", header=True, inferSchema=True)

print(f'Import "file2.csv" into PySpark DataFrame')
df2 = spark.read.csv("file2.csv", header=True, inferSchema=True)

print("Join dataframe 2 with DataFrame 1 using inner")
joined_df = df1.join(df2, on="id", how="inner")

print("Showing the joined DataFrame using inner join:")
joined_df.show()

print("Join dataframe 2 with DataFrame 1 using outer")
joined_df = df1.join(df2, on="id", how="outer")

print("Showing the joined DataFrame using outer join:")
joined_df.show()

# window functions
# aggregate: sum(), avg(), max(), min()
spark = SparkSession.builder.getOrCreate()

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("file.csv", header=True, inferSchema=True)

print("Printing the contents of the DataFrame")
df.show()

print("Create a `WindowSpec` object")
windowSpec = Window.partitionBy('department')

print("Printing the output from Windows function")
df.withColumn("avg_salary", avg('salary').over(windowSpec)).show()

# ranking: rank(), dense_rank(), row_number()
spark = SparkSession.builder.getOrCreate()

print("Create a `WindowSpec` object")
windowSpec = Window.partitionBy('sales_person').orderBy('sales_amount')

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("sales.csv", header=True, inferSchema=True)

print("Assigning rank to each row:")
df.withColumn('rank', rank().over(windowSpec)).show()

print("Assigning dense rank to each row:")
df.withColumn('dense_rank', dense_rank().over(windowSpec)).show()

print("Assigning row number to each row:")
df.withColumn('row_number', row_number().over(windowSpec)).show()

# analytic:
spark = SparkSession.builder.getOrCreate()

print("Create a `WindowSpec` object")
windowSpec = Window.partitionBy('sales_person').orderBy('sales_amount')

print(f'Import "file.csv" into PySpark DataFrame')
df = spark.read.csv("sales.csv", header=True, inferSchema=True)

print("Next month's sales for each category:")
df.withColumn(
    'next_month_sales',
    lead('sales_amount', 1).over(windowSpec)
).show()

print("Previous month's sales for each category:")
df.withColumn(
    'prev_month_sales',
    lag('sales_amount', 1).over(windowSpec)
).show()

print("Quartiles of sales for each category:")
df.withColumn('quartile', ntile(4).over(windowSpec)).show()
