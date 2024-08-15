"""
Tasks
To effectively analyze a customer churn dataset using PySpark, we’ll focus on the following tasks:

Loading customer data into a PySpark DataFrame:
1.1 Load the customer data into a PySpark DataFrame from a suitable data source.
1.2 Ensure the data is properly formatted and structured for analysis.

Preprocessing and transformation of data:
2.1 Perform necessary preprocessing steps, such as handling missing values, data type conversions, and data cleansing.
2.2 Transform the data by applying filtering, grouping, aggregating, and feature engineering operations.
2.3 Prepare the data for exploratory data analysis and model building.

Exploratory data analysis (EDA):
3.1 Conduct exploratory data analysis to gain insights into the customer data.
3.2 Perform data visualization and statistical analysis to identify patterns, correlations, and factors that may indicate customer churn.
3.3 Explore relationships between different variables and their impact on churn.
"""
# Import the necessary libraries
from pyspark.sql.functions import col, sum, count, avg, corr
from pyspark.sql import SparkSession

# Create a SparkSession object
spark = SparkSession.builder.getOrCreate()

# Write code to read "churn.csv" that is available in your path in a variable called 'spark_df'. Make sure you include options 'header=T' and 'inferSchema=True'
telco_df = spark.read.csv("churn.csv", header=True, inferSchema=True)

# Write code to look at the first five rows of the "churn_df" DataFrame
telco_df.show(5)

# Write code to check the schema of the "churn_df" DataFrame
telco_df.printSchema()

# Create a SparkSession object
spark = SparkSession.builder.getOrCreate()

# Write code to read "churn.csv" in a variable called 'telco_df'. Make sure you include options 'header=T' and 'inferSchema=True'
telco_df = spark.read.csv("churn.csv", header=True, inferSchema=True)

# Filter the data to select only the customers who have churned and count the number of such customers.
churn_count = telco_df.filter(telco_df["Churn Label"] == 1).count()
print(f'Counting the number of churned customers: {churn_count}')

# Group the data by gender and compute the average monthly charges for each gender
telco_df.groupBy("Gender").agg({"Monthly Charges": "avg"}).show()

# Multiply monthly charges and tenure months columns to get the new column "Total Charges" and print the new column along with the original columns
telco_df.withColumn(
    "Total Charges",
    col("Monthly Charges") * col("Tenure Months")
).show()

# Compute and print the correlation between the monthly charges and the total charges
telco_df.select(corr(col("Total Charges"), col("Monthly Charges"))).show()

# Relationship between customer churn and contract type
(telco_df
 .groupBy("Contract")
 .agg((sum("Churn Value") / telco_df.count()).alias("Aggregated Churn Value"))
 .show())

# Relationship between customer churn and tenure
(telco_df
 .groupBy("Churn Value")
 .agg(avg("Tenure Months").alias("Aggregated Churn Value"))
 .show())

# Relationship between customer churn and payment method
(telco_df
 .groupBy("Payment Method")
 .agg((sum("Churn Value") / telco_df.count()).alias("Aggregated Churn Value"))
 .show())
