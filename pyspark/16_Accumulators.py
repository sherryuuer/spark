# Import the necessary libraries
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Define the function to determine if a country is the United States
print("Create a function to determine if the country is United States or not")


def is_united_states(country):
    return 1 if country == "United States" else 0


# Create an Accumulator for counting the number of countries
print("Create an accumulator for counting countries")
country_counter = spark.sparkContext.accumulator(0)

# Read the employers DataFrame
print("Import employers into PySpark DataFrame")
ordersDF = spark.read.csv("employers.csv", header=True, inferSchema=True)

# Apply the function to each row using foreach
print("Apply the function to each row")
ordersDF.foreach(lambda row: country_counter.add(
    is_united_states(row["country_territory"])))

# Show the count
print(f"Number of countries meeting the condition: {country_counter.value}")
