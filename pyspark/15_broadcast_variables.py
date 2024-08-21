# Import all necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast  # Import the broadcast function

# Create a SparkSession using the getOrCreate() function
spark = SparkSession.builder.getOrCreate()

# Read the employers DataFrame
print("Import employers.csv files into a PySpark DataFrame")
employersDF = spark.read.csv("employers.csv", header=True, inferSchema=True)

# Filter the DataFrame for employees greater than 5000 and then use broadcast
# to broadcast the resulting DataFrame to all worker nodes
print("Filtering and broadcasting the DataFrame")
filteredDF = employersDF.filter(employersDF.employees >= 5000)
# Broadcast the filtered DataFrame
print("Broadcast the filtered DataFrame")
broadcastedDF = broadcast(filteredDF)

# Perform the groupBy and count operations on the broadcasted DataFrame
print("Perform some groupby operation and then count the number of countries")
resultDF = broadcastedDF.groupBy("country_territory").count()

# Show the result
print("Print the resulting DataFrame")
resultDF.show()
