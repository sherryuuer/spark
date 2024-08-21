import time
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Read the employers DataFrame
print("Import employers.csv files into a PySpark DataFrame")
employersDF = spark.read.csv("employers.csv", header=True, inferSchema=True)

# Define a function to measure execution time with different partitions
print("Define a function to measure execution time with various partition sizes")


def test_partition_size(partitions):
    employersDF_repartitioned = employersDF.repartition(partitions)

    start_time = time.time()
    employersDF_repartitioned.filter(
        employersDF_repartitioned.employees > 5000).count()
    end_time = time.time()

    execution_time = end_time - start_time
    return execution_time


# List of partition sizes to experiment with
print("Different partition sizes used for experimentation")
partition_sizes = [1, 2, 4, 8, 16]

# Experiment and record results
print("Record results")
results = {}
for partitions in partition_sizes:
    execution_time = test_partition_size(partitions)
    results[partitions] = execution_time

# Print results and identify the best partition size
print("Partition Sizes and Execution Times:")
for partitions, execution_time in results.items():
    print(f"Partitions: {partitions}, Execution Time: {
          round(execution_time, 2)} seconds")

best_partition_size = min(results, key=results.get)
print(f"\nThe Best Partition Size for Optimal Performance: {
      best_partition_size}")
