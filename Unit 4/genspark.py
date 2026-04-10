import numpy as np 
import pandas as pd 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder \
    .appName("Anomaly_Integrity_Audit") \
    .getOrCreate()
df = spark.read.csv("/kaggle/input/datasets/devrajnsut/anamoly-results/model_vs_logic_comparison.csv", header=True, inferSchema=True)

logic_victory_df = df.filter(
    (col("label") == 1) & 
    (col("logic_detected") == 1) & 
    (col("nn_detected") == 0)
)

# Zone B: NN Wins (NN found it, Logic missed it)
nn_victory_df = df.filter(
    (col("label") == 1) & 
    (col("nn_detected") == 1) & 
    (col("logic_detected") == 0)
)
print("--- AUDIT REPORT: SYSTEM DISCREPANCIES ---")

print(f"Total High-Criticality Misses by NN (Logic Saved the Day): {logic_victory_df.count()}")
# Show the nature of data where Logic is superior (Deterministic Thresholds)
logic_victory_df.select("max_temp_recorded", "Actual_Data_Type").show(6)

print(f"Total Pattern Misses by Logic (NN Saved the Day): {nn_victory_df.count()}")
# Show where NN is superior (Subtle Pattern Detection)
nn_victory_df.select("t1", "t2", "t3", "t4", "t5", "Actual_Data_Type").show(5)

from pyspark.sql import functions as F

# 5. Distributed Aggregation for the Research Paper
# We calculate the average detection rates grouped by the data type
summary = df.groupBy("Actual_Data_Type").agg({
    "nn_detected": "avg", 
    "logic_detected": "avg"
}).withColumnRenamed("avg(nn_detected)", "NN_Success_Rate") \
  .withColumnRenamed("avg(logic_detected)", "Logic_Success_Rate")

# To remove the last row, we first calculate the total count of groups
total_rows = summary.count()

# Use limit to take all rows except the last one
# Note: In Spark, 'last' refers to the last row in the current partition order
summary_final = summary.limit(total_rows - 1)

# Display the resulting distributed computation
summary_final.show()