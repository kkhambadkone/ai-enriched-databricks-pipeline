# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Enriched Customer Analytics Pipeline
# MAGIC PySpark + Unity Catalog + Databricks AI Functions.
# MAGIC Bronze (raw synthetic orders/reviews) -> Silver (customer summary) ->
# MAGIC AI enrichment (sentiment/classification/summary) -> Gold (joined analytics) ->
# MAGIC optional ai_query outreach drafts for unhappy customers.
# MAGIC
# MAGIC Runs entirely on serverless compute - no cluster configuration needed.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------
CATALOG_NAME = "main"  # if this fails on permissions, run SHOW CATALOGS; and swap in one that exists
SCHEMA_NAME = "ai_pipeline_demo"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Bronze layer: synthetic orders + reviews

# COMMAND ----------
import random
from datetime import datetime, timedelta
from pyspark.sql import Row

random.seed(42)

customers = [f"CUST{str(i).zfill(4)}" for i in range(1, 51)]
products = ["Wireless Mouse", "Mechanical Keyboard", "USB-C Hub",
            "Laptop Stand", "Webcam HD", "Noise Cancelling Headphones"]

review_templates = [
    "Absolutely love this {product}, works perfectly and arrived fast!",
    "The {product} broke after two days, very disappointed.",
    "Decent {product} for the price, does what it says.",
    "Terrible experience, {product} was defective on arrival.",
    "Best {product} I've bought all year, highly recommend!",
    "Average quality, {product} is okay but nothing special.",
    "Customer service was unhelpful when my {product} stopped working.",
    "Five stars, this {product} exceeded my expectations.",
]

base_date = datetime(2026, 1, 1)
order_rows, review_rows = [], []

for i in range(500):
    cust = random.choice(customers)
    product = random.choice(products)
    order_date = base_date + timedelta(days=random.randint(0, 200))
    amount = round(random.uniform(15, 250), 2)
    order_id = f"ORD{str(i).zfill(5)}"
    order_rows.append(Row(order_id=order_id, customer_id=cust, product=product,
                           order_date=order_date, amount=amount))
    if random.random() < 0.6:
        review_text = random.choice(review_templates).format(product=product)
        review_rows.append(Row(order_id=order_id, customer_id=cust, product=product,
                                review_text=review_text,
                                review_date=order_date + timedelta(days=random.randint(1, 14))))

orders_df = spark.createDataFrame(order_rows)
reviews_df = spark.createDataFrame(review_rows)

orders_df.write.format("delta").mode("overwrite").saveAsTable("bronze_orders")
reviews_df.write.format("delta").mode("overwrite").saveAsTable("bronze_reviews")

print(f"Bronze tables written: {orders_df.count()} orders, {reviews_df.count()} reviews")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Silver layer: customer order summary

# COMMAND ----------
from pyspark.sql import functions as F

silver_df = (
    spark.table("bronze_orders")
    .groupBy("customer_id")
    .agg(
        F.count("order_id").alias("total_orders"),
        F.round(F.sum("amount"), 2).alias("total_spend"),
        F.round(F.avg("amount"), 2).alias("avg_order_value"),
        F.max("order_date").alias("last_order_date"),
    )
)

silver_df.write.format("delta").mode("overwrite").saveAsTable("silver_customer_summary")
print(f"Silver table written: {silver_df.count()} customers")

# COMMAND ----------
# MAGIC %md
# MAGIC ## AI enrichment: sentiment, classification, summary
# MAGIC Uses Databricks AI Functions - no API keys required.

# COMMAND ----------
spark.sql("""
CREATE OR REPLACE TABLE reviews_enriched AS
SELECT
    order_id,
    customer_id,
    product,
    review_text,
    review_date,
    ai_analyze_sentiment(review_text) AS sentiment,
    ai_classify(review_text, ARRAY('product quality', 'shipping', 'customer service', 'price')) AS category,
    ai_summarize(review_text) AS review_summary
FROM bronze_reviews
""")

print("Reviews enriched with sentiment, category, and summary")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Gold layer: joined analytics table

# COMMAND ----------
spark.sql("""
CREATE OR REPLACE TABLE gold_customer_analytics AS
SELECT
    s.customer_id,
    s.total_orders,
    s.total_spend,
    s.avg_order_value,
    s.last_order_date,
    COUNT(r.order_id) AS review_count,
    SUM(CASE WHEN r.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_review_count
FROM silver_customer_summary s
LEFT JOIN reviews_enriched r ON s.customer_id = r.customer_id
GROUP BY s.customer_id, s.total_orders, s.total_spend, s.avg_order_value, s.last_order_date
""")

print("Gold table built: gold_customer_analytics")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional: draft outreach emails for unhappy customers using ai_query
# MAGIC The model name may need adjusting based on what's live under Serving in your workspace.

# COMMAND ----------
unhappy_customers = spark.sql("""
SELECT customer_id, negative_review_count
FROM gold_customer_analytics
WHERE negative_review_count >= 2
ORDER BY negative_review_count DESC
LIMIT 5
""")

for row in unhappy_customers.collect():
    try:
        draft = spark.sql(f"""
        SELECT ai_query(
            'databricks-meta-llama-3-3-70b-instruct',
            'Write a short, empathetic customer service outreach email to a customer with ID {row.customer_id} who has left {row.negative_review_count} negative reviews. Keep it under 100 words.'
        ) AS draft_email
        """).collect()[0]["draft_email"]
        print(f"--- Draft for {row.customer_id} ---")
        print(draft)
    except Exception as e:
        print(f"[ai_query skipped for {row.customer_id}] {e}")
