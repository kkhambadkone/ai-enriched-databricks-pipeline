"""
run_pipeline_local.py

Runs the AI-enriched pipeline from your local machine (Mac) while every
Spark operation and AI Function call actually executes on Databricks
serverless compute, via Databricks Connect. Only this driver script runs
locally - all data, Delta tables, and AI Function calls live in your
Databricks workspace / Unity Catalog.

One-time setup:
  1. brew tap databricks/tap && brew install databricks
  2. databricks configure --host https://<your-workspace-url>
     (paste your personal access token when prompted - Settings > Developer
     > Access tokens > Generate new token in the Databricks UI)
  3. pip install databricks-connect
     (match the version to your workspace's serverless environment if you
     hit connection errors - check Environment version in a notebook)

Then just run:
  python run_pipeline_local.py
"""

import random
from datetime import datetime, timedelta

from databricks.connect import DatabricksSession
from pyspark.sql import Row, functions as F

CATALOG_NAME = "main"  # if this fails on permissions, check SHOW CATALOGS in your workspace
SCHEMA_NAME = "ai_pipeline_demo"


def get_spark():
    # Uses the DEFAULT profile from ~/.databrickscfg by default.
    # Pass profile="your-profile-name" to DatabricksSession.builder if you
    # configured a non-default profile.
    return DatabricksSession.builder.serverless(True).getOrCreate()


def build_bronze(spark):
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
    print(f"[bronze] {orders_df.count()} orders, {reviews_df.count()} reviews written to Unity Catalog")


def build_silver(spark):
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
    print(f"[silver] {silver_df.count()} customers summarized")


def enrich_with_ai(spark):
    spark.sql("""
        CREATE OR REPLACE TABLE reviews_enriched AS
        SELECT
            order_id, customer_id, product, review_text, review_date,
            ai_analyze_sentiment(review_text) AS sentiment,
            ai_classify(review_text, ARRAY('product quality', 'shipping', 'customer service', 'price')) AS category,
            ai_summarize(review_text) AS review_summary
        FROM bronze_reviews
    """)
    print("[ai] reviews enriched with sentiment, category, summary")


def build_gold(spark):
    spark.sql("""
        CREATE OR REPLACE TABLE gold_customer_analytics AS
        SELECT
            s.customer_id, s.total_orders, s.total_spend, s.avg_order_value, s.last_order_date,
            COUNT(r.order_id) AS review_count,
            SUM(CASE WHEN r.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative_review_count
        FROM silver_customer_summary s
        LEFT JOIN reviews_enriched r ON s.customer_id = r.customer_id
        GROUP BY s.customer_id, s.total_orders, s.total_spend, s.avg_order_value, s.last_order_date
    """)
    print("[gold] gold_customer_analytics built")


def draft_outreach(spark):
    unhappy = spark.sql("""
        SELECT customer_id, negative_review_count
        FROM gold_customer_analytics
        WHERE negative_review_count >= 2
        ORDER BY negative_review_count DESC
        LIMIT 5
    """).collect()

    for row in unhappy:
        try:
            draft = spark.sql(f"""
                SELECT ai_query(
                    'databricks-meta-llama-3-3-70b-instruct',
                    'Write a short, empathetic customer service outreach email to a customer with ID {row.customer_id} who has left {row.negative_review_count} negative reviews. Keep it under 100 words.'
                ) AS draft_email
            """).collect()[0]["draft_email"]
            print(f"\n--- Draft for {row.customer_id} ---\n{draft}")
        except Exception as e:
            print(f"[ai_query skipped for {row.customer_id}] {e} "
                  f"(check Serving in your workspace for the exact model name available to you)")


def run_pipeline(spark):
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
    spark.sql(f"USE CATALOG {CATALOG_NAME}")
    spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

    build_bronze(spark)
    build_silver(spark)
    enrich_with_ai(spark)
    build_gold(spark)
    draft_outreach(spark)


if __name__ == "__main__":
    spark = get_spark()
    run_pipeline(spark)
    print(f"\nPipeline complete. Check {CATALOG_NAME}.{SCHEMA_NAME} in the Databricks workspace UI.")
