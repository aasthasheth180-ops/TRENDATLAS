import os
import json
import boto3
from dotenv import load_dotenv

from ai.build_fact_package import build_fact_package

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

SERVING_PREFIX = "serving/trendatlas/"
AI_FACT_PACKAGE_KEY = f"{SERVING_PREFIX}ai_fact_package.json"


def upload_json_to_s3(data, key):
    """
    Upload a JSON-serializable serving artifact to S3.
    """
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET environment variable is not configured.")

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
    )

    body = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def build_ai_fact_package_artifact():
    """
    Build the expensive AI fact package once and persist the compact
    result for the Streamlit serving layer.
    """
    print("Building TrendAtlas AI fact package...")

    fact_package = build_fact_package()

    upload_json_to_s3(
        fact_package,
        AI_FACT_PACKAGE_KEY,
    )

    print(
        f"Uploaded AI fact package to "
        f"s3://{S3_BUCKET}/{AI_FACT_PACKAGE_KEY}"
    )


def main():
    build_ai_fact_package_artifact()
    print("TrendAtlas serving artifacts built successfully.")


if __name__ == "__main__":
    main()