import os
import json
import boto3
import pandas as pd
from dotenv import load_dotenv

from ai.build_fact_package import (
    build_fact_package,
    load_raw_history,
    add_prediction_results,
)
from analytics.historical_intelligence import prepare_historical_intelligence

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

SERVING_PREFIX = "serving/trendatlas/"
AI_FACT_PACKAGE_KEY = f"{SERVING_PREFIX}ai_fact_package.json"
CURRENT_PREDICTIONS_KEY = f"{SERVING_PREFIX}current_predictions.json"
LATEST_VIDEO_STATE_KEY = f"{SERVING_PREFIX}latest_video_state.json"
LIFECYCLE_HISTORY_KEY = f"{SERVING_PREFIX}lifecycle_history.json"
CONTENT_REGIONS_KEY = f"{SERVING_PREFIX}content_regions_history.json"
DATA_HEALTH_KEY = f"{SERVING_PREFIX}data_health.json"


def upload_json_to_s3(data, key):
    """Upload a JSON-serializable serving artifact to S3."""
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
        default=str,
    ).encode("utf-8")

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def build_ai_fact_package_artifact():
    """Build and persist the compact AI evidence package."""
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


def build_current_predictions_artifact(raw, intel):
    """Precompute the latest persistence predictions for Streamlit."""
    print("Building TrendAtlas current predictions...")




    scored, prediction_date = add_prediction_results(
        raw,
        intel["latest_video_state"],
    )

    artifact = {
        "prediction_date": str(prediction_date),
        "rows": scored.to_dict(orient="records"),
    }

    upload_json_to_s3(
        artifact,
        CURRENT_PREDICTIONS_KEY,
    )

    print(
        f"Uploaded current predictions to "
        f"s3://{S3_BUCKET}/{CURRENT_PREDICTIONS_KEY}"
    )

def build_latest_video_state_artifact(raw, intel):
    """Precompute the latest video state used by Pulse and Momentum."""
    print("Building TrendAtlas latest video state...")



    latest_state = intel["latest_video_state"].copy()

    artifact = {
        "rows": latest_state.to_dict(orient="records"),
    }

    upload_json_to_s3(
        artifact,
        LATEST_VIDEO_STATE_KEY,
    )

    print(
        f"Uploaded latest video state to "
        f"s3://{S3_BUCKET}/{LATEST_VIDEO_STATE_KEY}"
    )


def build_lifecycle_history_artifact(raw, intel):
    """
    Build the bounded daily history needed by the Lifecycle chapter.

    Keeps one observation per video Ã— region Ã— calendar day instead
    of sending every six-hour historical snapshot to Streamlit.
    """
    print("Building TrendAtlas lifecycle history...")



    history = intel["video_history"].copy()

    history["fetch_date"] = pd.to_datetime(
        history["fetch_date"],
        errors="coerce",
    ).dt.normalize()

    history = history.dropna(
        subset=["id", "region_code", "fetch_date"]
    )

    # Lifecycle calculations operate at video Ã— region Ã— day.
    # Keep the final observation from each collected calendar day.
    daily = (
        history
        .sort_values("fetch_timestamp")
        .drop_duplicates(
            ["id", "region_code", "fetch_date"],
            keep="last",
        )
        .copy()
    )

    # Keep only columns required by the Lifecycle chapter.
    required_columns = [
    "id",
    "region_code",
    "fetch_date",
    "fetch_timestamp",
    "video_title",
    "category_name",
    "trending_rank",
    "view_count",
    ]

    columns = [
        column
        for column in required_columns
        if column in daily.columns
    ]

    daily = daily[columns].copy()

    artifact = {
        "rows": daily.to_dict(orient="records"),
    }

    upload_json_to_s3(
        artifact,
        LIFECYCLE_HISTORY_KEY,
    )

    print(
        f"Lifecycle serving rows: {len(daily):,}"
    )

    print(
        f"Uploaded lifecycle history to "
        f"s3://{S3_BUCKET}/{LIFECYCLE_HISTORY_KEY}"
    )



def build_content_regions_artifact(raw, intel):
    """
    Build the compact historical dataset required by
    Content and Regions without exposing the full intelligence table.
    """
    print("Building TrendAtlas Content + Regions history...")

    history = intel["video_history"].copy()

    required_columns = [
        "id",
        "region_code",
        "fetch_date",
        "category_name",
        "view_velocity",
        "engagement_rate",
        "channel_name",
    ]

    # Preserve channel_id when available because Content prefers
    # the stable identifier over channel name.
    if "channel_id" in history.columns:
        required_columns.append("channel_id")

    columns = [
        column
        for column in required_columns
        if column in history.columns
    ]

    serving = history[columns].copy()

    artifact = {
        "rows": serving.to_dict(orient="records"),
    }

    upload_json_to_s3(
        artifact,
        CONTENT_REGIONS_KEY,
    )

    print(
        f"Content + Regions serving rows: {len(serving):,}"
    )

    print(
        f"Uploaded Content + Regions history to "
        f"s3://{S3_BUCKET}/{CONTENT_REGIONS_KEY}"
    )


def build_data_health_artifact(raw, intel):
    """
    Precompute the historical evidence required by Data Health.

    Raw-history quality metrics are summarized here so Streamlit
    does not need to load the full S3 archive.
    """
    print("Building TrendAtlas Data Health artifact...")

    q = raw.copy()

    q["fetch_timestamp"] = pd.to_datetime(
        q["fetch_timestamp"],
        utc=True,
        errors="coerce",
    )

    q["fetch_date"] = pd.to_datetime(
        q["fetch_date"],
        errors="coerce",
    ).dt.normalize()

    expected_regions = {"CA", "GB", "IN", "US"}

    # ---------------------------------------------------------
    # Region completeness
    # ---------------------------------------------------------

    region_days = (
        q.dropna(subset=["fetch_date", "region_code"])
        .groupby("fetch_date")["region_code"]
        .agg(lambda x: set(x.dropna().astype(str)))
    )

    complete_dates = [
        d
        for d, regions in region_days.items()
        if expected_regions.issubset(regions)
    ]

    latest_complete = (
        max(complete_dates)
        if complete_dates
        else q["fetch_date"].max()
    )

    latest_regions = (
        region_days.get(latest_complete, set())
        if pd.notna(latest_complete)
        else set()
    )

    region_coverage = len(
        expected_regions.intersection(latest_regions)
    )

    # ---------------------------------------------------------
    # Enrichment
    # ---------------------------------------------------------

    channel_fields = [
        "channel_name",
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count",
    ]

    present_channel_fields = [
        column
        for column in channel_fields
        if column in q.columns
    ]

    enrichment_pct = (
        100
        * q[present_channel_fields]
        .notna()
        .all(axis=1)
        .mean()
        if present_channel_fields and len(q)
        else None
    )

    category_pct = (
        100 * q["category_name"].notna().mean()
        if "category_name" in q.columns and len(q)
        else None
    )

    duplicate_count = int(
        q.duplicated(
            subset=[
                "id",
                "region_code",
                "fetch_timestamp",
            ]
        ).sum()
    )

    # ---------------------------------------------------------
    # Market enrichment
    # ---------------------------------------------------------

    market_rows = []

    for region in ["US", "CA", "GB", "IN"]:
        rq = q[q["region_code"] == region]

        region_enrichment = (
            100
            * rq[present_channel_fields]
            .notna()
            .all(axis=1)
            .mean()
            if present_channel_fields and not rq.empty
            else None
        )

        market_rows.append(
            {
                "Market": region,
                "Observations": len(rq),
                "Active days": int(
                    rq["fetch_date"].nunique()
                ),
                "Channel enrichment": region_enrichment,
            }
        )

    # ---------------------------------------------------------
    # Snapshot completion summary
    # ---------------------------------------------------------

    snapshot_counts = (
        q.dropna(
            subset=[
                "region_code",
                "fetch_timestamp",
            ]
        )
        .groupby(
            [
                "region_code",
                "fetch_timestamp",
            ]
        )
        .size()
        .reset_index(
            name="records_received"
        )
    )

    below_expected_snapshots = int(
        (
            snapshot_counts["records_received"] < 100
        ).sum()
    )

    # ---------------------------------------------------------
    # Lifecycle explorer
    # ---------------------------------------------------------

    history = intel["video_history"].copy()

    history["fetch_date"] = pd.to_datetime(
        history["fetch_date"],
        errors="coerce",
    ).dt.normalize()

    history["trending_rank"] = pd.to_numeric(
        history["trending_rank"],
        errors="coerce",
    )

    daily = (
        history
        .sort_values("fetch_timestamp")
        .drop_duplicates(
            [
                "id",
                "region_code",
                "fetch_date",
            ],
            keep="last",
        )
        .copy()
    )

    lifecycle = (
        daily
        .sort_values(
            [
                "id",
                "region_code",
                "fetch_date",
            ]
        )
        .groupby(
            [
                "id",
                "region_code",
            ],
            as_index=False,
        )
        .agg(
            video_title=("video_title", "last"),
            channel_name=("channel_name", "last"),
            category_name=("category_name", "last"),
            first_observed=("fetch_date", "min"),
            last_observed=("fetch_date", "max"),
            observed_lifetime_days=(
                "fetch_date",
                "nunique",
            ),
            entry_rank=("trending_rank", "first"),
            latest_rank=("trending_rank", "last"),
            observations=("fetch_date", "size"),
        )
    )

    lifecycle["entry_rank_band"] = pd.cut(
        lifecycle["entry_rank"],
        bins=[
            0,
            10,
            50,
            100,
            float("inf"),
        ],
        labels=[
            "Top 10",
            "Rank 11â€“50",
            "Rank 51â€“100",
            "101+",
        ],
        include_lowest=True,
    )

    artifact = {
        "latest_complete": str(latest_complete),
        "region_coverage": region_coverage,
        "enrichment_pct": enrichment_pct,
        "category_pct": category_pct,
        "duplicate_count": duplicate_count,
        "below_expected_snapshots": below_expected_snapshots,
        "market_rows": market_rows,
        "lifecycle_rows": lifecycle.to_dict(
            orient="records"
        ),
    }

    upload_json_to_s3(
        artifact,
        DATA_HEALTH_KEY,
    )

    print(
        f"Data Health lifecycle rows: "
        f"{len(lifecycle):,}"
    )

    print(
        f"Uploaded Data Health artifact to "
        f"s3://{S3_BUCKET}/{DATA_HEALTH_KEY}"
    )





def main():
    # AI fact package currently owns its own historical loading path.
    build_ai_fact_package_artifact()

    print(
        "Loading shared TrendAtlas historical intelligence..."
    )

    # One historical load for all remaining serving artifacts.
    raw = load_raw_history()
    intel = prepare_historical_intelligence(raw)

    build_current_predictions_artifact(raw, intel)
    build_latest_video_state_artifact(raw, intel)
    build_lifecycle_history_artifact(raw, intel)

    build_content_regions_artifact(raw, intel)
    build_data_health_artifact(raw, intel)

    print(
        "TrendAtlas serving artifacts built successfully."
    )


if __name__ == "__main__":
    main()
