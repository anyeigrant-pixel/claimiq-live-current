"""Upload local ClaimIQ inputs/artifacts to S3; no credentials are stored in this repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.getenv("CLAIMIQ_S3_BUCKET"))
    parser.add_argument("--prefix", default="claimiq")
    parser.add_argument("--path", default="data/raw/synthetic_claims.csv")
    args = parser.parse_args()
    if not args.bucket or args.bucket.startswith("replace-"):
        raise ValueError("Set CLAIMIQ_S3_BUCKET or pass --bucket.")
    import boto3  # Optional dependency: required only for cloud execution.
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)
    key = f"{args.prefix}/input/{path.name}"
    boto3.client("s3").upload_file(str(path), args.bucket, key)
    print(f"Uploaded {path} to s3://{args.bucket}/{key}")


if __name__ == "__main__":
    main()
