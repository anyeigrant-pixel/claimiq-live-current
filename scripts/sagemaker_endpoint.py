"""Endpoint deployment placeholder for an approved ClaimIQ model package."""

from __future__ import annotations

import os


def main() -> None:
    import boto3  # Optional cloud-only dependency.
    endpoint = os.getenv("CLAIMIQ_ENDPOINT_NAME")
    model_package_arn = os.getenv("CLAIMIQ_MODEL_PACKAGE_ARN")
    role = os.getenv("CLAIMIQ_SAGEMAKER_ROLE_ARN")
    if not all([endpoint, model_package_arn, role]):
        raise ValueError("Set CLAIMIQ_ENDPOINT_NAME, CLAIMIQ_MODEL_PACKAGE_ARN, and CLAIMIQ_SAGEMAKER_ROLE_ARN.")
    client = boto3.client("sagemaker", region_name=os.getenv("AWS_REGION"))
    model_name = f"{endpoint}-model"
    client.create_model(ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer={"ModelPackageName": model_package_arn})
    client.create_endpoint_config(
        EndpointConfigName=f"{endpoint}-config",
        ProductionVariants=[{"VariantName": "AllTraffic", "ModelName": model_name, "InitialInstanceCount": 1, "InstanceType": "ml.m5.large"}],
    )
    client.create_endpoint(EndpointName=endpoint, EndpointConfigName=f"{endpoint}-config")
    print(f"Deployment requested for {endpoint}")


if __name__ == "__main__":
    main()
