"""Register an existing SageMaker model package without embedding cloud credentials."""

from __future__ import annotations

import os


def main() -> None:
    import boto3  # Optional cloud-only dependency.
    group = os.getenv("CLAIMIQ_MODEL_PACKAGE_GROUP")
    model_data = os.getenv("CLAIMIQ_MODEL_DATA_URL")
    image_uri = os.getenv("CLAIMIQ_INFERENCE_IMAGE_URI")
    if not all([group, model_data, image_uri]):
        raise ValueError("Set CLAIMIQ_MODEL_PACKAGE_GROUP, CLAIMIQ_MODEL_DATA_URL, and CLAIMIQ_INFERENCE_IMAGE_URI.")
    client = boto3.client("sagemaker", region_name=os.getenv("AWS_REGION"))
    response = client.create_model_package(
        ModelPackageGroupName=group,
        InferenceSpecification={"Containers": [{"Image": image_uri, "ModelDataUrl": model_data}], "SupportedContentTypes": ["application/json"], "SupportedResponseMIMETypes": ["application/json"]},
    )
    print(response["ModelPackageArn"])


if __name__ == "__main__":
    main()
