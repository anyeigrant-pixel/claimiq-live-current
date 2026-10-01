# SageMaker deployment scaffold

ClaimIQ remains fully local by default. The scripts in `scripts/` provide credential-free integration points for an AWS deployment:

1. Copy `.env.example` to `.env` and supply environment variables through your shell or secret manager.
2. Run `python scripts/sagemaker_upload.py` to put the generated input data in S3.
3. Package a model artifact with your approved training container, then set `CLAIMIQ_MODEL_DATA_URL` and `CLAIMIQ_INFERENCE_IMAGE_URI`.
4. Run `python scripts/sagemaker_registry.py` to add the package to a Model Registry group.
5. After approval, set `CLAIMIQ_MODEL_PACKAGE_ARN` and run `python scripts/sagemaker_endpoint.py`.

No AWS SDK is required for local ClaimIQ use; `boto3` is imported only when a cloud script is executed. In a production deployment, use IAM roles, least-privilege policies, VPC endpoints, encrypted S3 buckets, Model Monitor, and CloudWatch alarms.
