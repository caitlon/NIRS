#!/usr/bin/env python
"""
Script to run an MLflow tracking server.

This script starts an MLflow tracking server with options for local or
remote storage of tracking data and artifacts.

Examples:
    # Run with local storage:
    $ python run_mlflow_server.py --host 0.0.0.0 --port 5000

    # Run with S3 artifact storage:
    $ python run_mlflow_server.py --backend-store-uri sqlite:///mlflow.db --artifacts-uri s3://my-bucket/mlflow-artifacts

    # Run with MinIO (S3-compatible) storage:
    $ python run_mlflow_server.py --backend-store-uri sqlite:///mlflow.db --artifacts-uri s3://mlflow --endpoint-url http://localhost:9000
"""  

import argparse
import logging
import os
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an MLflow tracking server."
    )

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )

    # Storage configuration
    parser.add_argument(
        "--backend-store-uri",
        type=str,
        default=None,
        help="URI for backend store (e.g., sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--default-artifact-root",
        type=str,
        default=None,
        help="Default artifact root (e.g., ./mlruns)",
    )

    # S3 specific configuration
    parser.add_argument(
        "--artifacts-uri",
        type=str,
        default=None,
        help="S3/MinIO URI for artifact store (e.g., s3://my-bucket/path)",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="Custom endpoint URL for S3-compatible storage (e.g., MinIO)",
    )

    return parser.parse_args()


def setup_s3_env(artifacts_uri, endpoint_url=None):
    """Set up environment variables for S3 access."""
    if not artifacts_uri.startswith("s3://"):
        logger.error("Artifacts URI must start with 's3://' for S3 storage")
        sys.exit(1)

    # Set MinIO/S3 endpoint if provided
    if endpoint_url:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint_url
        logger.info(f"Set S3 endpoint URL to {endpoint_url}")

    # Check if AWS credentials are set
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        logger.warning(
            "AWS_ACCESS_KEY_ID and/or AWS_SECRET_ACCESS_KEY not set in environment. "  
            "Make sure they are set for S3 access to work properly."
        )


def main():
    """Main function to run MLflow server."""
    args = parse_args()

    # Build command
    cmd = ["mlflow", "server"]

    # Add host and port
    cmd.extend(["--host", args.host])
    cmd.extend(["--port", str(args.port)])

    # Add backend store URI if provided
    if args.backend_store_uri:
        cmd.extend(["--backend-store-uri", args.backend_store_uri])

    # Add default artifact root if provided
    if args.default_artifact_root:
        cmd.extend(["--default-artifact-root", args.default_artifact_root])
    elif args.artifacts_uri:
        cmd.extend(["--default-artifact-root", args.artifacts_uri])

        # Set up S3 environment if using S3
        if args.artifacts_uri.startswith("s3://"):
            setup_s3_env(args.artifacts_uri, args.endpoint_url)

    # Log the command
    logger.info(f"Starting MLflow server with command: {' '.join(cmd)}")

    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start MLflow server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("MLflow server stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
