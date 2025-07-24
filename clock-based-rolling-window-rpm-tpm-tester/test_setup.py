#!/usr/bin/env python3
"""
Quick test to verify AWS credentials and Bedrock access
"""

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from utils import BedrockTPMTester

def test_aws_credentials():
    """Test if AWS credentials are available."""
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            return False, "No AWS credentials found"
        
        # Try to get caller identity
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        return True, f"AWS Account: {identity.get('Account')}, User: {identity.get('Arn')}"
    except Exception as e:
        return False, str(e)

def test_bedrock_connection():
    """Test basic Bedrock connection and model access."""
    try:
        # First test credentials
        creds_ok, creds_msg = test_aws_credentials()
        if not creds_ok:
            print(f"✗ AWS Credentials Error: {creds_msg}")
            return False
        else:
            print(f"✓ AWS Credentials: {creds_msg}")
        
        tester = BedrockTPMTester()
        print(f"✓ Bedrock client initialized successfully")
        print(f"✓ Using model: {tester.model_id}")
        
        # Test a small request
        print("Testing small request...")
        status_code, error, response_time = tester.make_bedrock_request(100)
        
        if status_code == 200:
            print(f"✓ Test request successful (response time: {response_time:.2f}s)")
            return True
        else:
            print(f"✗ Test request failed with status {status_code}: {error}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Bedrock TPM Test Setup")
    print("=" * 40)
    
    success = test_bedrock_connection()
    
    if success:
        print("\n✓ Setup test passed! Ready to run TPM limit tests.")
        print("You can now run the Jupyter notebook: bedrock_tpm_test.ipynb")
    else:
        print("\n✗ Setup test failed. Please check:")
        print("  - AWS credentials are configured")
        print("  - You have access to Bedrock in your region")
        print("  - Nova Pro model is available in your account")