# Bedrock TPM Limit Testing: Absolute vs Relative Minutes

This project tests whether AWS Bedrock's TPM (Tokens Per Minute) limits are based on:
- **Absolute minutes**: Quota resets at the start of each clock minute (xx:00)
- **Relative minutes**: Quota resets 60 seconds after first token usage

## Overview

AWS Bedrock enforces TPM limits to control usage, but the exact timing mechanism isn't clearly documented. This test determines whether the quota operates on:

1. **Clock-based (Absolute)**: Quota resets every minute at xx:00 seconds
2. **Rolling window (Relative)**: Quota resets 60 seconds after first consumption

## Test Strategy

The test uses Nova Pro model (8,000 TPM limit) with the following sequence:

1. **xx:35** - Consume all 8,000 TPM quota in chunks
2. **xx:45-59** - Make 1000-token request (should get 429 throttled)
3. **yy:05** - Make 1000-token request (KEY DIFFERENTIATOR)
4. **yy:50** - Make 1000-token request (should always succeed)

### Expected Results

| Timing Model | Request at yy:05 | Conclusion |
|--------------|------------------|------------|
| **Absolute** | 200 OK | Quota reset at new minute |
| **Relative** | 429 Throttled | Must wait full 60 seconds |

## Files

- `utils.py` - Core testing utilities with BedrockTPMTester class
- `bedrock_tpm_test.ipynb` - Interactive Jupyter notebook for running tests
- `test_setup.py` - Setup verification script
- `code_generation_prompt.md` - Original requirements and prompt

## Setup

### Prerequisites

- Python 3.8+
- AWS credentials configured
- Access to AWS Bedrock Nova Pro model
- 8,000 TPM quota available

### Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install boto3 jupyter ipython

# Verify setup
python test_setup.py
```

## Usage

### Quick Test

```bash
python test_setup.py
```

This verifies:
- AWS credentials are working
- Bedrock access is available
- Nova Pro model is accessible

### Full TPM Test

```bash
jupyter notebook bedrock_tpm_test.ipynb
```

Run all cells in sequence. The notebook will:
1. Wait for precise timing (xx:35)
2. Consume TPM quota systematically
3. Test quota behavior at key intervals
4. Analyze results and provide conclusion

## Key Features

### Precise Timing Control
- Waits for exact seconds within minutes
- Ensures consistent test conditions

### Token Padding
- Uses '0' characters to reach target token counts
- Roughly 4 characters per token estimation

### No Automatic Retries
- Boto3 retries disabled for immediate throttling detection
- Essential for accurate 429 response timing

### Comprehensive Logging
- Tracks all requests with precise timestamps
- Records tokens, status codes, and errors

### Automatic Analysis
- Determines absolute vs relative based on results
- Provides clear conclusion with reasoning

## Technical Details

### BedrockTPMTester Class

```python
# Initialize with retry disabled
tester = BedrockTPMTester(region_name='us-east-1')

# Consume quota in chunks
results = tester.consume_tpm_quota(total_tokens=8000, chunk_size=1000)

# Make individual test requests
status_code, error, response_time = tester.make_bedrock_request(1000)
```

### Timing Functions

```python
# Wait for specific second
tester.wait_until_second(35)

# Log results with timestamps
tester.log_result("test_name", timestamp, tokens, status_code, error)
```

## Troubleshooting

### Common Issues

1. **Credentials Error**
   ```
   Error when retrieving credentials from custom-process
   ```
   - Configure AWS credentials: `aws configure`
   - Or set environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

2. **Model Access Denied**
   ```
   AccessDeniedException: User is not authorized
   ```
   - Enable Nova Pro model in Bedrock console
   - Check IAM permissions for bedrock:InvokeModel

3. **Quota Already Exhausted**
   ```
   ThrottlingException: Too many requests
   ```
   - Wait for quota to reset
   - Check current usage in AWS console

### Debug Mode

Add debug logging to see detailed request/response info:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Results Interpretation

The test output will show:

```
TEST SUMMARY
============
Test: Consume_batch_1
  Time: 14:35:01 (minute 35, second 1)
  Tokens: 1000
  Status: 200

Test: Test_at_next_minute_05
  Time: 14:36:05 (minute 36, second 5)  
  Tokens: 1000
  Status: 200  # <-- KEY RESULT

CONCLUSION:
✓ TPM limits appear to be based on ABSOLUTE MINUTES
  The quota reset at the start of a new clock minute
```

## Contributing

When modifying the test:

1. Maintain precise timing requirements
2. Keep token calculations accurate
3. Preserve comprehensive logging
4. Test with different regions/models as needed

## License

This project is for testing and educational purposes. Follow AWS service terms and usage policies.