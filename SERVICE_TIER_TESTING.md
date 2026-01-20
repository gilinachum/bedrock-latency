# Service Tier Latency Testing

This document describes how to test and compare latency across Amazon Bedrock service tiers.

## Overview

Amazon Bedrock offers four service tiers for model inference:

- **Reserved**: Dedicated capacity with 99.5% uptime target (requires AWS account team contact)
- **Priority**: Fastest response times for mission-critical applications (premium pricing)
- **Standard**: Consistent performance for everyday AI tasks (default tier)
- **Flex**: Cost-effective processing with longer processing times (discounted pricing)

## Service Tier Benchmark Script

The `service_tier_benchmark.py` script compares latency across Priority, Standard, and Flex tiers using the Converse API.

### Features

- Tests all three on-demand service tiers (Priority, Standard, Flex)
- Uses Converse API with proper `serviceTier` parameter
- Generates prompts with configurable token counts
- Collects detailed latency metrics (mean, median, p95, p99, etc.)
- Outputs results to CSV for further analysis
- Creates box plot visualization comparing tiers
- Tracks resolved service tier from API responses

### Requirements

```bash
pip install boto3 botocore matplotlib numpy tqdm
```

### Usage

**Basic test with defaults (1000 input tokens, 100 output tokens, 20 invocations per tier):**
```bash
python service_tier_benchmark.py
```

**Custom configuration:**
```bash
python service_tier_benchmark.py \
  --model-id us.amazon.nova-2-lite-v1:0 \
  --region us-east-1 \
  --input-tokens 1000 \
  --output-tokens 100 \
  --invocations 30 \
  --sleep 1 \
  --randomize \
  --output-csv service_tier_results.csv \
  --output-chart service_tier_chart.png
```

**Quick test (fewer invocations):**
```bash
python service_tier_benchmark.py --invocations 10
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-id` | `amazon.nova-2-lite-v1:0` | Bedrock model ID to test |
| `--region` | `us-east-1` | AWS region |
| `--input-tokens` | `1000` | Target number of input tokens |
| `--output-tokens` | `100` | Maximum output tokens |
| `--invocations` | `20` | Number of test invocations per tier |
| `--sleep` | `1` | Seconds to sleep between invocations |
| `--output-csv` | `service_tier_results.csv` | CSV output file path |
| `--output-chart` | `service_tier_comparison.png` | Chart output file path |
| `--log-level` | `INFO` | Logging level |

### Output

The script produces:

1. **CSV file** with detailed results for each invocation:
   - Service tier requested and resolved
   - Latency measurements
   - Token counts (input, output, total)
   - Success/failure status
   - Error messages (if any)

2. **Box plot visualization** comparing latency distributions across tiers

3. **Console summary** with statistics:
   - Success rates
   - Mean, median, min, max latency
   - Standard deviation
   - P95 and P99 latency
   - Tier-to-tier comparisons

### Example Output

```
SERVICE TIER BENCHMARK SUMMARY
================================================================================

PRIORITY Tier:
  Total Requests:    20
  Successful:        20
  Success Rate:      100.0%
  Mean Latency:      1.234s
  Median Latency:    1.210s
  Min Latency:       1.050s
  Max Latency:       1.450s
  Std Dev:           0.089s
  P95 Latency:       1.398s
  P99 Latency:       1.445s

STANDARD Tier:
  Total Requests:    20
  Successful:        20
  Success Rate:      100.0%
  Mean Latency:      1.567s
  Median Latency:    1.543s
  Min Latency:       1.320s
  Max Latency:       1.890s
  Std Dev:           0.134s
  P95 Latency:       1.823s
  P99 Latency:       1.878s

FLEX Tier:
  Total Requests:    20
  Successful:        20
  Success Rate:      100.0%
  Mean Latency:      2.123s
  Median Latency:    2.089s
  Min Latency:       1.780s
  Max Latency:       2.650s
  Std Dev:           0.198s
  P95 Latency:       2.534s
  P99 Latency:       2.623s

--------------------------------------------------------------------------------
TIER COMPARISON:
--------------------------------------------------------------------------------
Priority vs Standard: -21.2% latency change
Flex vs Standard:     +35.5% latency change
================================================================================
```

## Supported Models

The following models support Priority and Flex tiers (as of documentation):

### Amazon Nova Models
- `amazon.nova-2-lite-v1:0` ✅ (tested in this script)
- `amazon.nova-2-pro-preview-20251202-v1:0`
- `amazon.nova-2-lite-omni-v1`
- `amazon.nova-pro-v1:0`
- `amazon.nova-premier-v1:0`

### Other Providers
- OpenAI GPT-OSS models
- Qwen models
- DeepSeek V3.1
- Google Gemma 3 models
- Minimax M2
- Mistral models
- Kimi K2 Thinking
- NVIDIA Nemotron Nano 2

Check the [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html) for the complete list and regional availability.

## Important Notes

1. **Model Access**: Ensure the model is enabled in your AWS region via the Bedrock console
2. **Quotas**: On-demand quota is shared across Priority, Standard, and Flex tiers
3. **Pricing**: Priority tier costs more than Standard; Flex tier is discounted
4. **Regional Availability**: Not all models support service tiers in all regions
5. **Reserved Tier**: Requires contacting AWS account team; not included in this script

## Interpreting Results

### Expected Behavior

- **Priority**: Should show ~25% better latency than Standard (per AWS documentation)
- **Standard**: Baseline performance
- **Flex**: Higher latency than Standard, suitable for non-time-critical workloads

### Factors Affecting Results

- Time of day (load varies by region timezone)
- Network latency from your location to AWS region
- Model availability and current demand
- Input/output token counts
- Concurrent usage in your account

### Best Practices

1. Run tests during different times of day to see variance
2. Use sufficient invocations (20-50) for statistical significance
3. Test from the same network location for consistency
4. Consider testing multiple regions if your application is multi-region
5. Monitor CloudWatch metrics for additional insights

## Troubleshooting

### "Model not found" error
- Verify model ID is correct
- Check model is enabled in Bedrock console for your region
- Confirm region supports the model

### "Throttling" errors
- Reduce invocations or increase sleep time
- Check your account quotas in Bedrock console
- Consider testing during off-peak hours

### "Service tier not supported" error
- Verify the model supports Priority/Flex tiers
- Check regional availability in AWS documentation
- Ensure boto3 is up to date: `pip install --upgrade boto3`

## Integration with Existing Tools

This script complements the existing benchmarking tools:

- `bedrock-latency-benchmark.ipynb`: Interactive analysis across models/regions
- `stress.test.bedrock.py`: High-volume concurrent stress testing
- `service_tier_benchmark.py`: Service tier comparison (this script)

Use them together for comprehensive latency analysis.
