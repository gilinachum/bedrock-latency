# Latency Benchmarking tools for Amazon Bedrock
A collection of tools to measure inference latency for foundations models in Amazon Bedrock and OpenAI. Reports time to first token and total time.
1. [bedrock-latency-benchmark.ipynb](./bedrock-latency-benchmark.ipynb) - Measure LLM single request latency across scenarios like:
    - Different number of in/out tokens
    - Compare latency across models from Amazon Bedrock and OpenAI.
    - Latency for the same model across AWS Regions.
2. [stress.test.bedrock.py](./stress.test.bedrock.py) - A utility to stress test Claude 3 models on Bedrock with high number of concurrent requests (e.g., launch 1000 requests/min).
<<<<<<< HEAD
3. [clock-based-rolling-window-rpm-tpm-tester](./clock-based-rolling-window-rpm-tpm-tester/) - Test whether Bedrock TPM limits are based on absolute minutes (clock-based) or relative minutes (rolling window). Determines if quota resets at xx:00 or 60 seconds after first usage.
=======
3. [service_tier_benchmark.py](./service_tier_benchmark.py) - Compare latency across Bedrock service tiers (Priority, Standard, Flex) using the Converse API. See [SERVICE_TIER_TESTING.md](./SERVICE_TIER_TESTING.md) for details.
>>>>>>> 3cb5806 (feat: Add service tier latency benchmarking tool)
## Installing
1. `git clone https://github.com/gilinachum/bedrock-latency`
2. Open relevant notebook.
## License
[Apache License Version 2.0](LICENSE)
