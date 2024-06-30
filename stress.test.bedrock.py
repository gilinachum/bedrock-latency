"""
Bedrock Stress Testing Utility

This script simulates multiple concurrent requests to perform stress testing
and load testing on Amazon Bedrock, AWS's service for building generative AI
applications.

Usage:
    python bedrock_stress_test.py [options]

Options:
    --threads INT               Number of concurrent threads (default: 10)
    --invocations INT           Invocations per thread (default: 5)
    --sleep FLOAT               Sleep between invocations in ms (default: 0)
    --duration INT              Test duration in seconds (default: None)
    --model-id STRING           Bedrock model ID (default: anthropic.claude-3-haiku-20240307-v1:0)
    --region STRING             Bedrock region (default: us-east-1)
    --max-tokens INT            Maximum tokens for response (default: 500)
    --temperature FLOAT         Sampling temperature (default: 0)
    --system-prompt STRING      System prompt (default: "You are a helpful assistant.")
    --user-message STRING       User message (default: "Explain quantum computing briefly")
    --output-csv STRING         Output CSV file path (default: None)
    --log-level STRING          Logging level (default: INFO)
    --log-file STRING           Log file path (default: None)

Example:
    python stress.test.bedrock.py --threads 10 --invocations 10 --sleep 0 --output-csv results.csv
"""

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median

import boto3
import botocore
from botocore.exceptions import ClientError
from tqdm import tqdm

import logging

# Set up argument parser
parser = argparse.ArgumentParser(description="Bedrock Stress Testing Utility")
parser.add_argument("--threads", type=int, default=10, help="Number of concurrent threads")
parser.add_argument("--invocations", type=int, default=5, help="Invocations per thread")
parser.add_argument("--sleep", type=float, default=0, help="Sleep between invocations in ms")
parser.add_argument("--duration", type=int, default=None, help="Test duration in seconds")
parser.add_argument("--model-id", type=str, default="anthropic.claude-3-haiku-20240307-v1:0", help="Bedrock model ID")
parser.add_argument("--region", type=str, default="us-east-1", help="Bedrock region name")
parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens for response")
parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.", help="System prompt")
parser.add_argument("--user-message", type=str, default="Explain quantum computing briefly", help="User message")
parser.add_argument("--output-csv", type=str, default=None, help="Output CSV file path")
parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
parser.add_argument("--log-file", type=str, default=None, help="Log file path")

args = parser.parse_args()

def log_parameters(args):
    """Log all parameter values at the start of the script."""
    logger.info("Script started with the following parameters:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

# Configure logging
logging.basicConfig(level=args.log_level, filename=args.log_file, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for result tracking
results = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "empty_responses": 0,
    "throttled_requests": 0,
    "response_times": [],
    "errors": {}
}

def get_bedrock_client(region):
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=region,
        config=botocore.config.Config(
            retries=dict(max_attempts=0),
            max_pool_connections=args.threads
        )
    )

def construct_body(messages, system_prompt, max_tokens, temperature):
    return json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        "temperature": temperature,
    })

bedrock = get_bedrock_client(args.region)

def invoke_agent(thread_id):
    messages = [{"role": "user", "content": args.user_message}]
    body = construct_body(messages, args.system_prompt, args.max_tokens, args.temperature)
    
    try:
        start_time = time.time()
        response = bedrock.invoke_model(body=body, modelId=args.model_id)
        end_time = time.time()
        
        response_time = end_time - start_time
        results["response_times"].append(response_time)
        
        response_body = json.loads(response.get('body').read())
        status_code = response['ResponseMetadata']['HTTPStatusCode']
        
        if status_code == 200:
            results["successful_requests"] += 1
            if not response_body:
                results["empty_responses"] += 1
        else:
            results["failed_requests"] += 1
            error_message = f"Error status code: {status_code}"
            results["errors"][error_message] = results["errors"].get(error_message, 0) + 1
        
        logger.debug(f"Thread {thread_id}: Response received in {response_time:.2f} seconds")
    
    except ClientError as err:
        results["failed_requests"] += 1
        error_code = err.response['Error']['Code']
        if 'ThrottlingException' in error_code:
            results["throttled_requests"] += 1
        results["errors"][error_code] = results["errors"].get(error_code, 0) + 1
        logger.warning(f"Thread {thread_id}: ClientError - {error_code}")
    
    except Exception as e:
        results["failed_requests"] += 1
        error_message = str(e)
        results["errors"][error_message] = results["errors"].get(error_message, 0) + 1
        logger.error(f"Thread {thread_id}: Unexpected error - {error_message}")
    
    finally:
        results["total_requests"] += 1

def run_test():
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        if args.duration:
            future_to_thread = {executor.submit(invoke_agent, i): i for i in range(args.threads)}
            with tqdm(total=args.duration, unit="s") as pbar:
                while time.time() - start_time < args.duration:
                    time.sleep(0.1)
                    pbar.update(0.1)
            for future in future_to_thread:
                future.cancel()
        else:
            total_invocations = args.threads * args.invocations
            with tqdm(total=total_invocations, unit="invocations") as pbar:
                for _ in range(args.invocations):
                    futures = [executor.submit(invoke_agent, i) for i in range(args.threads)]
                    for future in futures:
                        future.result()
                        pbar.update(1)
                    if _ < args.invocations - 1:
                        time.sleep(args.sleep / 1000)

    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time

def print_results(total_time):
    print("\nTest Results:")
    print(f"Total requests: {results['total_requests']}")
    print(f"Successful requests: {results['successful_requests']}")
    print(f"Failed requests: {results['failed_requests']}")
    print(f"Empty responses: {results['empty_responses']}")
    print(f"Throttled requests: {results['throttled_requests']}")
    print(f"Total time taken: {total_time:.2f} seconds")
    
    if results['response_times']:
        print(f"Average response time: {mean(results['response_times']):.2f} seconds")
        print(f"Median response time: {median(results['response_times']):.2f} seconds")
        print(f"Min response time: {min(results['response_times']):.2f} seconds")
        print(f"Max response time: {max(results['response_times']):.2f} seconds")
    
    if results['errors']:
        print("\nErrors:")
        for error, count in results['errors'].items():
            print(f"  {error}: {count}")

def save_to_csv(total_time):
    if not args.output_csv:
        return
    
    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Requests', results['total_requests']])
        writer.writerow(['Successful Requests', results['successful_requests']])
        writer.writerow(['Failed Requests', results['failed_requests']])
        writer.writerow(['Empty Responses', results['empty_responses']])
        writer.writerow(['Throttled Requests', results['throttled_requests']])
        writer.writerow(['Total Time (s)', f"{total_time:.2f}"])
        
        if results['response_times']:
            writer.writerow(['Avg Response Time (s)', f"{mean(results['response_times']):.2f}"])
            writer.writerow(['Median Response Time (s)', f"{median(results['response_times']):.2f}"])
            writer.writerow(['Min Response Time (s)', f"{min(results['response_times']):.2f}"])
            writer.writerow(['Max Response Time (s)', f"{max(results['response_times']):.2f}"])
        
        if results['errors']:
            for error, count in results['errors'].items():
                writer.writerow([f'Error: {error}', count])

if __name__ == "__main__":
    log_parameters(args)  # Add this line to log all parameters
    logger.info("Starting Bedrock Stress Test")
    total_time = run_test()
    print_results(total_time)
    save_to_csv(total_time)
    logger.info("Bedrock Stress Test Completed")