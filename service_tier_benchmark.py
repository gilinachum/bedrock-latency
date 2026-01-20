"""
Bedrock Service Tier Latency Benchmark

This script benchmarks the latency differences between Amazon Bedrock service tiers
(Priority, Standard, and Flex) using the Converse API.

Usage:
    python service_tier_benchmark.py [options]

Options:
    --model-id STRING           Bedrock model ID (default: amazon.nova-2-lite-v1:0)
    --region STRING             AWS region (default: us-east-1)
    --input-tokens INT          Target input tokens (default: 1000)
    --output-tokens INT         Max output tokens (default: 100)
    --invocations INT           Number of invocations per tier (default: 20)
    --sleep FLOAT               Sleep between invocations in seconds (default: 1)
    --output-csv STRING         Output CSV file path (default: service_tier_results.csv)
    --output-chart STRING       Output chart file path (default: service_tier_comparison.png)

Example:
    python service_tier_benchmark.py --invocations 30 --output-csv tier_test.csv
"""

import argparse
import csv
import json
import time
import random
from statistics import mean, median, stdev
from typing import List, Dict, Tuple

import boto3
import botocore
from botocore.exceptions import ClientError
from tqdm import tqdm

import logging

# Set up argument parser
parser = argparse.ArgumentParser(description="Bedrock Service Tier Latency Benchmark")
parser.add_argument("--model-id", type=str, default="amazon.nova-2-lite-v1:0", 
                    help="Bedrock model ID")
parser.add_argument("--region", type=str, default="us-east-1", 
                    help="AWS region")
parser.add_argument("--input-tokens", type=int, default=1000, 
                    help="Target input tokens")
parser.add_argument("--output-tokens", type=int, default=100, 
                    help="Max output tokens")
parser.add_argument("--invocations", type=int, default=20, 
                    help="Number of invocations per tier")
parser.add_argument("--sleep", type=float, default=1, 
                    help="Sleep between invocations in seconds")
parser.add_argument("--output-csv", type=str, default="service_tier_results.csv", 
                    help="Output CSV file path")
parser.add_argument("--output-chart", type=str, default="service_tier_comparison.png", 
                    help="Output chart file path")
parser.add_argument("--log-level", type=str, default="INFO", 
                    help="Logging level")
parser.add_argument("--randomize", action="store_true",
                    help="Randomize the order of tier testing to eliminate time-based bias")

args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=args.log_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_bedrock_client(region: str):
    """Create Bedrock runtime client with retries disabled for accurate latency measurement."""
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=region,
        config=botocore.config.Config(
            retries=dict(max_attempts=0)
        )
    )


def generate_prompt(target_tokens: int) -> str:
    """
    Generate a prompt with approximately the target number of tokens.
    Uses filler text to reach the desired token count.
    """
    # Approximate: 1 token ≈ 4 characters for English text
    target_chars = target_tokens * 4
    
    base_prompt = (
        "You are a helpful AI assistant. Please provide a detailed analysis of the following topic. "
        "Be thorough and comprehensive in your response. "
    )
    
    # Add filler content to reach target token count
    filler_words = ["technology", "innovation", "development", "implementation", "optimization", 
                    "architecture", "infrastructure", "performance", "scalability", "reliability"]
    
    filler_text = " ".join([random.choice(filler_words) for _ in range(target_tokens // 2)])
    
    prompt = f"{base_prompt}\n\nContext: {filler_text}\n\nQuestion: Based on the context above, what are the key considerations for building scalable cloud applications?"
    
    return prompt


def benchmark_service_tier(
    client,
    model_id: str,
    service_tier: str,
    prompt: str,
    max_tokens: int,
    invocations: int,
    sleep_time: float
) -> List[Dict]:
    """
    Benchmark a specific service tier with multiple invocations.
    
    Returns list of result dictionaries with timing and metadata.
    """
    results = []
    
    logger.info(f"Starting benchmark for service tier: {service_tier}")
    
    for i in tqdm(range(invocations), desc=f"Testing {service_tier} tier"):
        try:
            # Prepare the request
            messages = [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
            
            inference_config = {
                "maxTokens": max_tokens,
                "temperature": 0.7
            }
            
            # Add service_tier parameter
            request_params = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": inference_config
            }
            
            # Map our tier names to API values
            tier_mapping = {
                "priority": "priority",
                "standard": "default",
                "flex": "flex"
            }
            
            # Add serviceTier parameter (requires boto3 >= 1.42.0)
            request_params["serviceTier"] = {
                "type": tier_mapping[service_tier]
            }
            
            # Measure latency
            start_time = time.time()
            
            response = client.converse(**request_params)
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract response metadata
            output_text = response['output']['message']['content'][0]['text']
            usage = response.get('usage', {})
            input_tokens = usage.get('inputTokens', 0)
            output_tokens = usage.get('outputTokens', 0)
            
            # Get resolved service tier from response metadata
            stop_reason = response.get('stopReason', 'unknown')
            resolved_tier = response.get('serviceTier', {}).get('type', 'unknown')
            
            result = {
                'iteration': i + 1,
                'service_tier': service_tier,
                'resolved_tier': resolved_tier,
                'latency': round(latency, 3),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': True,
                'error': None
            }
            
            results.append(result)
            logger.debug(f"Iteration {i+1}: {latency:.3f}s")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error'].get('Message', '')
            logger.warning(f"ClientError on iteration {i+1}: {error_code} - {error_message}")
            
            result = {
                'iteration': i + 1,
                'service_tier': service_tier,
                'resolved_tier': None,
                'latency': None,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': False,
                'error': error_code
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Unexpected error on iteration {i+1}: {str(e)}")
            
            result = {
                'iteration': i + 1,
                'service_tier': service_tier,
                'resolved_tier': None,
                'latency': None,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': False,
                'error': str(e)
            }
            results.append(result)
        
        # Sleep between invocations (except last one)
        if i < invocations - 1:
            time.sleep(sleep_time)
    
    return results


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate statistics from benchmark results."""
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        return {
            'count': 0,
            'success_rate': 0,
            'mean_latency': None,
            'median_latency': None,
            'min_latency': None,
            'max_latency': None,
            'stdev_latency': None,
            'p95_latency': None,
            'p99_latency': None
        }
    
    latencies = [r['latency'] for r in successful_results]
    latencies_sorted = sorted(latencies)
    
    stats = {
        'count': len(successful_results),
        'success_rate': len(successful_results) / len(results) * 100,
        'mean_latency': round(mean(latencies), 3),
        'median_latency': round(median(latencies), 3),
        'min_latency': round(min(latencies), 3),
        'max_latency': round(max(latencies), 3),
        'stdev_latency': round(stdev(latencies), 3) if len(latencies) > 1 else 0,
        'p95_latency': round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 3),
        'p99_latency': round(latencies_sorted[int(len(latencies_sorted) * 0.99)], 3)
    }
    
    return stats


def save_results_to_csv(all_results: Dict[str, List[Dict]], output_file: str):
    """Save all results to CSV file."""
    logger.info(f"Saving results to {output_file}")
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['service_tier', 'iteration', 'resolved_tier', 'latency', 
                      'input_tokens', 'output_tokens', 'total_tokens', 
                      'timestamp', 'success', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for tier, results in all_results.items():
            for result in results:
                writer.writerow(result)
    
    logger.info(f"Results saved to {output_file}")


def print_summary(all_results: Dict[str, List[Dict]]):
    """Print summary statistics for all service tiers."""
    print("\n" + "="*80)
    print("SERVICE TIER BENCHMARK SUMMARY")
    print("="*80)
    
    for tier in ['priority', 'standard', 'flex']:
        if tier not in all_results:
            continue
            
        results = all_results[tier]
        stats = calculate_statistics(results)
        
        print(f"\n{tier.upper()} Tier:")
        print(f"  Total Requests:    {len(results)}")
        print(f"  Successful:        {stats['count']}")
        print(f"  Success Rate:      {stats['success_rate']:.1f}%")
        
        if stats['mean_latency']:
            print(f"  Mean Latency:      {stats['mean_latency']:.3f}s")
            print(f"  Median Latency:    {stats['median_latency']:.3f}s")
            print(f"  Min Latency:       {stats['min_latency']:.3f}s")
            print(f"  Max Latency:       {stats['max_latency']:.3f}s")
            print(f"  Std Dev:           {stats['stdev_latency']:.3f}s")
            print(f"  P95 Latency:       {stats['p95_latency']:.3f}s")
            print(f"  P99 Latency:       {stats['p99_latency']:.3f}s")
    
    # Compare tiers
    print("\n" + "-"*80)
    print("TIER COMPARISON:")
    print("-"*80)
    
    tier_stats = {}
    for tier in ['priority', 'standard', 'flex']:
        if tier in all_results:
            tier_stats[tier] = calculate_statistics(all_results[tier])
    
    if 'standard' in tier_stats and 'priority' in tier_stats:
        if tier_stats['standard']['mean_latency'] and tier_stats['priority']['mean_latency']:
            improvement = (tier_stats['standard']['mean_latency'] - tier_stats['priority']['mean_latency']) / tier_stats['standard']['mean_latency'] * 100
            print(f"Priority vs Standard: {improvement:+.1f}% latency change")
    
    if 'standard' in tier_stats and 'flex' in tier_stats:
        if tier_stats['standard']['mean_latency'] and tier_stats['flex']['mean_latency']:
            difference = (tier_stats['flex']['mean_latency'] - tier_stats['standard']['mean_latency']) / tier_stats['standard']['mean_latency'] * 100
            print(f"Flex vs Standard:     {difference:+.1f}% latency change")
    
    print("="*80 + "\n")


def create_visualization(all_results: Dict[str, List[Dict]], output_file: str):
    """Create box plot visualization comparing service tiers."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        logger.info(f"Creating visualization: {output_file}")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        
        for tier in ['priority', 'standard', 'flex']:
            if tier in all_results:
                successful_results = [r for r in all_results[tier] if r['success']]
                latencies = [r['latency'] for r in successful_results]
                if latencies:
                    data_to_plot.append(latencies)
                    stats = calculate_statistics(all_results[tier])
                    labels.append(f"{tier.capitalize()}\n(n={len(latencies)}, μ={stats['mean_latency']:.3f}s)")
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Latency (seconds)', fontsize=12)
        ax.set_title(f'Bedrock Service Tier Latency Comparison\n{args.model_id}', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_file}")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")


def main():
    logger.info("Starting Bedrock Service Tier Benchmark")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Region: {args.region}")
    logger.info(f"Target Input Tokens: {args.input_tokens}")
    logger.info(f"Max Output Tokens: {args.output_tokens}")
    logger.info(f"Invocations per tier: {args.invocations}")
    logger.info(f"Randomized order: {args.randomize}")
    
    # Create Bedrock client
    client = get_bedrock_client(args.region)
    
    # Generate prompt
    prompt = generate_prompt(args.input_tokens)
    logger.info(f"Generated prompt length: {len(prompt)} characters")
    
    # Test each service tier
    service_tiers = ['priority', 'standard', 'flex']
    all_results = {}
    
    if args.randomize:
        # Interleave requests across all tiers to eliminate time-based bias
        logger.info("Running in randomized mode - interleaving requests across tiers")
        
        # Create a list of (tier, iteration) tuples
        test_schedule = []
        for tier in service_tiers:
            for i in range(args.invocations):
                test_schedule.append((tier, i))
        
        # Shuffle the schedule
        random.shuffle(test_schedule)
        
        # Initialize results storage
        for tier in service_tiers:
            all_results[tier] = []
        
        # Execute tests in randomized order
        logger.info(f"Executing {len(test_schedule)} total requests in randomized order")
        for tier, iteration in tqdm(test_schedule, desc="Testing all tiers (randomized)"):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]
                
                inference_config = {
                    "maxTokens": args.output_tokens,
                    "temperature": 0.7
                }
                
                tier_mapping = {
                    "priority": "priority",
                    "standard": "default",
                    "flex": "flex"
                }
                
                request_params = {
                    "modelId": args.model_id,
                    "messages": messages,
                    "inferenceConfig": inference_config,
                    "serviceTier": {"type": tier_mapping[tier]}
                }
                
                start_time = time.time()
                response = client.converse(**request_params)
                end_time = time.time()
                latency = end_time - start_time
                
                output_text = response['output']['message']['content'][0]['text']
                usage = response.get('usage', {})
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
                resolved_tier = response.get('serviceTier', {}).get('type', 'unknown')
                
                result = {
                    'iteration': len(all_results[tier]) + 1,
                    'service_tier': tier,
                    'resolved_tier': resolved_tier,
                    'latency': round(latency, 3),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'success': True,
                    'error': None
                }
                
                all_results[tier].append(result)
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error'].get('Message', '')
                logger.warning(f"ClientError for {tier} tier: {error_code} - {error_message}")
                
                result = {
                    'iteration': len(all_results[tier]) + 1,
                    'service_tier': tier,
                    'resolved_tier': None,
                    'latency': None,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'success': False,
                    'error': error_code
                }
                all_results[tier].append(result)
                
            except Exception as e:
                logger.error(f"Unexpected error for {tier} tier: {str(e)}")
                
                result = {
                    'iteration': len(all_results[tier]) + 1,
                    'service_tier': tier,
                    'resolved_tier': None,
                    'latency': None,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'success': False,
                    'error': str(e)
                }
                all_results[tier].append(result)
            
            # Small sleep between requests
            time.sleep(args.sleep)
    else:
        # Original sequential testing
        for tier in service_tiers:
            results = benchmark_service_tier(
                client=client,
                model_id=args.model_id,
                service_tier=tier,
                prompt=prompt,
                max_tokens=args.output_tokens,
                invocations=args.invocations,
                sleep_time=args.sleep
            )
            all_results[tier] = results
    
    # Save results
    save_results_to_csv(all_results, args.output_csv)
    
    # Print summary
    print_summary(all_results)
    
    # Create visualization
    create_visualization(all_results, args.output_chart)
    
    logger.info("Benchmark completed successfully")


if __name__ == "__main__":
    main()
