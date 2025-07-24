import boto3
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class BedrockTPMTester:
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize Bedrock client for TPM testing."""
        # Disable automatic retries to get immediate throttling responses
        from botocore.config import Config
        config = Config(
            retries={'max_attempts': 1, 'mode': 'standard'},
            read_timeout=30,
            connect_timeout=10
        )
        self.client = boto3.client('bedrock-runtime', region_name=region_name, config=config)
        self.model_id = 'amazon.nova-pro-v1:0'
        self.test_results = []
        
    def create_large_prompt(self, target_tokens: int) -> str:
        """Create a prompt with approximately target_tokens input tokens."""
        # Rough estimate: 1 token ≈ 4 characters for English text
        # Using '0' characters to pad the prompt
        base_prompt = "Please analyze the following data: "
        padding_needed = max(0, (target_tokens * 4) - len(base_prompt))
        padding = '0' * padding_needed
        return base_prompt + padding
    
    def make_bedrock_request(self, input_tokens: int) -> Tuple[int, Optional[str], float]:
        """
        Make a Bedrock Converse API request.
        
        Returns:
            Tuple of (status_code, error_message, response_time)
        """
        prompt = self.create_large_prompt(input_tokens)
        
        request_payload = {
            "modelId": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 10,  # Minimal output to focus on input token testing
                "temperature": 0.1
            }
        }
        
        start_time = time.time()
        try:
            response = self.client.converse(**request_payload)
            end_time = time.time()
            return 200, None, end_time - start_time
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            # Check if it's a throttling error (429)
            if "ThrottlingException" in error_msg or "too many requests" in error_msg.lower():
                return 429, error_msg, end_time - start_time
            else:
                return 500, error_msg, end_time - start_time
    
    def wait_until_second(self, target_second: int):
        """Wait until the specified second of the current minute."""
        while True:
            current_time = datetime.now()
            if current_time.second == target_second:
                break
            time.sleep(0.1)  # Check every 100ms
    
    def consume_tpm_quota(self, total_tokens: int, chunk_size: int = 1000) -> List[Dict]:
        """
        Consume TPM quota by making multiple requests.
        
        Args:
            total_tokens: Total tokens to consume
            chunk_size: Tokens per request
            
        Returns:
            List of request results
        """
        results = []
        tokens_consumed = 0
        
        while tokens_consumed < total_tokens:
            remaining_tokens = total_tokens - tokens_consumed
            request_tokens = min(chunk_size, remaining_tokens)
            
            timestamp = datetime.now()
            status_code, error, response_time = self.make_bedrock_request(request_tokens)
            
            result = {
                'timestamp': timestamp.isoformat(),
                'second': timestamp.second,
                'tokens_requested': request_tokens,
                'status_code': status_code,
                'error': error,
                'response_time': response_time
            }
            
            results.append(result)
            tokens_consumed += request_tokens
            
            # If we get throttled, stop consuming
            if status_code == 429:
                break
                
            # Small delay between requests to avoid overwhelming
            time.sleep(0.1)
        
        return results
    
    def log_result(self, test_name: str, timestamp: datetime, tokens: int, 
                   status_code: int, error: Optional[str] = None):
        """Log a test result."""
        result = {
            'test_name': test_name,
            'timestamp': timestamp.isoformat(),
            'minute': timestamp.minute,
            'second': timestamp.second,
            'tokens': tokens,
            'status_code': status_code,
            'error': error
        }
        self.test_results.append(result)
        print(f"[{timestamp.strftime('%H:%M:%S')}] {test_name}: {tokens} tokens -> {status_code}")
        if error:
            print(f"  Error: {error}")
    
    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        for result in self.test_results:
            timestamp = datetime.fromisoformat(result['timestamp'])
            print(f"Test: {result['test_name']}")
            print(f"  Time: {timestamp.strftime('%H:%M:%S')} (minute {result['minute']}, second {result['second']})")
            print(f"  Tokens: {result['tokens']}")
            print(f"  Status: {result['status_code']}")
            if result['error']:
                print(f"  Error: {result['error']}")
            print()
        
        # Analysis
        print("ANALYSIS:")
        print("-" * 40)
        
        # Group by test phases
        consume_requests = [r for r in self.test_results if 'consume' in r['test_name'].lower()]
        test_requests = [r for r in self.test_results if 'test' in r['test_name'].lower()]
        
        if consume_requests:
            total_consumed = sum(r['tokens'] for r in consume_requests if r['status_code'] == 200)
            print(f"Total tokens consumed in quota exhaustion: {total_consumed}")
        
        if test_requests:
            print("Test request results:")
            for req in test_requests:
                status_text = "SUCCESS" if req['status_code'] == 200 else "THROTTLED"
                print(f"  - {req['test_name']}: {status_text}")
        
        # Determine if limits are absolute or relative
        if len(test_requests) >= 2:
            first_test = test_requests[0]
            second_test = test_requests[1]
            
            print(f"\nCONCLUSION:")
            print("-" * 40)
            
            if first_test['status_code'] == 429 and second_test['status_code'] == 200:
                print("✓ TPM limits appear to be based on ABSOLUTE MINUTES")
                print("  The quota reset at the start of a new clock minute")
            elif first_test['status_code'] == 429 and second_test['status_code'] == 429:
                print("✓ TPM limits appear to be based on RELATIVE MINUTES") 
                print("  The quota requires 60 seconds to pass from first usage")
            else:
                print("? Results are inconclusive - may need to repeat test")