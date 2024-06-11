import json
import os
import threading
import time
import boto3, botocore
from botocore.exceptions import ClientError

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

''' 
Get a boto3 bedrock runtime client for invoking requests
region - the AWS region to use
model_id_for_warm_up - the model id to warm up the client against, use None for no warmup
Note: Removing auto retries to ensure we're measuring a single transcation (e.g., in case of throttling).
'''
def _get_bedrock_client(region, model_id_for_warm_up = None):
    client = boto3.client(service_name='bedrock-runtime',
                          region_name=region,
                          config = botocore.config.Config(retries=dict(max_attempts=0))) 
    return client

def _send_request(client, req):
    response = client.invoke_model(**req)
    return response


def construct_body(messages, system_prompt, max_tokens, temperature) -> str:
    body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,                
            "temperature": temperature,
        })
    return body


bedrock = _get_bedrock_client('us-east-1', None)

# Set up variables to track results
total_requests = 0
successful_requests = 0
failed_requests = 0
empty_responses = 0
throttled_requests = 0


# Define the function to be executed by each thread
def invoke_agent(thread_id):
    global total_requests, successful_requests, failed_requests, empty_responses, throttled_requests
    try:
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        #model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        user_message =  {"role": "user", "content": "Explain how distributed training works in 1000 words"}
        messages = [user_message]
        system_prompt = "You are a nice person."
        max_tokens = 500
        temperature = 1
        body = construct_body(messages, system_prompt, max_tokens, temperature)
        total_requests += 1
        
        response = bedrock.invoke_model(body=body, modelId=model_id)
        
        logger.log(logging.DEBUG, f'threadId={thread_id}, response={response}')
        response_body = json.loads(response.get('body').read())
        stop_reason = response_body['stop_reason']
        first_byte = time.time()
        last_byte = first_byte
        logger.log(logging.DEBUG, f"threadId={thread_id}, body={response_body}")
        statusCode = response['ResponseMetadata']['HTTPStatusCode']
        if statusCode == 200:
            successful_requests += 1
            if len(response_body) == 0:
                empty_responses += 1
        else:
            failed_requests += 1
            logger.log(logging.WARNING, f"threadId={thread_id}, Error statusCode={statusCode}")
    except ClientError as err:
        if 'Thrott' in err.response['Error']['Code']:
            failed_requests += 1
            throttled_requests += 1
            logger.log(logging.WARNING, f'thread_id={thread_id}, Got ThrottlingException')
        raise err
    except Exception as e:
        print(f"Error: {e}")


# Create and start the threads
threads = []
start_time = time.time()
for i in range(1000):
    thread = threading.Thread(target=invoke_agent, args=[str(i)])
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

end_time = time.time()

# Print the results
print(f"Total requests: {total_requests}")
print(f"Successful requests: {successful_requests}")
print(f"Failed requests: {failed_requests}")
print(f"Empty responses: {empty_responses}")
print(f"Throttled responses: {throttled_requests}")
print(f"Total time taken: {end_time - start_time} seconds")