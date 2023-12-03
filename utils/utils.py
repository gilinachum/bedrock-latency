import anthropic, boto3, botocore, os, random, pprint
from openai import OpenAI
import time, json
from copy import deepcopy
from botocore.exceptions import ClientError
from utils.key import OPENAI_API_KEY

SLEEP_ON_THROTTLING_SEC = 5

# This internal method will include arbitrary long input that is designed to generate an extremely long model output
def _get_prompt_template(num_input_tokens, modelId):
    # Determine the service based on modelId prefix
    is_openai_model = modelId.startswith('gpt-')
    tokens = ''
    if not is_openai_model:
        tokens += 'Human:'
    tokens += 'Ignore X' + '<X>'
    for i in range(num_input_tokens-1):
        tokens += random.choice(['hello', 'world', 'foo', 'bar']) + ' '
    tokens += '</X>'
    tokens += "Task: Print numbers from 1 to 9999 as words. Continue listing the numbers in word format until the space runs out. \n"
    tokens += "One \n"
    tokens += "two \n"
    tokens += "three\n"
    tokens += "..."
    if not is_openai_model:
        tokens += '\n\nAssistant:one two'  # model will continue with "four five..."
    return tokens

def _construct_body(modelId, prompt, max_tokens_to_sample, temperature):
    """
    Private method to construct the body for model invocation based on the model type.
    """
    # OpenAI Models
    if modelId.startswith('gpt-'):
        body = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": modelId,
            "max_tokens": max_tokens_to_sample,
            "temperature": temperature
        })
    # Anthropic Models
    elif modelId.startswith('anthropic.'):
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": temperature,
            "top_p": 0.9  # Example value, adjust as needed
        })
    # A2I Models
    elif modelId.startswith('ai21.'):
        body = json.dumps({
            "prompt": prompt,
            "maxTokens": max_tokens_to_sample,
            "temperature": temperature,
            "topP": 0.5  # Example value, adjust as needed
        })
    else:
        # Default body format if modelId does not match any of the above
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": temperature
        })

    return body

''' 
This method creates a prompt of input length `expected_num_tokens` which instructs the LLM to generate extremely long model resopnse
'''
anthropic_client = anthropic.Anthropic() # used to count tokens only
def create_prompt(expected_num_tokens, modelId):
    # print(f"create_prompt called with modelId: {modelId}")
    num_tokens_in_prompt_template = anthropic_client.count_tokens(_get_prompt_template(0, modelId))
    additional_tokens_needed = max(expected_num_tokens - num_tokens_in_prompt_template,0)
    
    prompt_template = _get_prompt_template(additional_tokens_needed, modelId)
    
    actual_num_tokens = anthropic_client.count_tokens(prompt_template)
    #print(f'expected_num_tokens={expected_num_tokens}, actual_tokens={actual_num_tokens}')
    assert expected_num_tokens==actual_num_tokens, f'Failed to generate prompt at required length: expected_num_tokens{expected_num_tokens} != actual_num_tokens={actual_num_tokens}'
    
    return prompt_template
'''
This method will invoke the model, possibly in streaming mode,
In case of throttling error, the method will retry. Throttling and related sleep time isn't measured.
The method ensures the response includes `max_tokens_to_sample` by verify the stop_reason is `max_tokens`

client - the bedrock runtime client to invoke the model
modelId - the model id to invoke
prompt - the prompt to send to the model
max_tokens_to_sample - the number of tokens to sample from the model's response
stream - whether to invoke the model in streaming mode
temperature - the temperature to use for sampling the model's response

Returns the time to first byte, last byte, and invocation time as iso8601 (seconds)
'''
def benchmark(client, modelId, prompt, max_tokens_to_sample, stream=True, temperature=0):
    import time
    from datetime import datetime
    import pytz
    accept = 'application/json'
    contentType = 'application/json'
    
    body = _construct_body(modelId, prompt, max_tokens_to_sample, temperature)
    
    # Determine the service based on modelId prefix
    is_openai_model = modelId.startswith('gpt-')
    
    while True:
        try:
            start = time.time()
            
            if is_openai_model:
                
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=modelId,
                    max_tokens=max_tokens_to_sample,
                    stream=stream
                )
                if not stream:
                    stop_reason = response.choices[0].finish_reason
                    last_byte = time.time()
                    first_byte = start   
            elif not is_openai_model and stream:
                response = client.invoke_model_with_response_stream(
                    body=body, modelId=modelId, accept=accept, contentType=contentType)
            elif not is_openai_model and not stream:
                response = client.invoke_model(
                    body=body, modelId=modelId, accept=accept, contentType=contentType)
                # print(f'response is {response}')
                response_body = json.loads(response.get('body').read())
                # print(f"response body is {response_body}")
                stop_reason = response_body['completions'][0]['finishReason']['reason']
            
            first_byte = None
            dt = datetime.fromtimestamp(time.time(), tz=pytz.utc)
            invocation_timestamp_iso = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            if stream and is_openai_model:
                first_byte = time.time()
                for chunk in response:
                    if chunk.choices[0].finish_reason is not None:
                        stop_reason = chunk.choices[0].finish_reason
                last_byte = time.time()                
            elif not is_openai_model and stream:
                event_stream = response.get('body')
                for event in event_stream:
                    chunk = event.get('chunk')
                    if chunk:
                        if not first_byte:
                            first_byte = time.time() # update the time to first byte
                # end of stream - check stop_reson in last chunk
                stop_reason = json.loads(chunk.get('bytes').decode())['stop_reason']    
                last_byte = time.time()
            elif is_openai_model and not stream:
                first_byte = time.time()
                last_byte = first_byte
                response_body = response.choices[0].message.content
                stop_reason = response.choices[0].finish_reason
            else:
                first_byte = time.time()
                last_byte = first_byte
                if modelId.startswith('ai21'):
                    stop_reason = response_body['completions'][0]['finishReason']['reason']
                else:
                    stop_reason = response_body['stop_reason']
            
            # verify we got all of the intended output tokens by verifying stop_reason
            valid_stop_reasons = ['max_tokens', 'length']
            assert stop_reason in valid_stop_reasons, f"stop_reason is {stop_reason} instead of 'max_tokens' or 'length', this means the model generated less tokens than required or stopped for a different reason."
            duration_to_first_byte = round(first_byte - start, 2)
            duration_to_last_byte = round(last_byte - start, 2)
        except ClientError as err:
            if 'Thrott' in err.response['Error']['Code']:
                print(f'Got ThrottlingException. Sleeping {SLEEP_ON_THROTTLING_SEC} sec and retrying.')
                time.sleep(SLEEP_ON_THROTTLING_SEC)
                continue
            raise err
        break
    return duration_to_first_byte, duration_to_last_byte, invocation_timestamp_iso

'''
This method will benchmark the given scenarios.
scenarios - a list of scenarios to benchmark
scenario_config - a dictionary of configuration parameters
early_break - if true, will break after a single scenario, useful for debugging.
Returns a list of benchmarked scenarios with a list of invocation (latency and timestamp)
'''
def execute_benchmark(scenarios, scenario_config, early_break = False):
    scenarios = scenarios.copy()
    pp = pprint.PrettyPrinter(indent=2)
    scenarios_list = []
    for scenario in scenarios:
        scenario_copy = deepcopy(scenario)
        for i in range(scenario_config["invocations_per_scenario"]): # increase to sample each use case more than once to discover jitter
            scenario_label = f"{scenario_copy['model_id']} \n in={scenario_copy['in_tokens']}, out={scenario_copy['out_tokens']}"
            try:
                modelId = scenario_copy['model_id']
                prompt = create_prompt(scenario_copy['in_tokens'], modelId)
                
                # Determine the service based on modelId prefix
                is_openai_model = modelId.startswith('gpt-')
                
                if is_openai_model:
                    client = OpenAI(
                        api_key=OPENAI_API_KEY
                    )
                else:
                    client = get_cached_client(scenario_copy['region'], scenario_copy['model_id'])
                time_to_first_token, time_to_last_token, timestamp = benchmark(client, modelId, prompt, scenario_copy['out_tokens'], stream=scenario_copy['stream'])

                if 'invocations' not in scenario: scenario_copy['invocations'] = list()
                invocation = {
                    'time-to-first-token':  time_to_first_token,
                    'time-to-last-token':  time_to_last_token,
                    'timestamp_iso' : timestamp
                }
                scenario_copy['invocations'].append(invocation)

                scenario_label = f"{scenario_copy['model_id']} \n in={scenario_copy['in_tokens']}, out={scenario_copy['out_tokens']}"
                print(f"Scenario: [{scenario_label}, " + 
                      f'invocation: {pp.pformat((invocation))}')

                post_iteration(is_last_invocation = i == scenario_config["invocations_per_scenario"] - 1, scenario_config=scenario_config)
            except Exception as e:
                print(f"Error is: {e}")
                print(f"Error while processing scenario: {scenario_label}.")
            if early_break:
                break
        scenarios_list.append(scenario_copy)
    print(f'scenarios at the end of execute benchmark is: {scenario_copy}')
    return scenarios_list


''' 
Get a boto3 bedrock runtime client for invoking requests
region - the AWS region to use
model_id_for_warm_up - the model id to warm up the client against, use None for no warmup
Note: Removing auto retries to ensure we're measuring a single transcation (e.g., in case of throttling).
'''
def _get_bedrock_client(region, model_id_for_warm_up = None):
    client = boto3.client( service_name='bedrock-runtime',
                          region_name=region,
                          config=botocore.config.Config(retries=dict(max_attempts=0))) 
    # if model_id_for_warm_up:
    #     benchmark(client, model_id_for_warm_up, create_prompt(50, model_id_for_warm_up), 1)
    return client

'''
Get a possible cache client per AWS region 
region - the AWS region to use
model_id_for_warm_up - the model id to warm up the client against, use None for no warmup
'''
client_per_region={}
def get_cached_client(region, model_id_for_warm_up = None):
    print(f"get_cached_client called with region: {region}, model_id_for_warm_up: {model_id_for_warm_up}")
    if client_per_region.get(region) is None:
        client_per_region[region] = _get_bedrock_client(region, model_id_for_warm_up)
    return client_per_region[region]


def post_iteration(is_last_invocation, scenario_config):
    if scenario_config["sleep_between_invocations"] > 0 and not is_last_invocation:
        print(f'Sleeping for {scenario_config["sleep_between_invocations"]} seconds.')
        time.sleep(scenario_config["sleep_between_invocations"])
        

'''
This method draws a boxplot graph of each scenario.
scenarios - list of scenarios
title - title of the graph
metric - metric to be plotted (time-to-first-token or time-to-last-token)
'''
def graph_scenarios_boxplot(scenarios, title, figsize=(10, 6)):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    xlables = []
    combined_times = []
    
    # Angle labels if covering many scenarios, to avoid collisions
    if len(scenarios) > 4:
        x_ticks_angle=45
    else:
        x_ticks_angle=0

    for scenario in scenarios:
        if 'invocations' in scenario:
            # Combine the times to first and last token into one list for each scenario
            times_combined = [d['time-to-first-token'] for d in scenario['invocations']] + \
                             [d['time-to-last-token'] for d in scenario['invocations']]
            combined_times.append(times_combined)

            scenario_label = f"{scenario['model_id']} \n in={scenario['in_tokens']}, out={scenario['out_tokens']}"
            xlables.append(scenario_label)
        else:
            print(f"No 'invocations' key found in scenario: {scenario}")

    # Plotting combined times for each scenario
    ax.boxplot(combined_times)

    ax.set_title(title)
    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels(xlables, rotation=45, ha="right")
    ax.set_ylabel('Time (sec)')

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plt.show()
