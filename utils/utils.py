import anthropic, boto3, botocore, os, random, pprint
import matplotlib.pyplot as plt

anthropic_client = anthropic.Anthropic() # used to count tokens only

# This internal method will include arbitrary long input that is designed to generate an extremely long model output
def _get_prompt_template(num_input_tokens):
    tokens = 'Human:'
    tokens += 'Ignore X' + '<X>'
    for i in range(num_input_tokens-1):
        tokens += random.choice(['hello', 'world', 'foo', 'bar']) + ' '
    tokens += '</X>'
    tokens += "print numbers 1 to 9999 as words. don't omit for brevity"
    tokens += '\n\nAssistant:one two'  # model will continue with " three four five..."
    return tokens

''' 
This method creates a prompt of input length `expected_num_tokens` which instructs the LLM to generate extremely long model resopnse
'''
def create_prompt(expected_num_tokens):
    num_tokens_in_prompt_template = anthropic_client.count_tokens(_get_prompt_template(0))
    additional_tokens_needed = max(expected_num_tokens - num_tokens_in_prompt_template,0)
    
    prompt_template = _get_prompt_template(additional_tokens_needed)
    
    actual_num_tokens = anthropic_client.count_tokens(prompt_template)
    #print(f'expected_num_tokens={expected_num_tokens}, actual_tokens={actual_num_tokens}')
    assert expected_num_tokens==actual_num_tokens, f'Failed to generate prompt at required length: expected_num_tokens{expected_num_tokens} != actual_num_tokens={actual_num_tokens}'
    
    return prompt_template


import time, json
from botocore.exceptions import ClientError
sleep_on_throttling_sec = 5

def benchmark(bedrock, prompt, max_tokens_to_sample, stream=True, temprature=0):
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'
    
    body = json.dumps({
    "prompt": prompt,
    "max_tokens_to_sample": max_tokens_to_sample,
    "temperature": 0,
})
    while True:
        try:
            start = time.time()

            if stream:
                response = bedrock.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept, contentType=contentType)
            else:
                response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
            #print(response)
            
            first_byte = None
            if stream:
                event_stream = response.get('body')
                for event in event_stream:
                    chunk = event.get('chunk')
                    if chunk:
                        if not first_byte:
                            first_byte = time.time() # update the time to first byte
                        #print(f'chunk:\n {json.loads(chunk.get('bytes').decode())}')
                # end of stream - check stop_reson in last chunk
                stop_reason = json.loads(chunk.get('bytes').decode())['stop_reason']    
                last_byte = time.time()
            else:
                #no streaming flow
                first_byte = time.time()
                last_byte = first_byte
                response_body = json.loads(response.get('body').read())
                stop_reason = response_body['stop_reason']

            
            # verify we got all of the intended output tokens by verifying stop_reason
            assert stop_reason == 'max_tokens', f"stop_reason is {stop_reason} instead of 'max_tokens', this means the model generated less tokens than required."

            duration_to_first_byte = first_byte - start
            duration_to_last_byte = last_byte - start
        except ClientError as err:
            if 'Thrott' in err.response['Error']['Code']:
                print(f'Got ThrottlingException. Sleeping {sleep_on_throttling_sec} sec and retrying.')
                time.sleep(sleep_on_throttling_sec)
                continue
            raise err
        break
    return duration_to_first_byte, duration_to_last_byte


def execute_benchmark(scenarios,scenario_config, early_break):
    pp = pprint.PrettyPrinter(indent=2)
    for scenario in scenarios:
        for i in range(scenario_config["invocations_per_scenario"]): # increase to sample each use case more than once to discover jitter
            try:
                prompt = create_prompt(scenario['in_tokens'])
                client = get_cached_client(scenario['region'])
                time_to_first_token,time_to_last_token = benchmark(client, prompt, scenario['out_tokens'], stream=scenario['stream'])

                if 'durations' not in scenario: scenario['durations'] = list()
                duration = {
                    'time-to-first-token':  time_to_first_token,
                    'time-to-last-token':  time_to_last_token,
                }
                scenario['durations'].append(duration)

                print(f"Scenario: [{scenario['name']}, " + 
                      f'Duration: {pp.pformat((duration))}')

                post_iteration(is_last_invocation = i == scenario_config["invocations_per_scenario"] - 1, scenario_config=scenario_config)
            except Exception as e:
                print(e)
                print(f"Error while processing scenario: {scenario['name']}.")
            if early_break:
                break
    show_results(scenarios)

import time
''' 
Get a boto3 bedrock runtime client for invoking requests
region - the AWS region to use
Note: Removing auto retries to ensure we're measuring a single transcation (e.g., in case of throttling).
'''
def _get_bedrock_client(region, warmup=True):
    client = boto3.client( service_name='bedrock-runtime',
                          region_name=region,
                          config=botocore.config.Config(retries=dict(max_attempts=0))) 
    if warmup:
        benchmark(client, create_prompt(50), 1)
    return client

'''
Get a possible cache client per AWS region 
'''
client_per_region={}
def get_cached_client(region):
    if client_per_region.get(region) is None:
        client_per_region[region] = _get_bedrock_client(region)
    return client_per_region[region]


def post_iteration(is_last_invocation, scenario_config):
    if scenario_config["sleep_between_invocations"] > 0 and not is_last_invocation:
        print(f'Sleeping for {scenario_config["sleep_between_invocations"]} seconds.')
        time.sleep(scenario_config["sleep_between_invocations"])
        
def show_results(scenarios):

    fig, ax = plt.subplots()

    metric = 'time-to-first-token'
    #metric = 'time-to-last-token'

    for scenario in scenarios:
      durations = [d[metric] for d in scenario['durations']]

      ax.boxplot(durations, positions=[scenarios.index(scenario)])

      ax.set_xticks(range(len(scenarios)))
      ax.set_xticklabels([s['name'] for s in scenarios])

      ax.set_ylabel(f'{metric} (sec)')

    fig.tight_layout()
    plt.show()