I want to create a test to find out of Bedrock inference TPM limits is per absolute minute, or relative minute want to test bedrock API to answer this question.

Use Nova Pro model, for which I have 8,000 tokens per minute (TPM).I want to test the following scenarios:
1. Consume all TPM at xx:35 seconds (expect 200OK responses), then make sure your get 429 for the next request that is 1000 tokens long, before the end of the minute.
2. Try again 35 seconds later, at yy:05 seconds. Then see if you get a 200OK or a 429 response code for this request.
3. Try another 1000 request at yy:50, this request should be 200OK becuase both the absolute minute passed and more than 60 seconds passed.
4. Summarize the requests, tokens used per request, the time they were made and the response you got. Then summarize the conclusion.

Use python virtual environment.
Create an IPython notebook to test for it. Put relevant helping code in utils.py
Use Bedrock Converse API. Make sure to turn off Boto3 automatic retries to avoid interfearing with throttling detection.
Pack the requests with lots of '0' to reach the needed input tokens. 
Serach the internet using Tavili as needed to find relevant APIs.
Read https://github.com/gilinachum/bedrock-latency/blob/main/stress.test.bedrock.py and https://github.com/gilinachum/bedrock-latency/blob/main/bedrock-latency-benchmark.ipynb to be inspired.