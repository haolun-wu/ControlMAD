# Test case:
size 5, game_id 1-100, no token limit

**Remark** 
- All price are presented per 1M token.
- OpenAi and Gemini models use US dollars and others use Chinese RMB
- ¥6-10-15 (32k-128k) means there is a price layer based on usage, and the two ladders are 32k and 128k.


## o3-mini
- Accuracy: 49/50
- Confidence: 6.2/10

## o4-mini
- Accuracy: 43/50
- Confidence: 6.4/10

## gpt-5-nano-medium-effort
- Accuracy: 45/50
- Confidence: 8.4/10
- Input: $0.05
- Output: $0.4

## gpt-5-nano-low-effort
- Accuracy: 16/50
- Confidence: 4.6/10
- Input: $0.05
- Output: $0.4

## gpt-5-mini-low-effort
- Accuracy: 40/50
- Confidence: 6.3/10
- Test Date: 2025-09-30 10:56:45

## gpt-5-mini-medium-effort
- Accuracy: 41/50
- Confidence: 6.6/10
- Test Date: 2025-09-30 10:56:45

## gpt-4o-mini
- Accuracy: 1/50
- Confidence: 6/10
- Input: $0.15
- Output: $0.6

## gpt-4.1-mini
- Accuracy: 1/50
- Confidence: 6.9/10
- Input: $0.4
- Output: $1.6

## gemini-2.5-flash-lite
- Accuracy: 0/50
- Confidence: 6.8/10
- Input: $0.1
- Output: $0.4

## gemini-2.5-flash
- Accuracy: 48/50
- Confidence: 9.1/10
- Input: $0.3
- Output: $2.5

## qwq-32b
- Accuracy: 43/50
- Confidence: 6.9/10
- very slow
- Input: ¥2
- Output: ¥6

## qwen-turbo-latest (max output token = None, thinking budget = 4096)
- Accuracy: 22/50
- Confidence: 7.0/10
- Input: ¥0.3
- Output: ¥3

## qwen3-30b-a3b-thinking-2507 (max output token = None, thinking budget = 4096)
- Accuracy: 26/50
- Confidence: 6.3/10
- Input: ¥0.75
- Output: ¥7.5


## qwen-flash (max output token = None, thinking budget = 4096)
- Accuracy: 23/50
- Confidence: 6.0/10
- Input: ¥0.15-0.6-1.2 (32k-128k)
- Output: ¥.5-6-12 (32k-128k)


## qwen-plus (max output token = None, thinking budget = 4096)
- Accuracy: 26/50
- Confidence: 6.3/10
- Input: ¥0.8
- Output: ¥8

## doubao-seed-1-6-flash-250828 (max output token = 4096, thinking_type = enabled, not able to limit the thinking effort)
- Accuracy: 37/50
- Confidence: 7.3/10
- Input: ¥0.075-0.15-0.3 (32k-128k)
- Output: ¥0.75-1.5-3 (32k-128k)

## doubao-seed-1-6-250615 (max output token = 4096, thinking_type = enabled, not able to limit the thinking effort)
- Accuracy: 47/50
- Confidence: 8.1/10
- Input: ¥0.4-0.6-1.2 (32k-128k)
- Output: ¥4-8-12 (32k-128k)