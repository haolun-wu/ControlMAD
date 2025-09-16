# Test case:
size 5, game_id 1-100, no token limit

**Remark** 
- All price are presented per 1M token.
- OpenAi and Gemini models use US dollars and others use Chinese RMB
- ¥6-10-15 (32k-128k) means there is a price layer based on usage, and the two ladders are 32k and 128k.

## gpt-5-nano, medium thinking effort
- Accuracy: 89/100
- Confidence: 89.1
- Input: $0.05
- Output: $0.4

## gpt-5-nano, low thinking effort
- Accuracy: 48/100
- Confidence: 62.0
- Input: $0.05
- Output: $0.4

## gpt-4o-mini
- Accuracy: 0/100
- Confidence: 96.3
- Input: $0.15
- Output: $0.6

## gpt-4.1-mini
- Accuracy: 2/100
- Confidence: 98.1
- Input: $0.4
- Output: $1.6

## gemini-2.5-flash-lite
- Accuracy: 6/100
- Confidence: 97.1
- Input: $0.1
- Output: $0.4

## gemini-2.5-flash
- Accuracy: 94/100
- Confidence: 98
- Input: $0.3
- Output: $2.5

## qwq-32b
- Accuracy: 85/100
- Confidence: 98.5
- very slow
- Input: ¥2
- Output: ¥6

## qwen-turbo-latest (max output token = 4096, thinking budget = 4096)
- Accuracy: 41/100
- Confidence: 98.0
- Input: ¥0.3
- Output: ¥3

## qwen3-30b-a3b-thinking-2507 (max output token = 4096, thinking budget = 4096)
- Accuracy: 51/100
- Confidence: 97.8
- Input: ¥0.75
- Output: ¥7.5


## qwen-flash (max output token = 4096, thinking budget = 4096)
- Accuracy: 46/100
- Confidence: 97.8
- Input: ¥0.15-0.6-1.2 (32k-128k)
- Output: ¥.5-6-12 (32k-128k)


## qwen-plus (max output token = 4096, thinking budget = 4096)
- Accuracy: 46/100
- Confidence: 94.4
- Input: ¥0.8
- Output: ¥8

## doubao-seed-1-6-flash-250828 (max output token = 4096, thinking_type = enabled, not able to limit the thinking effort)
- Accuracy: 76/100
- Confidence: 99
- Input: ¥0.075-0.15-0.3 (32k-128k)
- Output: ¥0.75-1.5-3 (32k-128k)

## doubao-seed-1-6-250615 (max output token = 4096, thinking_type = enabled, not able to limit the thinking effort)
- Accuracy: 89/100
- Confidence: 100
- Input: ¥0.4-0.6-1.2 (32k-128k)
- Output: ¥4-8-12 (32k-128k)