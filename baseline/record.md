# Test case:
size 5, game_id 1-100, no token limit

**Remark** 
- All price are presented per 1M token.
- OpenAi and Gemini models use US dollars and others use Chinese RMB
- ¥6-10-15 (32k-128k) means there is a price layer based on usage, and the two ladders are 32k and 128k.

## gpt-5-nano, medium thinking effort
- 85/100
- Input: $0.05
- Output: $0.4

## gpt-5-nano, low thinking effort
- 46/100
- Input: $0.05
- Output: $0.4

## gpt-4o-mini
- 0/100
- Input: $0.15
- Output: $0.6

## gpt-4.1-mini
- 1/100
- Input: $0.4
- Output: $1.6

## gemini-2.5-flash-lite
- 3/100
- Input: $0.1
- Output: $0.4

## gemini-2.5-flash
- 89/100 (10 requests with model overload. only 1 hard error)
- Input: $0.3
- Output: $2.5

## qwq-32b
- 78/100
- very slow
- Input: ¥2
- Output: ¥6

## qwen-turbo-latest (max output token = 4096, thinking budget = 2048)
- 28/100
- balanced response speed.
- Input: ¥0.3
- Output: ¥3

## qwen3-30b-a3b-thinking-2507 (max output token = 4096, thinking budget = 2048)
- 28/100
- Input: ¥0.75
- Output: ¥7.5


## qwen-flash (max output token = 4096, thinking budget = 2048)
- 24/100
- Input: ¥0.15-0.6-1.2 (32k-128k)
- Output: ¥.5-6-12 (32k-128k)


## qwen-plus (max output token = 4096, thinking budget = 4096)
- 34/100
- Input: ¥0.8
- Output: ¥8


## qwen3-max-preview (max output token = 4096, thinking budget = 4096)
- 23/100
- Input: ¥6-10-15 (32k-128k)
- Output: ¥24-40-60 (32k-128k)

## doubao-seed-1-6-flash-250828 (max output token = 4096, thinking_type = enabled, no able to limit the thinking effort)
- 80/100
- Input: ¥0.075-0.15-0.3 (32k-128k)
- Output: ¥0.75-1.5-3 (32k-128k)

## doubao-seed-1-6-250615 (max output token = 4096, thinking_type = enabled, no able to limit the thinking effort)
- 94/100
- Input: ¥0.4-0.6-1.2 (32k-128k)
- Output: ¥4-8-12 (32k-128k)