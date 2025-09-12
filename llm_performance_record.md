# Test case:
size 5, game_id 1-100, no token limit

## gpt-5-nano, medium thinking effort
- 85/100

## gpt-5-nano, low thinking effort
- 46/100

## gpt-4o-mini
- 0/100

## gpt-4.1-mini
- 1/100

## gemini-2.5-flash-lite
- 3/100

## gemini-2.5-flash
- 89/100 (10 requests with model overload. only 1 hard error)

## qwq-32b
- 78/100
- very slow

## qwen-turbo-latest (max token = 4096, thinking budget = 2048)
- 28/100
- balanced response speed.

## qwen3-30b-a3b-thinking-2507 (max token = 4096, thinking budget = 2048)
- 28/100

## qwen-flash (max token = 4096, thinking budget = 2048)
- 24/100

## qwen-plus (max token = 4096, thinking budget = 4096)
- 34/100

## qwen3-max-preview (max token = 4096, thinking budget = 4096)
- 23/100
