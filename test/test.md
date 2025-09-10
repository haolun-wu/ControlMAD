# Record of test configurations

## gpt-5-nano with low thinking efforts: 54/100 for size 5.
```python
test_config = test_setup(
    game_size = 5,
    provider = "openai",
    model = "gpt-5-nano",
    pass_at = 1,
    num_worker = 10,
    show_thinking = False,
    groundtruth_path = "./groundtruth",
    output_path = "./test",
    enable_thinking = False,
    stream = False,
    verbosity = "low",
    reasoning_effort = "low",
    reasoning_summary = "None",
    return_full_response = False,
    truncation = -1
)
```
