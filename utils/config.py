from .project_types import base_config, test_setup


game_config = base_config(
    name_pool=["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tina", "Uma", "Violet", "Wendy", "Xavier", "Yara", "Zane"],
    role_map={
        1: "knight",
        2: "knave",
        3: "spy"
    },
    tf_map={
        1: "telling the truth",
        2: "lying"
    },
    eo_map={
        1: "odd",
        2: "even"
    },
    number_map={
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten"
    },
    game_size = 5,
    num_spy = 1,
    num_hint = 0
)

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
    verbosity = "high",
    reasoning_effort = "low",
    reasoning_summary = "None",
    return_full_response = False,
    truncation = 50  # Use first 50 cases
)



