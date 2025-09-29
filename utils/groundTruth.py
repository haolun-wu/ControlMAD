from .project_types import base_config, ground_truth
from .config import game_config, test_config
from .validator import Validator
from .generator import Generator
from .utility import cstcloud
import json
import time

class GroundTruth:
    def __init__(self, config: base_config):
        self.config = config
        self.validator = Validator(config)
        self.generator = Generator(config)
        self.cst = cstcloud("secret.json")

    def generate_ground_truth_test(self, game_id):
        """
        Generate a single game and test the basic functionality of the project.
        """
        schemas, statements = self.generator.generate_single_game_abstract()
        solutions = self.validator.validate_single_game(schemas, statements)
        #print("number of solutions: ", len(solutions))
        if len(solutions) == 1:
            print(f"Game {game_id} has a unique solution. Testing LLM...")
            players = self.generator.generate_single_game_text(schemas, statements)
            game_text = ""
            for (index, single_player) in enumerate(players):
                game_text += "---\n"
                if index != len(players) - 1:
                    game_text += f"Player name: {single_player.name}\n"
                    game_text += f"Player statement: {single_player.statement}\n"
                else:
                    game_text += f"Message from the game manager: {single_player.statement}\n"
            for attempt in range(3):
                try:
                    # llm_solution = self.cst.chat_completion(game_text, kks_system_prompt.replace("{num_player}", str(self.config.game_size)), model = "gpt-oss-120b")  # Commented out - not used
                    break  # Success, exit the loop
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        llm_solution = "API request failed for three attempts."  # Re-raise the exception
                    print(f"Game {game_id} Attempt {attempt + 1} failed: {e}")
                    time.sleep(1)  # Wait 1 second
            
            parsed_solution = ""
            solution_map = {}
            try:
                solution_map = json.loads(llm_solution)
                for key, value in solution_map.items():
                    if key != "explanation":
                        parsed_solution += f"{key} is a {value}.\n"
                    else:
                        parsed_solution += f"Explanation:\n {value}\n"
            except json.JSONDecodeError:
                parsed_solution = llm_solution

            with open("log.txt", "a") as f:
                f.write("="*50)
                f.write(f"\nGame {game_id} setup: {self.config.game_size} players, {self.config.num_spy} spies.")
                f.write(f"\nSchemas:  {schemas}")
                f.write(f"\nStatements: {statements}\n")
                f.write(game_text)
                f.write("-"*50)
                f.write("\nSolution of the game: ")
                for i in range(self.config.game_size):
                    f.write(f"\n{players[i].name} is a {self.config.role_map[solutions[0][i]]}.")
                f.write("\n")
                f.write("-"*50)
                f.write("\nLLM solution: \n")
                f.write(parsed_solution)

            for i in range(self.config.game_size):
                if players[i].name in solution_map:
                    if (solution_map[players[i].name] in ["knight", "knave", "spy"]):
                        if  (solution_map[players[i].name] != self.config.role_map[solutions[0][i]]):
                            return True, True
                    else:
                        with open("error.txt", "a") as f:
                            f.write(f"Game {game_id} has an irregular output from LLM. \n")
                else:
                    with open("error.txt", "a") as f:
                        f.write(f"Game {game_id} has an irregular output from LLM. \n")
            return True, False
        return False, False

    def generate_ground_truth(self, num_game: int, config: base_config):
        """
        Generate a certain number of games with a certain configuration and test the basic functionality of the project.
        args:
        - num_game: int, the number of games to generate
        - config: base_config, the configuration of the games
        return:
        - saved in jsonl format in groundtruth folder. Named by the number of players (game size).
        """
        # Create output file paths
        output_file = f"./groundtruth/{config.game_size}.jsonl"
        md_output_file = f"./groundtruth/{config.game_size}_readable.md"
        
        count = 0
        while count < num_game:
            
            # Generate abstract game
            schemas, statements = self.generator.generate_single_game_abstract()
            solutions = self.validator.validate_single_game(schemas, statements)
            
            # Only proceed if there's exactly one unique solution
            if len(solutions) != 1:
                continue

            # Generate unique game ID
            game_id = self._id_generator(config)
            
            # Generate text version of the game
            players = self.generator.generate_single_game_text(schemas, statements)
            game_text = ""
            for (index, single_player) in enumerate(players):
                game_text += "---\n"
                if index != len(players) - 1:
                    game_text += f"Player name: {single_player.name}\n"
                    game_text += f"Player statement: {single_player.statement}\n"
                else:
                    game_text += f"Message from the game manager: {single_player.statement}\n"
            
            # Generate text solution
            text_solution = ""
            for i in range(config.game_size):
                text_solution += f"{players[i].name} is a {config.role_map[solutions[0][i]]}.\n"
            
            # Create ground_truth instance
            gt_instance = ground_truth(
                game_id=int(game_id.split('-')[1]),  # Extract numeric part
                num_player=config.game_size,
                num_spy=config.num_spy,
                num_hint=config.num_hint,
                raw_schema=schemas,
                raw_statement=statements,
                raw_solution=solutions[0],
                text_game=game_text,
                text_solution=text_solution
            )
            
            # Save to JSONL file
            with open(output_file, 'a') as f:
                import json
                f.write(json.dumps(gt_instance.__dict__) + '\n')
            
            # Save to markdown file for readability
            md_content = self.ground_truth_text_display(gt_instance, config)
            with open(md_output_file, 'a') as f:
                f.write(md_content)
            
            count += 1
            print(f"Generated game {count}/{num_game}: {game_id}")
        
        print(f"Successfully generated {num_game} games and saved to:")
        print(f"  - JSONL: {output_file}")
        print(f"  - Markdown: {md_output_file}")
    
    def ground_truth_text_display(self, gt_instance: ground_truth, config: base_config) -> str:
        """
        Convert a ground_truth instance to a readable markdown format.
        args:
        - gt_instance: ground_truth, the game instance to display
        - config: base_config, the configuration used for the game
        return:
        - formatted markdown string
        """
        md_content = f"""# Game id: {config.game_size}-{gt_instance.game_id}

## Game Text
{gt_instance.text_game}

## Solution
{gt_instance.text_solution}

==========
"""
        return md_content
    
    def _id_generator(self, config: base_config):
        """
        Generate a unique id for the game.
        args:
        - config: base_config, the configuration of the games
        return:
        - a unique id for the game
        flow:
        - load the file ./groundtruth/record.json
        - compose an id of the form game_size-(current_count+1)
        """
        import os
        record_path = "./groundtruth/record.json"
        
        # Load existing record
        if os.path.exists(record_path):
            with open(record_path, 'r') as f:
                record = json.load(f)
        else:
            record = {}
        
        # Get current count for this game size
        game_size_key = str(config.game_size)
        current_count = record.get(game_size_key, 0)
        
        # Generate new ID and update count
        new_count = current_count + 1
        game_id = f"{config.game_size}-{new_count}"
        
        # Update and save record
        record[game_size_key] = new_count
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)
        
        return game_id

if __name__ == "__main__":
    gt = GroundTruth(game_config)
    
    # Test generate_ground_truth function with 3 games
    print("Testing generate_ground_truth function with markdown output...")
    gt.generate_ground_truth(90, game_config)