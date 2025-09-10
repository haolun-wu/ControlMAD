from config import test_config
from project_types import ground_truth, test_setup
from utility import openai_client, ParallelProcessor, ali_client
from prompts import kks_system_prompt, kks_response_schema
import json
import os
from dataclasses import asdict
import threading

class Test:
    def __init__(self, test_config: test_setup):
        self.size = test_config.game_size
        self.provider = test_config.provider
        self.model = test_config.model
        self.pass_at = test_config.pass_at
        self.num_worker = test_config.num_worker
        self.show_thinking = test_config.show_thinking
        self.groundtruth_path = test_config.groundtruth_path + f"/{test_config.game_size}.jsonl"
        self.output_response_path = test_config.output_path + f"/{test_config.game_size}.jsonl"
        self.output_full_path = test_config.output_path + f"/{test_config.game_size}_full.jsonl"
        self._file_lock = threading.Lock()  # Thread-safe file writing
        self.enable_thinking = test_config.enable_thinking
        self.stream = test_config.stream
        self.verbosity = test_config.verbosity
        self.reasoning_effort = test_config.reasoning_effort
        self.reasoning_summary = test_config.reasoning_summary
        self.return_full_response = test_config.return_full_response
        self.truncation = test_config.truncation
        self.groundtruth = self.load_ground_truth()
    
    def load_ground_truth(self):
        """
        Load ground truth data from JSONL file.
        
        Returns:
            list[ground_truth]: List of ground truth objects
        """
        ground_truth_list = []
        
        try:
            with open(self.groundtruth_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        # Parse JSON line
                        data = json.loads(line)
                        
                        # Create ground_truth object
                        gt = ground_truth(
                            game_id=data['game_id'],
                            num_player=data['num_player'],
                            num_spy=data['num_spy'],
                            num_hint=data['num_hint'],
                            raw_schema=data['raw_schema'],
                            raw_statement=data['raw_statement'],
                            raw_solution=data['raw_solution'],
                            text_game=data['text_game'],
                            text_solution=data['text_solution']
                        )
                        
                        ground_truth_list.append(gt)
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
                    except KeyError as e:
                        print(f"Warning: Missing required field {e} on line {line_num}")
                        continue
                        
        except FileNotFoundError:
            print(f"Error: Ground truth file not found: {self.groundtruth_path}")
            return []
        except Exception as e:
            print(f"Error reading ground truth file: {e}")
            return []
        if isinstance(self.truncation, int):
            if self.truncation > 0 and self.truncation < len(ground_truth_list):
                ground_truth_list = ground_truth_list[:self.truncation]
        return ground_truth_list

    def _single_gemini_test_worker(self, case: ground_truth):
        """
        Worker function for parallel Gemini testing.
        
        Args:
            case: ground_truth object to test
            
        Returns:
            dict: {"game_id": int, "llm": response_format}
        """
        from utility import gemini_client
        
        try:
            # Initialize Gemini client
            client = gemini_client("secret.json")
            
            # Replace the number of players in the system prompt
            system_prompt = kks_system_prompt.replace("{num_player}", str(case.num_player))
            
            # Make API call using chat_completion
            response_obj = client.chat_completion(
                user_prompt=case.text_game,
                system_prompt=system_prompt,
                model=self.model,
                response_schema=kks_response_schema
            )
            
            # Wrap response with game_id
            wrapped_response = {
                "game_id": case.game_id,
                "llm": response_obj
            }
            
            return wrapped_response
            
        except Exception as e:
            print(f"Error testing case {case.game_id}: {e}")
            # Return error response wrapped with game_id
            from project_types import response_format, token_usage
            error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
            error_response = response_format(
                text="ERROR",
                usage=error_token_usage,
                summary="",
                error=str(e)
            )
            
            wrapped_error_response = {
                "game_id": case.game_id,
                "llm": error_response
            }
            
            return wrapped_error_response

    def _single_ali_test_worker(self, case: ground_truth):
        """
        Worker function for parallel Ali testing.
        
        Args:
            case: ground_truth object to test
            
        Returns:
            dict: {"game_id": int, "llm": response_format}
        """
        try:
            # Initialize Ali client
            client = ali_client("secret.json")
            
            # Replace the number of players in the system prompt
            system_prompt = kks_system_prompt.replace("{num_player}", str(case.num_player))
            
            # Make API call using chat_completion with streaming enabled
            response_obj = client.chat_completion(
                user_prompt=case.text_game,
                system_prompt=system_prompt,
                model=self.model,
                stream=self.stream
            )
            
            # Wrap response with game_id
            wrapped_response = {
                "game_id": case.game_id,
                "llm": response_obj
            }
            
            return wrapped_response
            
        except Exception as e:
            print(f"Error testing case {case.game_id}: {e}")
            # Return error response wrapped with game_id
            from project_types import response_format, token_usage
            error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
            error_response = response_format(
                text="ERROR",
                usage=error_token_usage,
                summary="",
                error=str(e)
            )
            
            wrapped_error_response = {
                "game_id": case.game_id,
                "llm": error_response
            }
            
            return wrapped_error_response

    def _single_openai_test_worker(self, case: ground_truth):
        """
        Worker function for parallel OpenAI testing.
        
        Args:
            case: ground_truth object to test
            
        Returns:
            dict: {"game_id": int, "llm": response_format}
        """
        try:
            # Initialize OpenAI client
            client = openai_client("secret.json")
            
            # Replace the number of players in the system prompt
            system_prompt = kks_system_prompt.replace("{num_player}", str(case.num_player))
            
            # Make API call using response_completion
            response_obj = client.response_completion(
                user_prompt=case.text_game,
                system_prompt=system_prompt,
                verbosity = self.verbosity,
                reasoning_effort = self.reasoning_effort
            )
            
            # Wrap response with game_id
            wrapped_response = {
                "game_id": case.game_id,
                "llm": response_obj
            }
            
            return wrapped_response
            
        except Exception as e:
            print(f"Error testing case {case.game_id}: {e}")
            # Return error response wrapped with game_id
            from project_types import response_format, token_usage
            error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
            error_response = response_format(
                text="ERROR",
                usage=error_token_usage,
                summary="",
                error=str(e)
            )
            
            wrapped_error_response = {
                "game_id": case.game_id,
                "llm": error_response
            }
            
            return wrapped_error_response
    
    def _write_response_to_file(self, wrapped_response):
        """
        Thread-safe method to write wrapped response to JSONL file.
        
        Args:
            wrapped_response: dict with {"game_id": int, "llm": response_format}
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_response_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Thread-safe file writing
        with self._file_lock:
            with open(self.output_response_path, 'a', encoding='utf-8') as file:
                # Convert to dictionary (including nested dataclasses)
                wrapped_dict = {
                    "game_id": wrapped_response["game_id"],
                    "llm": asdict(wrapped_response["llm"])
                }
                file.write(json.dumps(wrapped_dict, ensure_ascii=False) + '\n')
    
    def test_parallel_gemini(self, num_workers: int = None, cases: list[ground_truth] = None):
        """
        Test multiple cases in parallel using Gemini client.
        
        Args:
            cases: List of ground_truth objects to test (default: all loaded cases)
            num_workers: Number of parallel workers (default: self.num_worker)
            
        Returns:
            list: List of response_format objects in same order as input cases
        """
        if cases is None:
            cases = self.groundtruth
        
        if num_workers is None:
            num_workers = self.num_worker
        
        print(f"Starting parallel Gemini testing with {num_workers} workers for {len(cases)} cases...")
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Progress callback function - progress reporting + immediate error detection
        def progress_callback(index, result, completed, total):
            if result:
                game_id = result["game_id"]
                response_obj = result["llm"]
                # Check for errors in the response
                if response_obj.error:
                    print(f"⚠️  ERROR in case {game_id} (order #{index+1}, {completed}/{total}): {response_obj.error}")
                else:
                    print(f"✅ Completed case {game_id} (order #{index+1}, {completed}/{total})")
            else:
                print(f"❌ Failed case at order #{index+1} ({completed}/{total}) - task exception")
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_gemini_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Extract response objects in original order and write to file sequentially
        response_objects = []
        error_count = 0
        failed_count = 0
        
        print("\nWriting results to file in original order...")
        
        for i, result in enumerate(results):
            if result:
                game_id = result["game_id"]
                response_obj = result["llm"]
                response_objects.append(result)
                # Write wrapped response to file in original order
                self._write_response_to_file(result)
                
                # Track errors for summary
                if response_obj.error:
                    error_count += 1
                    print(f"⚠️  Written case {game_id} (position {i+1}/{len(results)}) - WITH ERROR (will count as wrong answer)")
                else:
                    print(f"✅ Written case {game_id} (position {i+1}/{len(results)})")
            else:
                # Create error response for completely failed tasks to preserve order
                from project_types import response_format, token_usage
                failed_case = cases[i]  # Get the original case to extract game_id
                
                error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                error_response = response_format(
                    text="ERROR",
                    usage=error_token_usage,
                    summary="",
                    error=f"Task completely failed for case {failed_case.game_id}"
                )
                
                # Wrap error response with game_id
                wrapped_error_response = {
                    "game_id": failed_case.game_id,
                    "llm": error_response
                }
                
                response_objects.append(wrapped_error_response)
                # Write wrapped error response to maintain JSONL order
                self._write_response_to_file(wrapped_error_response)
                
                failed_count += 1
                print(f"⚠️  Written case {failed_case.game_id} (position {i+1}/{len(results)}) - TASK FAILED (error response written to preserve order)")
        
        # Summary report
        successful_count = len([r for r in response_objects if r and not r["llm"].error])
        total_errors = error_count + failed_count  # Both API errors and task failures
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {len(results)} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation: JSONL line {i+1} = input case {i+1}")
        
        return response_objects

    def test_parallel_ali(self, cases: list[ground_truth] = None, num_workers: int = None):
        """
        Test multiple cases in parallel using Ali client.
        
        Args:
            cases: List of ground_truth objects to test (default: all loaded cases)
            num_workers: Number of parallel workers (default: self.num_worker)
            
        Returns:
            list: List of response_format objects in same order as input cases
        """
        if cases is None:
            cases = self.groundtruth
        
        if num_workers is None:
            num_workers = self.num_worker
        
        print(f"Starting parallel Ali testing with {num_workers} workers for {len(cases)} cases...")
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Progress callback function - progress reporting + immediate error detection
        def progress_callback(index, result, completed, total):
            if result:
                game_id = result["game_id"]
                response_obj = result["llm"]
                # Check for errors in the response
                if response_obj.error:
                    print(f"⚠️  ERROR in case {game_id} (order #{index+1}, {completed}/{total}): {response_obj.error}")
                else:
                    print(f"✅ Completed case {game_id} (order #{index+1}, {completed}/{total})")
            else:
                print(f"❌ Failed case at order #{index+1} ({completed}/{total}) - task exception")
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_ali_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Extract response objects in original order and write to file sequentially
        response_objects = []
        error_count = 0
        failed_count = 0
        
        print("\nWriting results to file in original order...")
        
        for i, result in enumerate(results):
            if result:
                game_id = result["game_id"]
                response_obj = result["llm"]
                response_objects.append(result)
                # Write wrapped response to file in original order
                self._write_response_to_file(result)
                
                # Track errors for summary
                if response_obj.error:
                    error_count += 1
                    print(f"⚠️  Written case {game_id} (position {i+1}/{len(results)}) - WITH ERROR (will count as wrong answer)")
                else:
                    print(f"✅ Written case {game_id} (position {i+1}/{len(results)})")
            else:
                # Create error response for completely failed tasks to preserve order
                from project_types import response_format, token_usage
                failed_case = cases[i]  # Get the original case to extract game_id
                
                error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                error_response = response_format(
                    text="ERROR",
                    usage=error_token_usage,
                    summary="",
                    error=f"Task completely failed for case {failed_case.game_id}"
                )
                
                # Wrap error response with game_id
                wrapped_error_response = {
                    "game_id": failed_case.game_id,
                    "llm": error_response
                }
                
                response_objects.append(wrapped_error_response)
                # Write wrapped error response to maintain JSONL order
                self._write_response_to_file(wrapped_error_response)
                
                failed_count += 1
                print(f"⚠️  Written case {failed_case.game_id} (position {i+1}/{len(results)}) - TASK FAILED (error response written to preserve order)")
        
        # Summary report
        successful_count = len([r for r in response_objects if r and not r["llm"].error])
        total_errors = error_count + failed_count  # Both API errors and task failures
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {len(results)} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation: JSONL line {i+1} = input case {i+1}")
        
        return response_objects

    def test_parallel_openai(self, cases: list[ground_truth] = None, num_workers: int = None):
        """
        Test multiple cases in parallel using OpenAI client.
        
        Args:
            cases: List of ground_truth objects to test (default: all loaded cases)
            num_workers: Number of parallel workers (default: self.num_worker)
            
        Returns:
            list: List of response_format objects in same order as input cases
        """
        if cases is None:
            cases = self.groundtruth
        
        if num_workers is None:
            num_workers = self.num_worker
        
        print(f"Starting parallel OpenAI testing with {num_workers} workers for {len(cases)} cases...")
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Progress callback function - progress reporting + immediate error detection
        def progress_callback(index, result, completed, total):
            if result:
                game_id = result["game_id"]
                response_obj = result["llm"]
                # Check for errors in the response
                if response_obj.error:
                    print(f"⚠️  ERROR in case {game_id} (order #{index+1}, {completed}/{total}): {response_obj.error}")
                else:
                    print(f"✅ Completed case {game_id} (order #{index+1}, {completed}/{total})")
            else:
                print(f"❌ Failed case at order #{index+1} ({completed}/{total}) - task exception")
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_openai_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Extract response objects in original order and write to file sequentially
        response_objects = []
        error_count = 0
        failed_count = 0
        
        print("\nWriting results to file in original order...")
        
        for i, result in enumerate(results):
            if result:
                game_id = result["game_id"]
                response_obj = result["llm"]
                response_objects.append(result)
                # Write wrapped response to file in original order
                self._write_response_to_file(result)
                
                # Track errors for summary
                if response_obj.error:
                    error_count += 1
                    print(f"⚠️  Written case {game_id} (position {i+1}/{len(results)}) - WITH ERROR (will count as wrong answer)")
                else:
                    print(f"✅ Written case {game_id} (position {i+1}/{len(results)})")
            else:
                # Create error response for completely failed tasks to preserve order
                from project_types import response_format, token_usage
                failed_case = cases[i]  # Get the original case to extract game_id
                
                error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                error_response = response_format(
                    text="ERROR",
                    usage=error_token_usage,
                    summary="",
                    error=f"Task completely failed for case {failed_case.game_id}"
                )
                
                # Wrap error response with game_id
                wrapped_error_response = {
                    "game_id": failed_case.game_id,
                    "llm": error_response
                }
                
                response_objects.append(wrapped_error_response)
                # Write wrapped error response to maintain JSONL order
                self._write_response_to_file(wrapped_error_response)
                
                failed_count += 1
                print(f"⚠️  Written case {failed_case.game_id} (position {i+1}/{len(results)}) - TASK FAILED (error response written to preserve order)")
        
        # Summary report
        successful_count = len([r for r in response_objects if r and not r["llm"].error])
        total_errors = error_count + failed_count  # Both API errors and task failures
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {len(results)} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation: JSONL line {i+1} = input case {i+1}")
        
        return response_objects

    def run_parallel_test_with_config(self):
        """
        Run parallel test based on the provider configuration.
        
        Returns:
            list: List of response_format objects in same order as input cases
        """
        provider_name = self.provider.lower().strip()
        
        if provider_name == "openai":
            return self.test_parallel_openai(cases=self.groundtruth, num_workers=self.num_worker)
        elif provider_name == "gemini":
            return self.test_parallel_gemini(cases=self.groundtruth, num_workers=self.num_worker)
        elif provider_name == "ali":
            return self.test_parallel_ali(cases=self.groundtruth, num_workers=self.num_worker)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Supported providers are: OpenAI, Gemini, Ali")

    def verify_test_results(self):
        """
        Verify the test results by comparing ground truth with LLM responses.
        
        Returns:
            int: Number of correct answers
        """
        # 1. Load the JSONL file with responses
        responses = []
        try:
            with open(self.output_response_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        # Parse JSON line (response_format as dict)
                        response_data = json.loads(line)
                        responses.append(response_data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in response file on line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"Error: Response file not found: {self.output_response_path}")
            return 0
        except Exception as e:
            print(f"Error reading response file: {e}")
            return 0
        
        print(f"Loaded {len(responses)} responses")
        
        # 2. Match ground truth and responses via game_id
        
        correct_answers = 0
        total_matched = 0
        
        for response_data in responses:
            # Extract game_id from wrapped response
            if "game_id" not in response_data or "llm" not in response_data:
                print(f"Warning: Response missing game_id or llm field, skipping")
                continue
                
            game_id = response_data["game_id"]
            llm_response = response_data["llm"]
            
            # Find corresponding ground truth by game_id
            current_gt = None
            for gt in self.groundtruth:
                if gt.game_id == game_id:
                    current_gt = gt
                    break
            
            if current_gt is None:
                print(f"Warning: No ground truth found for game_id {game_id}")
                continue
                
            total_matched += 1
            
            # 3. Parse the solution property of the groundtruth for pairs ["player", "role"]
            gt_pairs = self._parse_groundtruth_solution(current_gt)
            
            # 4. Parse the LLM's answer from JSON format to get list of pairs ["player", "role"]
            llm_pairs = self._parse_llm_response(llm_response)
            
            # 5. Compare the two answers using sets (order-independent)
            if gt_pairs is not None and llm_pairs is not None:
                gt_set = set(gt_pairs)
                llm_set = set(llm_pairs)
                
                if gt_set == llm_set:
                    correct_answers += 1
                    print(f"✓ Game {game_id}: Correct")
                else:
                    print(f"✗ Game {game_id}: Incorrect")
                    print(f"  Expected: {sorted(gt_pairs)}")
                    print(f"  Got:      {sorted(llm_pairs)}")
            else:
                # 6. If response is not parsable, treat as wrong answer
                print(f"✗ Game {game_id}: Unparsable response")
        
        # 7. Final output: number of correct answers
        print(f"\nResults: {correct_answers}/{total_matched} correct answers")
        return correct_answers
    
    def _parse_groundtruth_solution(self, gt: ground_truth):
        """
        Parse ground truth solution to extract player-role pairs.
        
        Args:
            gt: ground_truth object
            
        Returns:
            list: List of (player, role) tuples, or None if parsing fails
        """
        try:
            # Parse the text_solution which contains plain text like:
            # "Tina is a knight.\nXavier is a knight.\nUma is a knight.\nZane is a spy.\n"
            solution_text = gt.text_solution.strip()
            
            import re
            pairs = []
            
            # Pattern to match "PlayerName is a role."
            pattern = r'(\w+)\s+is\s+a\s+(knight|knave|spy)\.'
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            
            for player_name, role in matches:
                pairs.append((player_name, role.lower()))
            
            if pairs:
                return pairs
            else:
                print(f"No valid player-role pairs found in solution: {solution_text}")
                return None
                
        except Exception as e:
            print(f"Error parsing ground truth solution: {e}")
            return None
    
    def _parse_llm_response(self, response_data):
        """
        Parse LLM response to extract player-role pairs.
        
        Args:
            response_data: Dictionary containing response_format data
            
        Returns:
            list: List of (player, role) tuples, or None if parsing fails
        """
        try:
            # Extract the text from response_format
            response_text = response_data.get('text', '')
            
            if not response_text or response_text == "ERROR":
                return None
            
            # Try to parse JSON from the response text
            try:
                llm_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try to extract JSON pattern
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    llm_data = json.loads(json_str)
                else:
                    print(f"No valid JSON found in response: {response_text[:100]}...")
                    return None
            
            # Parse the JSON based on expected format
            pairs = []
            
            # Check if it's the new format with "players" array
            if "players" in llm_data and isinstance(llm_data["players"], list):
                for player_data in llm_data["players"]:
                    if isinstance(player_data, dict) and "name" in player_data and "role" in player_data:
                        pairs.append((player_data["name"], player_data["role"]))
            else:
                # Old format - direct name: role mapping
                for key, value in llm_data.items():
                    if key != "explanation" and isinstance(value, str):
                        pairs.append((key, value))
            
            return pairs
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None
        

if __name__ == "__main__":
    test = Test(test_config)
    cases = test.groundtruth[:10]
    print(f"Loaded {len(cases)} ground truth.")
    
    # Clear previous output file
    if os.path.exists(test.output_response_path):
        os.remove(test.output_response_path)
    
    # Test with parallel Gemini processing
    print("\n" + "="*80)
    print("PARALLEL TESTING STARTS")
    print("="*80)
    
    response_objects = test.run_parallel_test_with_config()
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    test.verify_test_results()
