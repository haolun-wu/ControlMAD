import sys
import os
import json
from dataclasses import asdict
import threading

# Add project root to Python path for direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change working directory to project root for file paths
os.chdir(project_root)

from utils.config import test_config
from utils.project_types import ground_truth, test_setup
from utils.utility import openai_client, ParallelProcessor, ali_client, volcano_client, cstcloud
from baseline.prompt_extract import PromptExtractor

# Placeholder for claude_client since it's not implemented in utils.utility
class claude_client:
    def __init__(self, secret_path: str):
        raise NotImplementedError("Claude client is not implemented yet")
    
    def chat_completion(self, **kwargs):
        raise NotImplementedError("Claude client is not implemented yet")

class Test:
    def __init__(self, test_config: test_setup):
        self.size = test_config.game_size
        self.provider = test_config.provider
        self.model = test_config.model
        self.pass_at = test_config.pass_at
        self.num_worker = test_config.num_worker
        self.show_thinking = test_config.show_thinking
        self.groundtruth_path = os.path.join(test_config.groundtruth_path, f"{test_config.game_size}.jsonl")
        # Handle both directory and file paths for output_path
        if test_config.output_path.endswith('.jsonl'):
            # If output_path is already a file path, use it directly (convert to absolute path)
            self.output_response_path = os.path.abspath(test_config.output_path)
            # For full path, replace .jsonl with _full.jsonl
            self.output_full_path = os.path.abspath(test_config.output_path.replace('.jsonl', '_full.jsonl'))
        else:
            # If output_path is a directory, use the old behavior
            self.output_response_path = os.path.abspath(os.path.join(test_config.output_path, f"{test_config.game_size}.jsonl"))
            self.output_full_path = os.path.abspath(os.path.join(test_config.output_path, f"{test_config.game_size}_full.jsonl"))
        self._file_lock = threading.Lock()  # Thread-safe file writing
        self.enable_thinking = test_config.enable_thinking
        self.thinking_budget = test_config.thinking_budget
        self.stream = test_config.stream
        self.verbosity = test_config.verbosity
        self.reasoning_effort = test_config.reasoning_effort
        self.reasoning_summary = test_config.reasoning_summary
        self.return_full_response = test_config.return_full_response
        self.truncation = test_config.truncation
        self.groundtruth = self.load_ground_truth()
        self.prompts = PromptExtractor().extract_prompts()
        self.schemas = PromptExtractor().extract_schemas()
        self.maximal_token = test_config.maximal_token
    
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
        from utils.utility import gemini_client
        
        try:
            # Initialize Gemini client
            client = gemini_client("secret.json")
            
            # Replace the number of players in the system prompt
            system_prompt = self.prompts["kks_system_prompt"].replace("{num_player}", str(case.num_player))
            
            # Make API call using chat_completion
            # Prepare kwargs for the API call
            api_kwargs = {
                "user_prompt": case.text_game,
                "system_prompt": system_prompt,
                "model": self.model,
                "response_schema": self.schemas["kks_response_schema_gemini"]
            }

            # Add max_output_tokens if maximal_token is not None
            if self.maximal_token is not None:
                api_kwargs["max_output_tokens"] = self.maximal_token

            response_obj = client.chat_completion(**api_kwargs)
            
            # Wrap response with game_id
            wrapped_response = {
                "game_id": case.game_id,
                "llm": response_obj
            }
            
            return wrapped_response
            
        except Exception as e:
            print(f"Error testing case {case.game_id}: {e}")
            # Return error response wrapped with game_id
            from utils.project_types import response_format, token_usage
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
            system_prompt = self.prompts["kks_system_prompt"].replace("{num_player}", str(case.num_player))
            
            # Make API call using chat_completion
            response_obj = client.chat_completion(
                user_prompt=case.text_game,
                system_prompt=system_prompt,
                model=self.model,
                stream=self.stream,
                enable_thinking=self.enable_thinking,
                thinking_budget=self.thinking_budget,
                maximal_token=self.maximal_token
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
            from utils.project_types import response_format, token_usage
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
            system_prompt = self.prompts["kks_system_prompt"].replace("{num_player}", str(case.num_player))
            
            # Make API call using response_completion
            response_obj = client.response_completion(
                user_prompt=case.text_game,
                system_prompt=system_prompt,
                model=self.model,
                verbosity=self.verbosity,
                reasoning_effort=self.reasoning_effort,
                schema_format=self.schemas["kks_response_schema_openai"],
                reasoning_summary=self.reasoning_summary
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
            from utils.project_types import response_format, token_usage
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

    def _single_volcano_test_worker(self, case: ground_truth):
        """
        Worker function for parallel Volcano Engine testing.
        
        Args:
            case: ground_truth object to test
            
        Returns:
            dict: {"game_id": int, "llm": response_format}
        """
        try:
            # Initialize Volcano client
            client = volcano_client("secret.json")
            
            # Replace the number of players in the system prompt
            system_prompt = self.prompts["kks_system_prompt"].replace("{num_player}", str(case.num_player))
            
            # Map enable_thinking config to thinking_type parameter
            if hasattr(self, 'enable_thinking') and isinstance(self.enable_thinking, bool):
                thinking_type = "enabled" if self.enable_thinking else "disabled"
            else:
                thinking_type = "auto"  # Default to auto mode
            
            # Make API call using chat_completion
            response_obj = client.chat_completion(
                user_prompt=case.text_game,
                system_prompt=system_prompt,
                model=self.model,
                max_tokens=self.maximal_token,
                thinking_type=thinking_type,
                stream=self.stream,
                temperature=0.7,  # Default temperature
                json_schema=self.schemas["kks_response_schema_volcano"]
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
            from utils.project_types import response_format, token_usage
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

    def _single_cstcloud_test_worker(self, case: ground_truth):
        """
        Worker function for parallel CST Cloud (DeepSeek) testing.
        
        Args:
            case: ground_truth object to test
            
        Returns:
            dict: {"game_id": int, "llm": response_format}
        """
        try:
            # Initialize CST Cloud client
            client = cstcloud("secret.json")
            
            # Replace the number of players in the system prompt
            system_prompt = self.prompts["kks_system_prompt"].replace("{num_player}", str(case.num_player))
            
            # Make API call using chat_completion
            response_obj = client.chat_completion(
                user_prompt=case.text_game,
                system_prompt=system_prompt,
                model=self.model,
                stream=self.stream,
                max_tokens=self.maximal_token
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
            from utils.project_types import response_format, token_usage
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

    def _single_claude_test_worker(self, case: ground_truth):
        """
        Worker function for parallel Claude testing.
        
        Args:
            case: ground_truth object to test
            
        Returns:
            dict: {"game_id": int, "llm": response_format}
        """
        try:
            # Initialize Claude client
            client = claude_client("secret.json")
            
            # Replace the number of players in the system prompt
            system_prompt = self.prompts["kks_system_prompt"].replace("{num_player}", str(case.num_player))
            
            # Make API call using chat_completion
            response_obj = client.chat_completion(
                user_prompt=case.text_game,
                system_prompt=system_prompt,
                model=self.model,
                stream=self.stream,
                max_tokens=self.maximal_token
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
            from utils.project_types import response_format, token_usage
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
    
    def _get_last_processed_game_id(self):
        """
        Get the last processed game_id from the output file.
        
        Returns:
            int: Last processed game_id, or -1 if file doesn't exist or is empty
        """
        try:
            if not os.path.exists(self.output_response_path):
                return -1
            
            last_game_id = -1
            with open(self.output_response_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if "game_id" in data:
                            last_game_id = max(last_game_id, data["game_id"])
                    except json.JSONDecodeError:
                        continue
            
            return last_game_id
            
        except Exception as e:
            print(f"Warning: Error reading output file to get last game_id: {e}")
            return -1

    def _filter_remaining_cases(self, cases: list[ground_truth], last_processed_game_id: int):
        """
        Filter cases to only include those that haven't been processed yet.
        
        Args:
            cases: List of ground_truth objects to filter
            last_processed_game_id: Last game_id that was already processed
            
        Returns:
            list[ground_truth]: Filtered list of remaining cases to process
        """
        if last_processed_game_id == -1:
            # No previous results, process all cases
            return cases
        
        # Filter out already processed cases
        remaining_cases = []
        for case in cases:
            if case.game_id > last_processed_game_id:
                remaining_cases.append(case)
        
        return remaining_cases

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
        
        # Resume functionality: Check what's already been processed and filter remaining cases
        last_processed_game_id = self._get_last_processed_game_id()
        remaining_cases = self._filter_remaining_cases(cases, last_processed_game_id)
        
        if not remaining_cases:
            print(f"All cases already processed (up to game_id {last_processed_game_id}). Nothing to do.")
            return []
        
        print(f"Resume detected: Last processed game_id was {last_processed_game_id}")
        print(f"Processing {len(remaining_cases)} remaining cases out of {len(cases)} total cases")
        
        # Use remaining_cases instead of cases for the rest of the function
        cases = remaining_cases
        
        print(f"Starting parallel Gemini testing with {num_workers} workers for {len(cases)} cases...")
        
        # Simple dynamic appending variables
        next_to_append = 0  # Pointer to next case index to write
        waiting_pool = {}   # {index: result} for out-of-order results
        written_count = 0
        
        # Helper function to write result and check waiting pool
        def write_and_check_pool(result):
            nonlocal next_to_append, written_count
            
            # Write the result
            self._write_response_to_file(result)
            written_count += 1
            next_to_append += 1
            
            # Check waiting pool for consecutive results
            while next_to_append in waiting_pool:
                next_result = waiting_pool.pop(next_to_append)
                self._write_response_to_file(next_result)
                written_count += 1
                next_to_append += 1
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Simple progress callback with dynamic appending
        def progress_callback(index, result, completed, total):
            original_case = cases[index]
            
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
                # Create error response for failed case
                if original_case:
                    from utils.project_types import response_format, token_usage
                    error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                    error_response = response_format(
                        text="ERROR",
                        usage=error_token_usage,
                        summary="",
                        error=f"Task completely failed for case {original_case.game_id}"
                    )
                    result = {
                        "game_id": original_case.game_id,
                        "llm": error_response
                    }
            
            # Dynamic appending logic
            if index == next_to_append:
                # This is the next case we're waiting for - write immediately
                write_and_check_pool(result)
            else:
                # Store in waiting pool for later
                waiting_pool[index] = result
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        print(f"\nDynamic writing enabled: Results will be written to file immediately as they complete...")
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_gemini_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Simple summary calculation
        successful_count = 0
        error_count = 0
        failed_count = 0
        
        for result in results:
            if result:
                response_obj = result["llm"]
                if response_obj.error:
                    error_count += 1
                else:
                    successful_count += 1
            else:
                failed_count += 1
        
        total_errors = error_count + failed_count
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {written_count} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation maintained with dynamic writing")
        
        # Return response objects for compatibility (though they're already written to file)
        return results

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
        
        # Resume functionality: Check what's already been processed and filter remaining cases
        last_processed_game_id = self._get_last_processed_game_id()
        remaining_cases = self._filter_remaining_cases(cases, last_processed_game_id)
        
        if not remaining_cases:
            print(f"All cases already processed (up to game_id {last_processed_game_id}). Nothing to do.")
            return []
        
        print(f"Resume detected: Last processed game_id was {last_processed_game_id}")
        print(f"Processing {len(remaining_cases)} remaining cases out of {len(cases)} total cases")
        
        # Use remaining_cases instead of cases for the rest of the function
        cases = remaining_cases
        
        print(f"Starting parallel Ali testing with {num_workers} workers for {len(cases)} cases...")
        print(f"Model: {self.model}, Stream: {self.stream}, Thinking: {self.enable_thinking}")
        
        # Simple dynamic appending variables
        next_to_append = 0  # Pointer to next case index to write
        waiting_pool = {}   # {index: result} for out-of-order results
        written_count = 0
        
        # Helper function to write result and check waiting pool
        def write_and_check_pool(result):
            nonlocal next_to_append, written_count
            
            # Write the result
            self._write_response_to_file(result)
            written_count += 1
            next_to_append += 1
            
            # Check waiting pool for consecutive results
            while next_to_append in waiting_pool:
                next_result = waiting_pool.pop(next_to_append)
                self._write_response_to_file(next_result)
                written_count += 1
                next_to_append += 1
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Simple progress callback with dynamic appending
        def progress_callback(index, result, completed, total):
            original_case = cases[index]
            
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
                # Create error response for failed case
                if original_case:
                    from utils.project_types import response_format, token_usage
                    error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                    error_response = response_format(
                        text="ERROR",
                        usage=error_token_usage,
                        summary="",
                        error=f"Task completely failed for case {original_case.game_id}"
                    )
                    result = {
                        "game_id": original_case.game_id,
                        "llm": error_response
                    }
            
            # Dynamic appending logic
            if index == next_to_append:
                # This is the next case we're waiting for - write immediately
                write_and_check_pool(result)
            else:
                # Store in waiting pool for later
                waiting_pool[index] = result
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        print(f"\nDynamic writing enabled: Results will be written to file immediately as they complete...")
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_ali_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Simple summary calculation
        successful_count = 0
        error_count = 0
        failed_count = 0
        
        for result in results:
            if result:
                response_obj = result["llm"]
                if response_obj.error:
                    error_count += 1
                else:
                    successful_count += 1
            else:
                failed_count += 1
        
        total_errors = error_count + failed_count
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {written_count} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation maintained with dynamic writing")
        
        # Return response objects for compatibility (though they're already written to file)
        return results

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
        
        # Resume functionality: Check what's already been processed and filter remaining cases
        last_processed_game_id = self._get_last_processed_game_id()
        remaining_cases = self._filter_remaining_cases(cases, last_processed_game_id)
        
        if not remaining_cases:
            print(f"All cases already processed (up to game_id {last_processed_game_id}). Nothing to do.")
            return []
        
        print(f"Resume detected: Last processed game_id was {last_processed_game_id}")
        print(f"Processing {len(remaining_cases)} remaining cases out of {len(cases)} total cases")
        
        # Use remaining_cases instead of cases for the rest of the function
        cases = remaining_cases
        
        print(f"Starting parallel OpenAI testing with {num_workers} workers for {len(cases)} cases...")
        
        # Simple dynamic appending variables
        next_to_append = 0  # Pointer to next case index to write
        waiting_pool = {}   # {index: result} for out-of-order results
        written_count = 0
        
        # Helper function to write result and check waiting pool
        def write_and_check_pool(result):
            nonlocal next_to_append, written_count
            
            # Write the result
            self._write_response_to_file(result)
            written_count += 1
            next_to_append += 1
            
            # Check waiting pool for consecutive results
            while next_to_append in waiting_pool:
                next_result = waiting_pool.pop(next_to_append)
                self._write_response_to_file(next_result)
                written_count += 1
                next_to_append += 1
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Simple progress callback with dynamic appending
        def progress_callback(index, result, completed, total):
            original_case = cases[index]
            
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
                # Create error response for failed case
                if original_case:
                    from utils.project_types import response_format, token_usage
                    error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                    error_response = response_format(
                        text="ERROR",
                        usage=error_token_usage,
                        summary="",
                        error=f"Task completely failed for case {original_case.game_id}"
                    )
                    result = {
                        "game_id": original_case.game_id,
                        "llm": error_response
                    }
            
            # Dynamic appending logic
            if index == next_to_append:
                # This is the next case we're waiting for - write immediately
                write_and_check_pool(result)
            else:
                # Store in waiting pool for later
                waiting_pool[index] = result
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        print(f"\nDynamic writing enabled: Results will be written to file immediately as they complete...")
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_openai_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Simple summary calculation
        successful_count = 0
        error_count = 0
        failed_count = 0
        
        for result in results:
            if result:
                response_obj = result["llm"]
                if response_obj.error:
                    error_count += 1
                else:
                    successful_count += 1
            else:
                failed_count += 1
        
        total_errors = error_count + failed_count
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {written_count} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation maintained with dynamic writing")
        
        # Return response objects for compatibility (though they're already written to file)
        return results

    def test_parallel_volcano(self, cases: list[ground_truth] = None, num_workers: int = None):
        """
        Test multiple cases in parallel using Volcano Engine client.
        
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
        
        # Resume functionality: Check what's already been processed and filter remaining cases
        last_processed_game_id = self._get_last_processed_game_id()
        remaining_cases = self._filter_remaining_cases(cases, last_processed_game_id)
        
        if not remaining_cases:
            print(f"All cases already processed (up to game_id {last_processed_game_id}). Nothing to do.")
            return []
        
        print(f"Resume detected: Last processed game_id was {last_processed_game_id}")
        print(f"Processing {len(remaining_cases)} remaining cases out of {len(cases)} total cases")
        
        # Use remaining_cases instead of cases for the rest of the function
        cases = remaining_cases
        
        print(f"Starting parallel Volcano Engine testing with {num_workers} workers for {len(cases)} cases...")
        print(f"Model: {self.model}, Stream: {self.stream}, Max tokens: {self.maximal_token}")
        
        # Simple dynamic appending variables
        next_to_append = 0  # Pointer to next case index to write
        waiting_pool = {}   # {index: result} for out-of-order results
        written_count = 0
        
        # Helper function to write result and check waiting pool
        def write_and_check_pool(result):
            nonlocal next_to_append, written_count
            
            # Write the result
            self._write_response_to_file(result)
            written_count += 1
            next_to_append += 1
            
            # Check waiting pool for consecutive results
            while next_to_append in waiting_pool:
                next_result = waiting_pool.pop(next_to_append)
                self._write_response_to_file(next_result)
                written_count += 1
                next_to_append += 1
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Simple progress callback with dynamic appending
        def progress_callback(index, result, completed, total):
            original_case = cases[index]
            
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
                # Create error response for failed case
                if original_case:
                    from utils.project_types import response_format, token_usage
                    error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                    error_response = response_format(
                        text="ERROR",
                        usage=error_token_usage,
                        summary="",
                        error=f"Task completely failed for case {original_case.game_id}"
                    )
                    result = {
                        "game_id": original_case.game_id,
                        "llm": error_response
                    }
            
            # Dynamic appending logic
            if index == next_to_append:
                # This is the next case we're waiting for - write immediately
                write_and_check_pool(result)
            else:
                # Store in waiting pool for later
                waiting_pool[index] = result
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        print(f"\nDynamic writing enabled: Results will be written to file immediately as they complete...")
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_volcano_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Simple summary calculation
        successful_count = 0
        error_count = 0
        failed_count = 0
        
        for result in results:
            if result:
                response_obj = result["llm"]
                if response_obj.error:
                    error_count += 1
                else:
                    successful_count += 1
            else:
                failed_count += 1
        
        total_errors = error_count + failed_count
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {written_count} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation maintained with dynamic writing")
        
        # Return response objects for compatibility (though they're already written to file)
        return results

    def test_parallel_cstcloud(self, cases: list[ground_truth] = None, num_workers: int = None):
        """
        Test multiple cases in parallel using CST Cloud (DeepSeek) client.
        
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
        
        # Resume functionality: Check what's already been processed and filter remaining cases
        last_processed_game_id = self._get_last_processed_game_id()
        remaining_cases = self._filter_remaining_cases(cases, last_processed_game_id)
        
        if not remaining_cases:
            print(f"All cases already processed (up to game_id {last_processed_game_id}). Nothing to do.")
            return []
        
        print(f"Resume detected: Last processed game_id was {last_processed_game_id}")
        print(f"Processing {len(remaining_cases)} remaining cases out of {len(cases)} total cases")
        
        # Use remaining_cases instead of cases for the rest of the function
        cases = remaining_cases
        
        print(f"Starting parallel CST Cloud (DeepSeek) testing with {num_workers} workers for {len(cases)} cases...")
        print(f"Model: {self.model}, Stream: {self.stream}, Max tokens: {self.maximal_token}")
        
        # Simple dynamic appending variables
        next_to_append = 0  # Pointer to next case index to write
        waiting_pool = {}   # {index: result} for out-of-order results
        written_count = 0
        
        # Helper function to write result and check waiting pool
        def write_and_check_pool(result):
            nonlocal next_to_append, written_count
            
            # Write the result
            self._write_response_to_file(result)
            written_count += 1
            next_to_append += 1
            
            # Check waiting pool for consecutive results
            while next_to_append in waiting_pool:
                next_result = waiting_pool.pop(next_to_append)
                self._write_response_to_file(next_result)
                written_count += 1
                next_to_append += 1
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Simple progress callback with dynamic appending
        def progress_callback(index, result, completed, total):
            original_case = cases[index]
            
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
                # Create error response for failed case
                if original_case:
                    from utils.project_types import response_format, token_usage
                    error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                    error_response = response_format(
                        text="ERROR",
                        usage=error_token_usage,
                        summary="",
                        error=f"Task completely failed for case {original_case.game_id}"
                    )
                    result = {
                        "game_id": original_case.game_id,
                        "llm": error_response
                    }
            
            # Dynamic appending logic
            if index == next_to_append:
                # This is the next case we're waiting for - write immediately
                write_and_check_pool(result)
            else:
                # Store in waiting pool for later
                waiting_pool[index] = result
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        print(f"\nDynamic writing enabled: Results will be written to file immediately as they complete...")
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_cstcloud_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Simple summary calculation
        successful_count = 0
        error_count = 0
        failed_count = 0
        
        for result in results:
            if result:
                response_obj = result["llm"]
                if response_obj.error:
                    error_count += 1
                else:
                    successful_count += 1
            else:
                failed_count += 1
        
        total_errors = error_count + failed_count
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {written_count} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation maintained with dynamic writing")
        
        # Return response objects for compatibility (though they're already written to file)
        return results

    def test_parallel_claude(self, cases: list[ground_truth] = None, num_workers: int = None):
        """
        Test multiple cases in parallel using Claude client.
        
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
        
        # Resume functionality: Check what's already been processed and filter remaining cases
        last_processed_game_id = self._get_last_processed_game_id()
        remaining_cases = self._filter_remaining_cases(cases, last_processed_game_id)
        
        if not remaining_cases:
            print(f"All cases already processed (up to game_id {last_processed_game_id}). Nothing to do.")
            return []
        
        print(f"Resume detected: Last processed game_id was {last_processed_game_id}")
        print(f"Processing {len(remaining_cases)} remaining cases out of {len(cases)} total cases")
        
        # Use remaining_cases instead of cases for the rest of the function
        cases = remaining_cases
        
        print(f"Starting parallel Claude testing with {num_workers} workers for {len(cases)} cases...")
        print(f"Model: {self.model}, Stream: {self.stream}, Max tokens: {self.maximal_token}")
        
        # Simple dynamic appending variables
        next_to_append = 0  # Pointer to next case index to write
        waiting_pool = {}   # {index: result} for out-of-order results
        written_count = 0
        
        # Helper function to write result and check waiting pool
        def write_and_check_pool(result):
            nonlocal next_to_append, written_count
            
            # Write the result
            self._write_response_to_file(result)
            written_count += 1
            next_to_append += 1
            
            # Check waiting pool for consecutive results
            while next_to_append in waiting_pool:
                next_result = waiting_pool.pop(next_to_append)
                self._write_response_to_file(next_result)
                written_count += 1
                next_to_append += 1
        
        # Create parallel processor
        processor = ParallelProcessor(num_workers=num_workers)
        
        # Simple progress callback with dynamic appending
        def progress_callback(index, result, completed, total):
            original_case = cases[index]
            
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
                # Create error response for failed case
                if original_case:
                    from utils.project_types import response_format, token_usage
                    error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
                    error_response = response_format(
                        text="ERROR",
                        usage=error_token_usage,
                        summary="",
                        error=f"Task completely failed for case {original_case.game_id}"
                    )
                    result = {
                        "game_id": original_case.game_id,
                        "llm": error_response
                    }
            
            # Dynamic appending logic
            if index == next_to_append:
                # This is the next case we're waiting for - write immediately
                write_and_check_pool(result)
            else:
                # Store in waiting pool for later
                waiting_pool[index] = result
        
        # Prepare task arguments - just pass the case objects
        task_args = cases
        
        print(f"\nDynamic writing enabled: Results will be written to file immediately as they complete...")
        
        # Process in parallel with callback
        results = processor.process_with_callback(
            task_func=self._single_claude_test_worker,
            task_args_list=task_args,
            callback_func=progress_callback
        )
        
        # Simple summary calculation
        successful_count = 0
        error_count = 0
        failed_count = 0
        
        for result in results:
            if result:
                response_obj = result["llm"]
                if response_obj.error:
                    error_count += 1
                else:
                    successful_count += 1
            else:
                failed_count += 1
        
        total_errors = error_count + failed_count
        
        print(f"\n📊 SUMMARY:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ⚠️  API errors: {error_count}")
        print(f"   ❌ Task failures: {failed_count}")
        print(f"   🔢 Total errors: {total_errors}")
        print(f"   📁 All {written_count} results written to: {self.output_response_path}")
        print(f"   📋 Perfect order preservation maintained with dynamic writing")
        
        # Return response objects for compatibility (though they're already written to file)
        return results

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
        elif provider_name == "volcano":
            return self.test_parallel_volcano(cases=self.groundtruth, num_workers=self.num_worker)
        elif provider_name == "cstcloud":
            return self.test_parallel_cstcloud(cases=self.groundtruth, num_workers=self.num_worker)
        elif provider_name == "claude":
            return self.test_parallel_claude(cases=self.groundtruth, num_workers=self.num_worker)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Supported providers are: OpenAI, Gemini, Ali, Volcano, CST Cloud, Claude")

    def verify_test_results(self):
        """
        Verify the test results by comparing ground truth with LLM responses.
        Also calculates average confidence score.

        Returns:
            tuple: (Number of correct answers, Average confidence score)
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
            return 0, None
        except Exception as e:
            print(f"Error reading response file: {e}")
            return 0, None

        print(f"Loaded {len(responses)} responses")

        # 2. Match ground truth and responses via game_id

        correct_answers = 0
        total_matched = 0
        confidence_scores = []
        total_player_accuracy = 0.0  # For smoother accuracy calculation

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

            # 4. Parse the LLM's answer from JSON format to get list of pairs ["player", "role"] and confidence
            llm_pairs, confidence = self._parse_llm_response(llm_response)

            # Track confidence scores
            if confidence is not None:
                confidence_scores.append(confidence)

            # 5. Compare the two answers using sets (order-independent)
            if gt_pairs is not None and llm_pairs is not None:
                gt_set = set(gt_pairs)
                llm_set = set(llm_pairs)

                # Calculate smoother accuracy (percentage of players predicted correctly)
                player_accuracy = self._calculate_player_accuracy(gt_pairs, llm_pairs)
                total_player_accuracy += player_accuracy

                if gt_set == llm_set:
                    correct_answers += 1
                    conf_str = f" (confidence: {confidence:.1f})" if confidence is not None else ""
                    print(f"✓ Game {game_id}: Correct{conf_str} (player accuracy: {player_accuracy:.1%})")
                else:
                    conf_str = f" (confidence: {confidence:.1f})" if confidence is not None else ""
                    print(f"✗ Game {game_id}: Incorrect{conf_str} (player accuracy: {player_accuracy:.1%})")
                    print(f"  Expected: {sorted(gt_pairs)}")
                    print(f"  Got:      {sorted(llm_pairs)}")
            else:
                # 6. If response is not parsable, treat as wrong answer (0% player accuracy)
                print(f"✗ Game {game_id}: Unparsable response (player accuracy: 0.0%)")

        # Calculate average confidence
        avg_confidence = None
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # 7. Final output: number of correct answers and average confidence
        binary_accuracy = 100*correct_answers/total_matched if total_matched > 0 else 0
        smoother_accuracy = 100*total_player_accuracy/total_matched if total_matched > 0 else 0
        
        print(f"\n📊 RESULTS:")
        print(f"  ✅ Binary Accuracy: {correct_answers}/{total_matched} correct answers ({binary_accuracy:.1f}%)")
        print(f"  📊 Smoother Accuracy: {smoother_accuracy:.1f}% (average percentage of players predicted correctly)")
        if avg_confidence is not None:
            print(f"  🎯 Average confidence: {avg_confidence:.1f}/10")
            print(f"  📈 Confidence scores provided: {len(confidence_scores)}/{total_matched}")
        else:
            print(f"  ⚠️ No valid confidence scores found in responses")

        return correct_answers, avg_confidence, smoother_accuracy
    
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
        Parse LLM response to extract player-role pairs and confidence score.

        Args:
            response_data: Dictionary containing response_format data

        Returns:
            tuple: (list of (player, role) tuples, confidence_score) or (None, None) if parsing fails
        """
        try:
            # Extract the text from response_format
            response_text = response_data.get('text', '')

            if not response_text or response_text == "ERROR":
                return None, None

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
                    return None, None

            # Parse the JSON based on expected format
            pairs = []
            confidence = None

            # Extract confidence score if present
            if "confidence" in llm_data:
                confidence = llm_data["confidence"]
                # Ensure confidence is a number and within valid range (1-10)
                if isinstance(confidence, (int, float)):
                    confidence = float(confidence)
                    # Only accept confidence scores between 1 and 10
                    if confidence < 1 or confidence > 10:
                        confidence = None
                else:
                    confidence = None

            # Check if it's the new format with "players" array
            if "players" in llm_data and isinstance(llm_data["players"], list):
                for player_data in llm_data["players"]:
                    if isinstance(player_data, dict) and "name" in player_data and "role" in player_data:
                        pairs.append((player_data["name"], player_data["role"]))
            else:
                # Old format - direct name: role mapping
                for key, value in llm_data.items():
                    if key not in ["reasoning", "confidence"] and isinstance(value, str):
                        pairs.append((key, value))

            return pairs, confidence

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None, None
    
    def _calculate_player_accuracy(self, gt_pairs, llm_pairs):
        """
        Calculate the percentage of players that were predicted correctly.
        
        Args:
            gt_pairs: List of (player, role) tuples from ground truth
            llm_pairs: List of (player, role) tuples from LLM response
            
        Returns:
            float: Percentage of players predicted correctly (0.0 to 1.0)
        """
        if not gt_pairs or not llm_pairs:
            return 0.0
        
        # Convert to dictionaries for easier lookup
        gt_dict = dict(gt_pairs)
        llm_dict = dict(llm_pairs)
        
        # Get all unique players from both ground truth and LLM response
        all_players = set(gt_dict.keys()) | set(llm_dict.keys())
        
        if not all_players:
            return 0.0
        
        correct_predictions = 0
        total_players = len(all_players)
        
        for player in all_players:
            gt_role = gt_dict.get(player)
            llm_role = llm_dict.get(player)
            
            # If both roles exist and match, it's correct
            if gt_role and llm_role and gt_role == llm_role:
                correct_predictions += 1
            # If player is missing from either side, it's incorrect
            # (This handles cases where LLM predicts fewer/more players than ground truth)
        
        return correct_predictions / total_players if total_players > 0 else 0.0
        

if __name__ == "__main__":
    import sys
    
    # Check if command line arguments are provided for model-specific testing
    if len(sys.argv) > 1:
        # Parse command line arguments for model-specific config
        model_name = sys.argv[1] if len(sys.argv) > 1 else test_config.model
        provider = sys.argv[2] if len(sys.argv) > 2 else test_config.provider
        output_path = sys.argv[3] if len(sys.argv) > 3 else test_config.output_path
        
        # Create a custom test config
        from utils.project_types import test_setup
        custom_config = test_setup(
            game_size=test_config.game_size,
            provider=provider,
            model=model_name,
            pass_at=test_config.pass_at,
            num_worker=test_config.num_worker,
            show_thinking=test_config.show_thinking,
            groundtruth_path=test_config.groundtruth_path,
            output_path=output_path,
            enable_thinking=test_config.enable_thinking,
            thinking_budget=test_config.thinking_budget,
            stream=test_config.stream,
            verbosity=test_config.verbosity,
            reasoning_effort=test_config.reasoning_effort,
            reasoning_summary=test_config.reasoning_summary,
            return_full_response=test_config.return_full_response,
            truncation=test_config.truncation,
            maximal_token=test_config.maximal_token
        )
        test = Test(custom_config)
    else:
        test = Test(test_config)
    
    # Clear previous output file
    #if os.path.exists(test.output_response_path):
    #    os.remove(test.output_response_path)
    
    # Test with parallel processing
    print("\n" + "="*80)
    print("PARALLEL TESTING STARTS")
    print(f"Provider: {test.provider}")
    print(f"Model: {test.model}")
    print("="*80)
    
    response_objects = test.run_parallel_test_with_config()
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    test.verify_test_results()
