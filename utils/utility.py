from itertools import combinations
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from .project_types import response_format, token_usage

class Utility:

    def binary_add_one(bin):
        """
        Adds 1 to a binary number represented as a list of 0s and 1s.
        
        Args:
            bin (list): A list of binary digits (0s and 1s) with least significant bit at index 0
            
        Returns:
            list: The binary number after adding 1
        """
        for i in range(len(bin)):
            if bin[i] == 0:
                bin[i] = 1
                return bin
            else:
                bin[i] = 0
        bin.append(1)
        return bin
    
    def enumerate_sublists(self, original_list: list, size: int):
        """
        Enumerates all sublists of a given list of a given size.
        
        Args:
            original_list (list): The original list to enumerate sublists from
            size (int): The size of the sublists to enumerate
            
        Returns:
            list: A list of all sublists of the original list of the given size
        """
        return [list(combo) for combo in combinations(original_list, size)]
    
    def count_intersection(self, list1: list, list2: list):
        """
        Counts the number of elements in the intersection of two lists.
        """
        return len(set(list1) & set(list2))

class ParallelProcessor:
    """
    Utility class for parallel processing of tasks using ThreadPoolExecutor.
    Supports configurable number of workers and maintains order of results.
    """
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize the parallel processor.
        
        Args:
            num_workers (int): Number of worker threads (default: 4)
        """
        self.num_workers = num_workers
    
    def process_tasks(self, task_func, task_args_list, preserve_order: bool = True):
        """
        Process tasks in parallel using ThreadPoolExecutor.
        
        Args:
            task_func (callable): Function to execute for each task
            task_args_list (list): List of argument tuples/dicts for each task
            preserve_order (bool): Whether to preserve the order of results (default: True)
            
        Returns:
            list: Results in the same order as input tasks (if preserve_order=True)
                 or in completion order (if preserve_order=False)
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            if preserve_order:
                # Submit all tasks and maintain order
                future_to_index = {}
                for i, args in enumerate(task_args_list):
                    if isinstance(args, dict):
                        future = executor.submit(task_func, **args)
                    elif isinstance(args, (list, tuple)):
                        future = executor.submit(task_func, *args)
                    else:
                        future = executor.submit(task_func, args)
                    future_to_index[future] = i
                
                # Create results list with correct size
                results = [None] * len(task_args_list)
                
                # Collect results in order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        print(f"Task {index} failed with error: {e}")
                        results[index] = None
            else:
                # Process in completion order (faster)
                futures = []
                for args in task_args_list:
                    if isinstance(args, dict):
                        future = executor.submit(task_func, **args)
                    elif isinstance(args, (list, tuple)):
                        future = executor.submit(task_func, *args)
                    else:
                        future = executor.submit(task_func, args)
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Task failed with error: {e}")
                        results.append(None)
        
        return results
    
    def process_with_callback(self, task_func, task_args_list, callback_func=None):
        """
        Process tasks in parallel with optional callback for each completion.
        
        Args:
            task_func (callable): Function to execute for each task
            task_args_list (list): List of argument tuples/dicts for each task
            callback_func (callable): Optional function to call with (index, result) for each completion
            
        Returns:
            list: Results in original order
        """
        results = [None] * len(task_args_list)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_index = {}
            for i, args in enumerate(task_args_list):
                if isinstance(args, dict):
                    future = executor.submit(task_func, **args)
                elif isinstance(args, (list, tuple)):
                    future = executor.submit(task_func, *args)
                else:
                    future = executor.submit(task_func, args)
                future_to_index[future] = i
            
            # Process completions
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed_count += 1
                try:
                    result = future.result()
                    results[index] = result
                    if callback_func:
                        callback_func(index, result, completed_count, len(task_args_list))
                except Exception as e:
                    print(f"Task {index} failed with error: {e}")
                    results[index] = None
                    if callback_func:
                        callback_func(index, None, completed_count, len(task_args_list))
        
        return results
    
class cstcloud:

    def __init__(self, secret_path: str):
        """
        Initialize the CST Cloud API client.
        
        Args:
            secret_path (str): Path to the file containing the API key
        """
        self.base_url = "https://uni-api.cstcloud.cn/v1"
        self.api_key = self._read_api_key(secret_path)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _read_api_key(self, secret_path: str) -> str:
        """
        Read API key from the secret file.
        
        Args:
            secret_path (str): Path to the file containing the API keys in JSON format
            
        Returns:
            str: The CST Cloud API key
            
        Raises:
            FileNotFoundError: If the secret file doesn't exist
            ValueError: If the API key is empty or invalid
            KeyError: If CST Cloud provider is not found
        """
        try:
            with open(secret_path, 'r') as f:
                api_keys_data = json.load(f)
            
            if not isinstance(api_keys_data, list):
                raise ValueError("Secret file should contain a list of API providers")
            
            # Find the CST Cloud API key
            for provider in api_keys_data:
                if provider.get("API provider") == "CST Cloud":
                    api_key = provider.get("API key", "").strip()
                    if not api_key:
                        raise ValueError("CST Cloud API key is empty")
                    return api_key
            
            raise KeyError("CST Cloud provider not found in secret file")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Secret file not found: {secret_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in secret file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading API key: {e}")

    def chat_completion(self, user_prompt: str, system_prompt: str = None, model: str = "deepseek-r1:671b-0528", stream: bool = False) -> str:
        """
        Send a chat completion request to the CST Cloud API.
        
        Args:
            user_prompt (str): The user's message
            system_prompt (str): The system prompt/instruction (default: None)
            model (str): The model to use (default: "deepseek-r1:671b-0528")
            stream (bool): Whether to stream the response (default: False)
            
        Returns:
            str: The model's response text
            
        Raises:
            Exception: If the API request fails
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Make the API request
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                return full_response
            else:
                # Handle non-streaming response
                return response.choices[0].message.content
                
        except Exception as e:
            raise Exception(f"API request failed: {e}")
    
    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            response = self.chat_completion(
                user_prompt="Say 'Hello, API is working!'",
                system_prompt="You are a helpful assistant.",
                model="gpt-oss-120b"
            )
            return "Hello, API is working!" in response
        except Exception:
            return False

class openai_client:
    """
    OpenAI API client with token usage tracking and parallel call support.
    Supports GPT-5 series, GPT-4o, GPT-4o-mini, GPT-4.1-mini, and GPT-4.1-nano models with cost calculation.
    Uses the new Response API with advanced parameters like verbosity and reasoning effort.
    """

    def __init__(self, secret_path: str):
        """
        Initialize the OpenAI API client.
        
        Args:
            secret_path (str): Path to the file containing the API key
        """
        import openai as openai_lib
        
        self.api_key = self._read_api_key(secret_path)
        self.client = openai_lib.OpenAI(api_key=self.api_key)
    
    def _read_api_key(self, secret_path: str) -> str:
        """
        Read API key from the secret file.
        
        Args:
            secret_path (str): Path to the file containing the API keys in JSON format
            
        Returns:
            str: The OpenAI API key
            
        Raises:
            FileNotFoundError: If the secret file doesn't exist
            ValueError: If the API key is empty or invalid
            KeyError: If OpenAI provider is not found
        """
        try:
            with open(secret_path, 'r') as f:
                api_keys_data = json.load(f)
            
            if not isinstance(api_keys_data, list):
                raise ValueError("Secret file should contain a list of API providers")
            
            # Find the OpenAI API key
            for provider in api_keys_data:
                if provider.get("API provider") == "OpenAI":
                    api_key = provider.get("API key", "").strip()
                    if not api_key:
                        raise ValueError("OpenAI API key is empty")
                    return api_key
            
            raise KeyError("OpenAI provider not found in secret file")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Secret file not found: {secret_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in secret file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading API key: {e}")
    
    def response_completion(self, user_prompt: str, system_prompt: str = None, model: str = "gpt-5-nano", stream: bool = False, verbosity: str = "medium", reasoning_effort: str = "medium", reasoning_summary: str = None, return_full_response: bool = False, call_id: str = None, schema_format: dict = None):
        """
        Send a chat completion request using the new response API.
        Supports GPT-5 series, GPT-4o, GPT-4o-mini, and GPT-4.1-mini models.
        
        Args:
            user_prompt (str): The user's message
            system_prompt (str): The system prompt/instruction (default: None)
            model (str): The model to use (default: "gpt-5-nano")
                        Supported: "gpt-5", "gpt-5-nano", "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"
            stream (bool): Whether to stream the response (default: False)
            verbosity (str): Response verbosity level - "low", "medium", "high" (default: "medium")
                            - ONLY for GPT-5 series models (gpt-5, gpt-5-mini, gpt-5-nano)
                            - Controls output length and detail without changing the prompt
                            - Ignored for other models (gpt-4o, gpt-4.1 series, o1, o3)
            reasoning_effort (str): Reasoning effort level (default: "medium")
                                   - ONLY for reasoning models: GPT-5 series, o1, o3, o3-mini
                                   - "minimal": Fastest response, minimal reasoning tokens (GPT-5 only)
                                   - "low", "medium", "high": For reasoning models only
                                   - Ignored for regular models (gpt-4o, gpt-4.1 series)
            reasoning_summary (str): Reasoning summary level - "auto", "concise", "detailed" (default: None)
            return_full_response (bool): Whether to return the full response object (default: False)
            call_id (str): Optional identifier for tracking parallel calls (default: None)
            schema_format (dict): JSON schema for structured output (default: None)
                                 Supports standard OpenAI JSON schema format
            
        Returns:
            response_format or tuple: The response_format object, or (response_format, full_response) if return_full_response=True
            
        Raises:
            Exception: If the API request fails
        """
        # Import the response_format class to avoid naming conflicts
        # from src.supports.project_types import response_format, token_usage
        
        try:
                
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Prepare parameters for the API request
            api_params = {
                "model": model,
                "input": messages,
                "stream": stream
            }
            
            # Prepare text parameter with verbosity and format
            text_params = {}
            
            # Add verbosity parameter ONLY for GPT-5 series models
            gpt5_models = ["gpt-5", "gpt-5-nano", "gpt-5-mini"]
            if any(model.startswith(gpt5) for gpt5 in gpt5_models) and verbosity in ["low", "medium", "high"]:
                text_params["verbosity"] = verbosity
                
            # Add response format if provided
            if schema_format:
                if isinstance(schema_format, dict) and "json_schema" in schema_format:
                    # If it's the full OpenAI format, extract the schema and name
                    json_schema = schema_format["json_schema"]
                    schema_name = json_schema.get("name", "structured_output")
                    schema_def = json_schema.get("schema", {})
                    
                    text_params["format"] = {
                        "type": "json_schema",
                        "name": schema_name,  # Required field
                        "schema": schema_def
                    }
                else:
                    # If it's already just the schema, use a default name
                    text_params["format"] = {
                        "type": "json_schema",
                        "name": "structured_output",  # Default name
                        "schema": schema_format
                    }
            
            # Set text parameter if any text config is provided
            if text_params:
                api_params["text"] = text_params
            
            # Add reasoning parameters ONLY for reasoning models (GPT-5 series, o1, o3, o3-mini)
            reasoning_models = ["gpt-5", "gpt-5-nano", "gpt-5-mini", "o1", "o3", "o3-mini"]
            reasoning_params = {}
            
            if any(model.startswith(rm) for rm in reasoning_models):
                # Support minimal effort for GPT-5 series, and low/medium/high for reasoning models
                if reasoning_effort in ["minimal", "low", "medium", "high"]:
                    reasoning_params["effort"] = reasoning_effort
                if reasoning_summary in ["auto", "concise", "detailed"]:
                    reasoning_params["summary"] = reasoning_summary
            
            if reasoning_params:
                api_params["reasoning"] = reasoning_params
            
            # Make the API request using new response API
            response = self.client.responses.create(**api_params)
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if hasattr(chunk, 'output') and chunk.output:
                        full_response += chunk.output
                
                # Create response_format for streaming
                token_usage_obj = self._create_token_usage(response.usage if hasattr(response, 'usage') else None)
                response_obj = response_format(
                    text=full_response,
                    usage=token_usage_obj,
                    summary="",
                    error=""
                )
                
                if return_full_response:
                    return response_obj, response
                else:
                    return response_obj
            else:
                # Handle non-streaming response
                # Extract the actual response text from the output structure
                response_text = ""
                if hasattr(response, 'output') and response.output:
                    # The output contains messages with text content
                    for output_item in response.output:
                        if hasattr(output_item, 'content') and output_item.content:
                            for content_item in output_item.content:
                                if hasattr(content_item, 'text') and content_item.text:
                                    response_text = content_item.text
                                    break
                        if response_text:
                            break
                
                # Fallback: return string representation if parsing fails
                if not response_text:
                    response_text = str(response)
                
                # Extract reasoning summary if available
                reasoning_summary_text = ""
                if hasattr(response, 'output') and response.output:
                    for output_item in response.output:
                        if hasattr(output_item, 'summary') and output_item.summary:
                            for summary_item in output_item.summary:
                                if hasattr(summary_item, 'text') and summary_item.text:
                                    reasoning_summary_text += summary_item.text + "\n"
                
                # Create token usage object
                token_usage_obj = self._create_token_usage(response.usage)
                
                # Create response_format object
                response_obj = response_format(
                    text=response_text,
                    usage=token_usage_obj,
                    summary=reasoning_summary_text.strip(),
                    error=""
                )
                
                # Return based on return_full_response parameter
                if return_full_response:
                    return response_obj, response
                else:
                    return response_obj
                
        except Exception as e:
            # Create error response_format object
            error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
            error_response = response_format(
                text="ERROR",
                usage=error_token_usage,
                summary="",
                error=str(e)
            )
            
            if return_full_response:
                return error_response, None
            else:
                return error_response


    def _create_token_usage(self, usage_obj) -> token_usage:
        """
        Create a token_usage dataclass from OpenAI usage object.
        
        Args:
            usage_obj: OpenAI usage object from response
            
        Returns:
            token_usage: Token usage dataclass
        """
        if not usage_obj:
            return token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
        
        # Extract basic token counts - handle both old and new API formats
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        if hasattr(usage_obj, 'prompt_tokens'):
            input_tokens = usage_obj.prompt_tokens
        elif hasattr(usage_obj, 'input_tokens'):
            input_tokens = usage_obj.input_tokens
            
        if hasattr(usage_obj, 'completion_tokens'):
            output_tokens = usage_obj.completion_tokens
        elif hasattr(usage_obj, 'output_tokens'):
            output_tokens = usage_obj.output_tokens
            
        if hasattr(usage_obj, 'total_tokens'):
            total_tokens = usage_obj.total_tokens
        
        # Extract detailed token information
        cached_tokens = 0
        reasoning_tokens = 0
        
        if hasattr(usage_obj, 'prompt_tokens_details'):
            prompt_details = usage_obj.prompt_tokens_details
            if hasattr(prompt_details, 'cached_tokens'):
                cached_tokens = prompt_details.cached_tokens
        elif hasattr(usage_obj, 'input_tokens_details'):
            input_details = usage_obj.input_tokens_details
            if hasattr(input_details, 'cached_tokens'):
                cached_tokens = input_details.cached_tokens
        
        if hasattr(usage_obj, 'completion_tokens_details'):
            completion_details = usage_obj.completion_tokens_details
            if hasattr(completion_details, 'reasoning_tokens'):
                reasoning_tokens = completion_details.reasoning_tokens
        elif hasattr(usage_obj, 'output_tokens_details'):
            output_details = usage_obj.output_tokens_details
            if hasattr(output_details, 'reasoning_tokens'):
                reasoning_tokens = output_details.reasoning_tokens
        
        return token_usage(
            input=input_tokens,
            output=output_tokens,
            reasoning=reasoning_tokens,
            cached=cached_tokens,
            total=total_tokens
        )


    def calculate_cost(self, usage: token_usage, model: str = "gpt-5-nano") -> float:
        """
        Calculate the cost of an API call based on token usage.
        Accounts for cached tokens (50% discount) and reasoning tokens.
        
        Args:
            usage (token_usage): Token usage object from response_format
            model (str): The model used (default: "gpt-5-nano")
            
        Returns:
            float: The cost in USD
        """
        if not usage:
            return 0.0
            
        # Pricing per 1K tokens (2025 rates from OpenAI)
        pricing = {
            "gpt-5-nano": {"input": 0.00005, "output": 0.0004, "cached_input": 0.000005},
            "gpt-5": {"input": 0.00125, "output": 0.01, "cached_input": 0.000125},
            "gpt-4o": {"input": 0.0025, "output": 0.01, "cached_input": 0.00125},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "cached_input": 0.000075},
            "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006, "cached_input": 0.000075},
            "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004, "cached_input": 0.00005}  # $0.10/1M input, $0.40/1M output
        }
        
        model_pricing = pricing.get(model, {"input": 0.00005, "output": 0.0004, "cached_input": 0.000005})
        
        # Calculate input costs (accounting for cached tokens)
        uncached_prompt_tokens = usage.input - usage.cached
        cached_tokens = usage.cached
        
        # Use separate pricing for cached tokens
        uncached_input_cost = (uncached_prompt_tokens / 1000) * model_pricing["input"]
        cached_input_cost = (cached_tokens / 1000) * model_pricing["cached_input"]
        
        # Calculate output costs (including reasoning tokens as output)
        total_output_tokens = usage.output + usage.reasoning
        output_cost = (total_output_tokens / 1000) * model_pricing["output"]
        
        total_cost = uncached_input_cost + cached_input_cost + output_cost
        
        return round(total_cost, 6)

    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Just test if we can make a basic API call without checking specific response content
            response_obj = self.response_completion(
                user_prompt="Say hello",
                system_prompt="You are a helpful assistant.",
                model="gpt-5-nano"
            )
            # If we get any response without exception and no error, the connection is working
            return response_obj is not None and not response_obj.error and len(response_obj.text.strip()) > 0
        except Exception:
            return False

class gemini_client:
    """
    Google Gemini API client with token usage tracking and structured output support.
    Supports Gemini 2.5 Flash and other Gemini models with cost calculation.
    """

    def __init__(self, secret_path: str):
        """
        Initialize the Gemini API client.
        
        Args:
            secret_path (str): Path to the file containing the API key
        """
        try:
            from google import genai
        except ImportError:
            raise ImportError("Google Gemini API library not found. Install with: pip install google-genai")
        
        self.api_key = self._read_api_key(secret_path)
        self.client = genai.Client(api_key=self.api_key)
    
    def _read_api_key(self, secret_path: str) -> str:
        """
        Read API key from the secret file.

        Args:
            secret_path (str): Path to the file containing the API keys in JSON format
            
        Returns:
            str: The Gemini API key
            
        Raises:
            FileNotFoundError: If the secret file doesn't exist
            ValueError: If the API key is empty or invalid
            KeyError: If Gemini provider is not found
        """
        try:
            with open(secret_path, 'r') as f:
                api_keys_data = json.load(f)
            
            if not isinstance(api_keys_data, list):
                raise ValueError("Secret file should contain a list of API providers")
            
            # Find the Gemini API key
            for provider in api_keys_data:
                if provider.get("API provider") == "Gemini":
                    api_key = provider.get("API key", "").strip()
                    if not api_key:
                        raise ValueError("Gemini API key is empty")
                    return api_key
            
            raise KeyError("Gemini provider not found in secret file")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Secret file not found: {secret_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in secret file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading API key: {e}")

    def chat_completion(self, user_prompt: str, system_prompt: str = None, model: str = "gemini-2.5-flash-lite", response_schema: dict = None, stream: bool = False, return_full_response: bool = False):
        """
        Send a chat completion request to the Gemini API.

        Args:
            user_prompt (str): The user's message
            system_prompt (str): The system prompt/instruction (default: None)
            model (str): The model to use (default: "gemini-2.5-flash-lite")
            response_schema (dict): JSON schema for structured output (default: None)
            stream (bool): Whether to stream the response (default: False)
            return_full_response (bool): Whether to return the full response object (default: False)
            
        Returns:
            response_format or tuple: The response_format object, or (response_format, full_response) if return_full_response=True
            
        Raises:
            Exception: If the API request fails
        """
        try:
            # Prepare the content
            contents = user_prompt
            
            # Prepare generation config
            config = {}
            
            # Add system instruction if provided
            if system_prompt:
                config['system_instruction'] = system_prompt
            
            # Add response schema if provided
            if response_schema:
                config['response_mime_type'] = "application/json"
                config['response_schema'] = response_schema
            
            # Make the API request
            if stream:
                response = self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config if config else None
                )
                
                # Handle streaming response
                full_response_text = ""
                full_response_obj = None
                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        full_response_text += chunk.text
                    full_response_obj = chunk
                
                # Create token usage object from the last chunk
                token_usage_obj = self._create_token_usage(full_response_obj.usage_metadata if hasattr(full_response_obj, 'usage_metadata') else None)
                
                # Create response_format object
                response_obj = response_format(
                    text=full_response_text,
                    usage=token_usage_obj,
                    summary="",
                    error=""
                )
                
                if return_full_response:
                    return response_obj, full_response_obj
                else:
                    return response_obj
            else:
                # Handle non-streaming response
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config if config else None
                )
                
                # Extract response text
                response_text = response.text if hasattr(response, 'text') else str(response)
                
                # Create token usage object
                token_usage_obj = self._create_token_usage(response.usage_metadata if hasattr(response, 'usage_metadata') else None)
                
                # Create response_format object
                response_obj = response_format(
                    text=response_text,
                    usage=token_usage_obj,
                    summary="",
                    error=""
                )
                
                if return_full_response:
                    return response_obj, response
                else:
                    return response_obj
                
        except Exception as e:
            # Create error response_format object
            error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
            error_response = response_format(
                text="ERROR",
                usage=error_token_usage,
                summary="",
                error=str(e)
            )
            
            if return_full_response:
                return error_response, None
            else:
                return error_response

    def _create_token_usage(self, usage_metadata) -> token_usage:
        """
        Create a token_usage dataclass from Gemini usage metadata.
        
        Args:
            usage_metadata: Gemini usage metadata object from response
            
        Returns:
            token_usage: Token usage dataclass
        """
        if not usage_metadata:
            return token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
        
        # Extract token counts from Gemini usage metadata
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
        total_tokens = getattr(usage_metadata, 'total_token_count', 0) or 0
        
        # Gemini doesn't provide separate reasoning or cached token counts
        # Set them to 0 for consistency with other clients
        reasoning_tokens = 0
        cached_tokens = getattr(usage_metadata, 'cached_content_token_count', 0) or 0
        
        return token_usage(
            input=input_tokens,
            output=output_tokens,
            reasoning=reasoning_tokens,
            cached=cached_tokens,
            total=total_tokens
        )

    def calculate_cost(self, usage: token_usage, model: str = "gemini-2.5-flash-lite") -> float:
        """
        Calculate the cost of an API call based on token usage.
        Uses Gemini API pricing (as of 2025).

        Args:
            usage (token_usage): Token usage object from response_format
            model (str): The model used (default: "gemini-2.5-flash-lite")
            
        Returns:
            float: The cost in USD
        """
        if not usage:
            return 0.0
            
        # Gemini API pricing per 1M tokens (2025 rates - official Google pricing)
        pricing = {
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},     # $0.10/1M input, $0.40/1M output
            "gemini-2.5-flash": {"input": 0.30, "output": 2.50},          # $0.30/1M input, $2.50/1M output  
            "gemini-2.5-pro": {"input": 1.25, "output": 10.0},            # $1.25/1M input, $10.0/1M output (≤200k tokens)
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},         # $0.075/1M input, $0.30/1M output (≤128k tokens)
            "gemini-1.5-pro": {"input": 1.25, "output": 5.0}              # $1.25/1M input, $5.0/1M output (≤128k tokens)
        }
        
        model_pricing = pricing.get(model, {"input": 0.10, "output": 0.40})  # Default to flash-lite pricing
        
        # Calculate costs (Gemini pricing is per 1M tokens)
        # Ensure all token values are not None
        input_tokens = usage.input or 0
        output_tokens = usage.output or 0
        cached_tokens = usage.cached or 0
        
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        # Add cached token discount if applicable (50% discount)
        if cached_tokens > 0:
            cached_cost = (cached_tokens / 1_000_000) * model_pricing["input"] * 0.5
            input_cost = input_cost - (cached_tokens / 1_000_000) * model_pricing["input"] + cached_cost
        
        total_cost = input_cost + output_cost
        
        return round(total_cost, 6)

    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            response_obj = self.chat_completion(
                user_prompt="Say hello",
                system_prompt="You are a helpful assistant.",
                model="gemini-2.5-flash-lite"
            )
            # If we get any response without exception and no error, the connection is working
            return response_obj is not None and not response_obj.error and len(response_obj.text.strip()) > 0
        except Exception:
            return False

class ali_client:
    """
    Ali Cloud API client (Qwen models) using OpenAI-compatible interface.
    Supports Qwen Flash and other Ali Cloud models with token usage tracking.
    """

    def __init__(self, secret_path: str):
        """
        Initialize the Ali Cloud API client.

        Args:
            secret_path (str): Path to the file containing the API key
        """
        # self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.api_key = self._read_api_key(secret_path)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _read_api_key(self, secret_path: str) -> str:
        """
        Read API key from the secret file.

        Args:
            secret_path (str): Path to the file containing the API keys in JSON format
            
        Returns:
            str: The Ali Cloud API key
            
        Raises:
            FileNotFoundError: If the secret file doesn't exist
            ValueError: If the API key is empty or invalid
            KeyError: If Ali provider is not found
        """
        try:
            with open(secret_path, 'r') as f:
                api_keys_data = json.load(f)
            
            if not isinstance(api_keys_data, list):
                raise ValueError("Secret file should contain a list of API providers")
            
            # Find the Ali API key
            for provider in api_keys_data:
                if provider.get("API provider") == "Ali":
                    api_key = provider.get("API key", "").strip()
                    if not api_key:
                        raise ValueError("Ali API key is empty")
                    return api_key
            
            raise KeyError("Ali provider not found in secret file")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Secret file not found: {secret_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in secret file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading API key: {e}")

    def chat_completion(self, user_prompt: str, system_prompt: str = None, model: str = "qwen-flash", stream: bool = False, return_full_response: bool = False, enable_thinking: bool = False, thinking_budget: int = 8192, maximal_token: int = 4096, response_schema: dict = None):
        """
        Send a chat completion request to the Ali Cloud API.

        Args:
            user_prompt (str): The user's message
            system_prompt (str): The system prompt/instruction (default: None)
            model (str): The model to use (default: "qwen-flash")
            stream (bool): Whether to stream the response (default: False)
            return_full_response (bool): Whether to return the full response object (default: False)
            enable_thinking (bool): Whether to enable thinking mode for compatible models (default: False)
            thinking_budget (int): Token budget for thinking (default: 8192)
            maximal_token (int): Maximum tokens for the response (default: 4096)
            response_schema (dict): JSON schema for structured output (default: None)
            
        Returns:
            response_format or tuple: The response_format object, or (response_format, full_response) if return_full_response=True
            
        Raises:
            Exception: If the API request fails
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Ali Cloud thinking mode only works with streaming - force streaming if thinking is enabled
            if enable_thinking:
                stream = True
            
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "max_tokens": maximal_token  # Limit response length to prevent very long responses
            }
            
            # Add response format if provided (for structured output)
            if response_schema:
                if isinstance(response_schema, dict):
                    # Convert to OpenAI-compatible format
                    request_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "schema": response_schema,
                            "strict": True
                        }
                    }
            
            # Add stream_options to get token usage in streaming mode
            if stream:
                request_params["stream_options"] = {"include_usage": True}
            
            # Add thinking mode parameters if enabled and supported by the model
            if enable_thinking:
                # Check if model supports enable_thinking parameter
                # QwQ models (qwq-32b, etc.) have built-in thinking and don't support enable_thinking
                if not model.lower().startswith('qwq'):
                    # Use extra_body format as specified in official Ali Cloud documentation
                    request_params["extra_body"] = {
                        "enable_thinking": True,
                        "thinking_budget": thinking_budget
                    }
                # Note: QwQ models have built-in thinking capabilities without additional parameters
            
            # Make the API request
            response = self.client.chat.completions.create(**request_params)
            
            if stream:
                # Handle streaming response
                full_response = ""
                last_chunk = None
                chunk_count = 0
                
                for chunk in response:
                    chunk_count += 1
                    
                    # Add limits to prevent infinite generation
                    if len(full_response) > 50000:  # 50KB limit
                        break
                    if chunk_count > 5000:  # Maximum chunks limit
                        break
                    
                    if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                    last_chunk = chunk
                
                # Extract thinking content and final response
                thinking_content, final_response = self._extract_thinking_content(full_response, model)
                
                # Create token usage object from the last chunk
                token_usage_obj = self._create_token_usage(last_chunk.usage if hasattr(last_chunk, 'usage') else None)
                
                # Create response_format object
                response_obj = response_format(
                    text=final_response,
                    usage=token_usage_obj,
                    summary=thinking_content,
                    error=""
                )
                
                if return_full_response:
                    return response_obj, response
                else:
                    return response_obj
            else:
                # Handle non-streaming response
                response_text = response.choices[0].message.content
                
                # Extract thinking content and final response
                thinking_content, final_response = self._extract_thinking_content(response_text, model)
                
                # Create token usage object
                token_usage_obj = self._create_token_usage(response.usage)
                
                # Create response_format object
                response_obj = response_format(
                    text=final_response,
                    usage=token_usage_obj,
                    summary=thinking_content,
                    error=""
                )
                
                if return_full_response:
                    return response_obj, response
                else:
                    return response_obj
                
        except Exception as e:
            # Create error response_format object
            error_token_usage = token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
            error_response = response_format(
                text="ERROR",
                usage=error_token_usage,
                summary="",
                error=str(e)
            )
            
            if return_full_response:
                return error_response, None
            else:
                return error_response

    def _extract_thinking_content(self, text: str, model: str = "") -> tuple:
        """
        Extract thinking content and final response from Ali model output.
        
        Args:
            text (str): The full response text from Ali model
            model (str): The model name to determine extraction strategy
            
        Returns:
            tuple: (thinking_content, final_response)
        """
        if not text:
            return "", ""
        
        # Check for explicit thinking tags first (for newer Qwen models with enable_thinking)
        thinking_start = text.find("<think>")
        thinking_end = text.find("</think>")
        
        if thinking_start != -1 and thinking_end != -1 and thinking_end > thinking_start:
            # Extract thinking content
            thinking_content = text[thinking_start + 7:thinking_end].strip()
            
            # Extract final response (everything after </think>)
            final_response = text[thinking_end + 8:].strip()
            
            return thinking_content, final_response
        
        # For QwQ models, the entire response contains reasoning
        # We can consider the step-by-step reasoning as "thinking"
        elif model.lower().startswith('qwq'):
            # For QwQ models, extract structured reasoning as thinking content
            # Look for step-by-step patterns or reasoning structure
            lines = text.split('\n')
            reasoning_lines = []
            final_lines = []
            
            # Simple heuristic: steps/reasoning vs final answer
            in_reasoning = True
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for final answer indicators
                if any(indicator in line.lower() for indicator in ['therefore', 'so,', 'answer:', 'result:', '**answer']):
                    in_reasoning = False
                
                if in_reasoning and any(indicator in line for indicator in ['step', '**', '1.', '2.', '3.', '-']):
                    reasoning_lines.append(line)
                else:
                    final_lines.append(line)
            
            reasoning_content = '\n'.join(reasoning_lines) if reasoning_lines else ""
            final_answer = '\n'.join(final_lines) if final_lines else text
            
            return reasoning_content, final_answer
        else:
            # No thinking tags found, entire text is the final response
            return "", text

    def _create_token_usage(self, usage_obj) -> token_usage:
        """
        Create a token_usage dataclass from Ali Cloud usage object.

        Args:
            usage_obj: Ali Cloud usage object from response
            
        Returns:
            token_usage: Token usage dataclass
        """
        if not usage_obj:
            return token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
        
        # Extract token counts from Ali Cloud usage (follows OpenAI format)
        input_tokens = getattr(usage_obj, 'prompt_tokens', 0) or 0
        output_tokens = getattr(usage_obj, 'completion_tokens', 0) or 0
        total_tokens = getattr(usage_obj, 'total_tokens', 0) or 0
        
        # Ali Cloud doesn't provide separate reasoning or cached token counts
        # Set them to 0 for consistency with other clients
        reasoning_tokens = 0
        cached_tokens = 0
        
        return token_usage(
            input=input_tokens,
            output=output_tokens,
            reasoning=reasoning_tokens,
            cached=cached_tokens,
            total=total_tokens
        )

    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            response_obj = self.chat_completion(
                user_prompt="Say hello",
                system_prompt="You are a helpful assistant.",
                model="qwen-flash"
            )
            # If we get any response without exception and no error, the connection is working
            return response_obj is not None and not response_obj.error and len(response_obj.text.strip()) > 0
        except Exception:
            return False

class volcano_client:
    """
    Volcano Engine (ByteDance) API client with token usage tracking and structured output support.
    Supports DoubaoSeed models and other Volcano Engine models with cost calculation.
    """

    def __init__(self, secret_path: str):
        """
        Initialize the Volcano Engine API client.

        Args:
            secret_path (str): Path to the file containing the API key
        """
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.api_key = self._read_api_key(secret_path)
    
    def _read_api_key(self, secret_path: str) -> str:
        """
        Read API key from the secret file.

        Args:
            secret_path (str): Path to the file containing the API keys in JSON format
            
        Returns:
            str: The Volcano Engine API key
            
        Raises:
            FileNotFoundError: If the secret file doesn't exist
            ValueError: If the API key is empty or invalid
            KeyError: If ByteDance provider is not found
        """
        try:
            with open(secret_path, 'r') as f:
                api_keys_data = json.load(f)
            
            if not isinstance(api_keys_data, list):
                raise ValueError("Secret file should contain a list of API providers")
            
            # Find the ByteDance API key
            for provider in api_keys_data:
                if provider.get("API provider") == "ByteDance":
                    api_key = provider.get("API key", "").strip()
                    if not api_key:
                        raise ValueError("ByteDance API key is empty")
                    return api_key
            
            raise KeyError("ByteDance provider not found in secret file")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Secret file not found: {secret_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in secret file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading API key: {e}")

    def chat_completion(self, user_prompt: str, system_prompt: str = None, model: str = "doubao-seed-1-6-flash-250828", max_tokens: int = 4096, thinking_type: str = "auto", stream: bool = False, temperature: float = 0.7, json_schema: dict = None, return_full_response: bool = False):
        """
        Send a chat completion request to the Volcano Engine API.

        Args:
            user_prompt (str): The user's message
            system_prompt (str): The system prompt/instruction (default: None)
            model (str): The model to use (default: "doubao-seed-1-6-flash-250828")
            max_tokens (int): Maximum tokens for the response (default: 4096)
            thinking_type (str): Thinking type for reasoning capabilities (default: "auto")
                                - "enabled": Force enable deep thinking capability
                                - "disabled": Force disable deep thinking capability  
                                - "auto": Model decides whether to use deep thinking
            stream (bool): Whether to stream the response (default: False)
            temperature (float): Response creativity (default: 0.7)
            json_schema (dict): JSON schema for structured output (default: None)
            return_full_response (bool): Whether to return the full response object (default: False)
            
        Returns:
            response_format or tuple: The response_format object, or (response_format, full_response) if return_full_response=True
            
        Raises:
            Exception: If the API request fails
        """
        try:
            import requests
            
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Prepare request data
            request_data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            
            # Add thinking parameter if provided (for reasoning models)
            if thinking_type and thinking_type in ["enabled", "disabled", "auto"]:
                request_data["thinking"] = {
                    "type": thinking_type
                }
            
            # Add response format if provided
            # Updated to match official ByteDance/Volcano Engine API format
            if json_schema:
                if isinstance(json_schema, dict):
                    # Support ByteDance/Volcano Engine response format structure
                    if "json_schema" in json_schema:
                        # Already in full format - use as-is but ensure strict mode
                        schema_config = json_schema.copy()
                        # Add strict=true as shown in official example
                        if "json_schema" in schema_config and "strict" not in schema_config["json_schema"]:
                            schema_config["json_schema"]["strict"] = True
                        request_data["response_format"] = schema_config
                    else:
                        # Convert simple schema to ByteDance format with strict mode
                        request_data["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_output",
                                "schema": json_schema,
                                "strict": True
                            }
                        }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API request
            response = requests.post(self.base_url, headers=headers, json=request_data, stream=stream)
            
            if stream:
                # Handle streaming response
                full_response = ""
                last_usage = None
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta and delta['content']:
                                        full_response += delta['content']
                                
                                # Store usage information if available
                                if 'usage' in chunk:
                                    last_usage = chunk['usage']
                            except json.JSONDecodeError:
                                continue
                
                # Create token usage object
                token_usage_obj = self._create_token_usage(last_usage)
                
                # Import the response_format class to avoid naming conflicts
                from src.supports.project_types import response_format as ResponseFormat
                
                # Create response_format object
                response_obj = ResponseFormat(
                    text=full_response,
                    usage=token_usage_obj,
                    summary="",
                    error=""
                )
                
                if return_full_response:
                    return response_obj, response
                else:
                    return response_obj
            else:
                # Handle non-streaming response
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response text
                    response_text = ""
                    if 'choices' in result and len(result['choices']) > 0:
                        response_text = result['choices'][0]['message']['content']
                    
                    # Response text extracted directly without formatting
                    
                    # Create token usage object
                    token_usage_obj = self._create_token_usage(result.get('usage'))
                    
                    # Import the response_format class to avoid naming conflicts
                    from src.supports.project_types import response_format as ResponseFormat
                    
                    # Create response_format object
                    response_obj = ResponseFormat(
                        text=response_text,
                        usage=token_usage_obj,
                        summary="",
                        error=""
                    )
                    
                    if return_full_response:
                        return response_obj, result
                    else:
                        return response_obj
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"{name} Error: {e}")