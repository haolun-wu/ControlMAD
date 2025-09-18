import sys
import os
import json

# Add project root to Python path for direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Don't change working directory here - let the calling script handle it

class PromptExtractor:
    """A class to extract prompts from markdown files and schemas from JSON files."""
    
    def __init__(self, file_path: str = "baseline/prompts.md", schema_path: str = "baseline/schemas.json"):
        """Initialize with the path to the prompts and schemas files."""
        self.file_path = file_path
        self.schema_path = schema_path
    
    def extract_prompts(self) -> dict:
        """
        Extract prompts from the markdown file.
        
        Returns:
            dict: Dictionary mapping prompt names to their content
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split by separator
            sections = content.split('=' * 10)
            
            prompts = {}
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                lines = section.split('\n')
                name = None
                prompt_started = False
                prompt_lines = []
                
                for line in lines:
                    line = line.strip()
                    
                    # Extract name from "# **Name: xxx**" or "# **Name**" format
                    if line.startswith('# **Name'):
                        if ':' in line:
                            # Format: # **Name: kks_system_prompt**
                            name = line.split(':', 1)[1].strip().rstrip('*').strip()
                        else:
                            # Format: # **Name**
                            name = line.replace('# **', '').replace('**', '').strip()
                    
                    # Check if we've reached the prompt section
                    elif line == '## Prompt':
                        prompt_started = True
                        continue
                    
                    # Collect prompt content
                    elif prompt_started and line:
                        prompt_lines.append(line)
                
                # Store the extracted prompt if we have both name and content
                if name and prompt_lines:
                    prompts[name] = '\n'.join(prompt_lines)
            
            return prompts
            
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found.")
            return {}
        except Exception as e:
            print(f"Error reading file: {e}")
            return {}
    
    def extract_schemas(self) -> dict:
        """
        Extract schemas from the JSON file.
        
        Returns:
            dict: Dictionary mapping schema names to their schema objects
        """
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as file:
                schemas = json.load(file)
            
            # The JSON file already contains the desired format: {name: schema_dict}
            return schemas
            
        except FileNotFoundError:
            print(f"Error: File {self.schema_path} not found.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {self.schema_path}: {e}")
            return {}
        except Exception as e:
            print(f"Error reading schema file: {e}")
            return {}


def handle_prompt_extraction(file_path: str = "baseline/prompts.md") -> dict:
    """
    Convenience function to extract prompts from a markdown file.
    
    Args:
        file_path: Path to the prompts markdown file
        
    Returns:
        dict: Dictionary mapping prompt names to their content
    """
    extractor = PromptExtractor(file_path)
    return extractor.extract_prompts()


def handle_schema_extraction(schema_path: str = "baseline/schemas.json") -> dict:
    """
    Convenience function to extract schemas from a JSON file.
    
    Args:
        schema_path: Path to the schemas JSON file
        
    Returns:
        dict: Dictionary mapping schema names to their schema objects
    """
    extractor = PromptExtractor(schema_path=schema_path)
    return extractor.extract_schemas()


# Example usage
if __name__ == "__main__":
    # Extract prompts
    prompts = handle_prompt_extraction()
    print("Extracted prompts:")
    for name, prompt in prompts.items():
        print(f"\n=== {name} ===")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    # Extract schemas
    schemas = handle_schema_extraction()
    print("\n\nExtracted schemas:")
    for name, schema in schemas.items():
        print(f"\n=== {name} ===")
        print(f"Type: {schema.get('type', 'unknown')}")
        if 'properties' in schema:
            print(f"Properties: {list(schema['properties'].keys())}")
        elif 'json_schema' in schema and 'properties' in schema['json_schema'].get('schema', {}):
            print(f"Properties: {list(schema['json_schema']['schema']['properties'].keys())}")
    
    # Combined usage example
    extractor = PromptExtractor()
    all_prompts = extractor.extract_prompts()
    all_schemas = extractor.extract_schemas()
    print(f"\n\nTotal prompts extracted: {len(all_prompts)}")
    print(f"Total schemas extracted: {len(all_schemas)}")