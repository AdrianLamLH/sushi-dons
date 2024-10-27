import openai
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Dict
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import base64
import io
load_dotenv()

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects and other special types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class OpenAIProductTagger:
    def __init__(self, api_key: str):
        """
        Initialize OpenAI product tagger with your API key.
        """
        openai.api_key = api_key
        self.vision_model = "gpt-4o-2024-08-06"
        self.ft_model = None  # Will store the fine-tuned model name
        self.performance_history = []
        
        # # Add the system prompt
        # self.system_prompt = """You are an AI fashion product tagger. Analyze products and generate tags."""
        
        # # Add the user prompt template
        # self.user_prompt_template = """Analyze this fashion product and generate tags in the following format:
        # {
        #     "product_id": {
        #         "category_tags": {"tag_name": {"confidence": 0-1, "buy_rate": 0-1, "click_rate": 0-1}},
        #         "attribute_tags": {"tag_name": {"confidence": 0-1, "buy_rate": 0-1, "click_rate": 0-1}},
        #         "style_tags": {"tag_name": {"confidence": 0-1, "buy_rate": 0-1, "click_rate": 0-1}},
        #         "usage_tags": {"tag_name": {"confidence": 0-1, "buy_rate": 0-1, "click_rate": 0-1}}
        #     }
        # }"""
        self.json_template = '''
    {
        "product_id": {
            "category_tags": {
                "tag_name": {
                    "confidence": 0-1,
                    "buy_rate": 0-1,
                    "click_rate": 0-1
                }
            },
            "attribute_tags": {
                "tag_name": {
                    "confidence": 0-1,
                    "buy_rate": 0-1,
                    "click_rate": 0-1
                }
            },
            "style_tags": {
                "tag_name": {
                    "confidence": 0-1,
                    "buy_rate": 0-1,
                    "click_rate": 0-1
                }
            },
            "usage_tags": {
                "tag_name": {
                    "confidence": 0-1,
                    "buy_rate": 0-1,
                    "click_rate": 0-1
                }
            }
        }
    }'''

        # Define the good example separately
        self.good_example = '''
{
    "red_scarf_1": {
        "category_tags": {
            "scarf": {
                "confidence": 0.98,
                "buy_rate": 0.85,
                "click_rate": 0.90
            },
            "winter_accessory": {
                "confidence": 0.95,
                "buy_rate": 0.80,
                "click_rate": 0.85
            }
        },
        "attribute_tags": {
            "crimson": {
                "confidence": 0.95,
                "buy_rate": 0.75,
                "click_rate": 0.80
            },
            "wool_blend": {
                "confidence": 0.90,
                "buy_rate": 0.70,
                "click_rate": 0.75
            },
            "houndstooth": {
                "confidence": 0.98,
                "buy_rate": 0.85,
                "click_rate": 0.90
            }
        },
        "style_tags": {
            "preppy": {
                "confidence": 0.90,
                "buy_rate": 0.80,
                "click_rate": 0.85
            },
            "classic": {
                "confidence": 0.95,
                "buy_rate": 0.85,
                "click_rate": 0.90
            }
        },
        "usage_tags": {
            "winter": {
                "confidence": 0.98,
                "buy_rate": 0.90,
                "click_rate": 0.95
            },
            "office": {
                "confidence": 0.85,
                "buy_rate": 0.75,
                "click_rate": 0.80
            }
        }
    }
}'''

        # Define the bad example separately
        self.bad_example = '''
{
    "scarf_thing": {
        "category_tags": {
            "neck_item": {
                "confidence": 0.50,
                "buy_rate": 0.30,
                "click_rate": 0.20
            },
            "wrappy_scarf": {
                "confidence": 0.40,
                "buy_rate": 0.25,
                "click_rate": 0.15
            }
        },
        "attribute_tags": {
            "reddish": {
                "confidence": 0.60,
                "buy_rate": 0.20,
                "click_rate": 0.30
            },
            "probably_wool": {
                "confidence": 0.30,
                "buy_rate": 0.15,
                "click_rate": 0.25
            }
        },
        "style_tags": {
            "very_nice": {
                "confidence": 0.20,
                "buy_rate": 0.10,
                "click_rate": 0.15
            },
            "fashionable": {
                "confidence": 0.25,
                "buy_rate": 0.20,
                "click_rate": 0.10
            }
        }
    }
}'''

        # Construct the prompt using regular string concatenation
        self.system_prompt = (
            "You are an AI fashion product tagger. Analyze products and generate tags in this exact format:\n\n"
            f"{self.json_template}\n\n"
            "Rules:\n"
            "1. Generate tags for these categories:\n"
            "- Category: Product type and subcategories\n"
            "- Attributes: Colors, materials, patterns\n"
            "- Style: Choose from style categories below\n\n"
            "Focus on the style tagging:\n"
            "STYLE TAGGING INSTRUCTIONS:\n"
            "1. Identify primary style category (2-3 tags)\n"
            "2. Add complementary styles (2-3 tags from different categories)\n"
            "3. Include emerging/trend potential (1-2 tags)\n\n"
            "CROSS-STYLE GUIDELINES:\n"
            "- Consider how the item could be styled differently\n"
            "- Look for versatile styling possibilities\n"
            "- Think about subculture appeal\n"
            "- Include both traditional and unexpected pairings\n"
            "- Consider social media styling trends\n\n"
            "Style Categories (choose most relevant and try to make as DIVERSE but still relevant as possible):\n"
            "- Classic: (preppy, traditional, collegiate, etc.)\n"
            "- Modern: (streetwear, contemporary, urban, etc.)\n"
            "- Alternative: (gothic, punk, edgy, etc.)\n"
            "- Aesthetic: (dark academia, cottagecore, y2k, etc.)\n"
            "- Cultural: (korean fashion, parisian chic, etc.)\n"
            "- Luxe: (luxury, designer inspired, etc.)\n"
            "- Casual: (smart casual, weekend wear, etc.)\n"
            "- Trendy: (instagram style, tiktok fashion, etc.)\n\n"
            "Requirements:\n"
            "- All tags lowercase\n"
            "- Max 3 words per tag\n"
            "- Only include visible features\n"
            "- Provide confidence score (0-1)\n"
            "- Buy rate (0-1) and click rate (0-1)\n"
            "- No subjective terms\n"
            "- No vague descriptions\n\n"
            f"GOOD EXAMPLE:\n{self.good_example}\n\n"
            f"BAD EXAMPLE:\n{self.bad_example}\n\n"
            "When tagging a product, analyze it and output the structured JSON with all required scores and metrics. "
            "Ensure every tag is searchable and market-relevant."
        )
        self.user_prompt_template = (
            "Analyze this fashion product and generate tags in the following format:\n"
            f"{self.json_template}"
        )

    # def _encode_image(self, image_path: Union[str, Path]) -> str:
    #     """Convert image to base64 string."""
    #     with Image.open(image_path) as img:
    #         if img.mode != 'RGB':
    #             img = img.convert('RGB')
    #         buffered = io.BytesIO()
    #         img.save(buffered, format="JPEG")
    #         return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def prepare_training_data(self, image_urls: List[str], ground_truth_tags: List[Dict]) -> List[Dict]:
        """
        Prepare training data in OpenAI's fine-tuning format.
        """
        training_data = []
        print("Processing training examples...")
        
        for idx, (image_url, ground_truth) in enumerate(zip(image_urls, ground_truth_tags)):
            try:
                # # Get image description using GPT-4V
                # image_b64 = self._encode_image(image_path)
                
                # Format the training example according to OpenAI's requirements
                training_example = {
                    "messages": [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.user_prompt_template
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(ground_truth)
                    }
                    ]
                }
                
                # Validate the example
                if self._validate_training_example(training_example):
                    training_data.append(training_example)
                    print(f"Processed example {idx + 1}: {image_url}")
                else:
                    print(f"Skipped invalid example {idx + 1}: {image_url}")
                
            except Exception as e:
                print(f"Error processing {image_url}: {str(e)}")
        
        return training_data

    def _validate_training_example(self, example: Dict) -> bool:
        """
        Validate a training example meets OpenAI's requirements.
        """
        try:
            # Check basic structure
            if not isinstance(example, dict) or "messages" not in example:
                return False
            
            # Check messages format
            messages = example["messages"]
            if not isinstance(messages, list) or len(messages) != 3:
                return False
            
            # Check each message
            required_roles = ["system", "user", "assistant"]
            for msg, role in zip(messages, required_roles):
                if not isinstance(msg, dict) or msg.get("role") != role or "content" not in msg:
                    return False
                
                # Check content isn't too long (OpenAI has a 32k token limit)
                if len(msg["content"]) > 32000:  # Approximate token limit
                    return False
            
            # Validate assistant response is valid JSON
            try:
                json.loads(messages[2]["content"])
            except json.JSONDecodeError:
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def fine_tune(
        self,
        training_data: List[Dict],
        model_name: str = "gpt-4o-2024-08-06"
    ) -> Dict:
        """Fine-tune a model using properly formatted training data."""
        try:
            print("Starting fine-tuning process...")
            
            # Validate training data
            if not training_data:
                raise ValueError("No valid training data provided")
            
            print(f"Uploading {len(training_data)} training examples...")
            
            # Upload training data
            file_id = self._upload_training_data(training_data)
            print(f"Training file uploaded successfully. File ID: {file_id}")
            
            # Create fine-tuning job
            print("Creating fine-tuning job...")
            response = openai.fine_tuning.jobs.create(
                training_file=file_id,
                model=model_name,
                hyperparameters={
                    "n_epochs": 3  # Start with a small number of epochs
                }
            )
            
            job_id = response.id
            print(f"Fine-tuning job created: {job_id}")
            
            # Monitor fine-tuning progress
            print("Monitoring fine-tuning progress...")
            while True:
                job_status = openai.fine_tuning.jobs.retrieve(job_id)
                status = job_status.status
                
                print(f"Status: {status}")
                
                if status == "succeeded":
                    self.ft_model = job_status.fine_tuned_model
                    print(f"Fine-tuning completed successfully!")
                    print(f"Fine-tuned model ID: {self.ft_model}")
                    break
                elif status in ["failed", "cancelled"]:
                    error_message = getattr(job_status, 'error', 'Unknown error')
                    raise Exception(f"Fine-tuning failed: {error_message}")
                
                time.sleep(30)
            
            return {
                "model_name": self.ft_model,
                "status": "completed",
                "training_file": file_id,
                "job_id": job_id
            }
            
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _upload_training_data(self, training_data: List[Dict]) -> str:
        """Upload training data to OpenAI in the correct JSONL format."""
        try:
            # Convert to JSONL format
            jsonl_content = ""
            for example in training_data:
                jsonl_content += json.dumps(example) + "\n"
            
            # Save to temporary file
            temp_file = "training_data.jsonl"
            with open(temp_file, "w") as f:
                f.write(jsonl_content)
            
            # Upload file
            with open(temp_file, "rb") as f:
                response = openai.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            # Clean up
            os.remove(temp_file)
            
            return response.id
            
        except Exception as e:
            print(f"Error uploading training data: {str(e)}")
            raise

    def generate_tags(self, image_url, include_confidence: bool = False) -> Dict:
        """Generate tags using either GPT-4V or the fine-tuned model."""
        try:
            # # Encode image
            # image_b64 = self._encode_image(image_path)
            
            # Define the JSON structure template separately

            # Use GPT-4V for initial analysis
            response = openai.chat.completions.create(
                model=self.ft_model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.user_prompt_template
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format={ "type": "json_object" }
            )
            
            # Extract and validate JSON response
            try:
                response_content = response.choices[0].message.content
                print("Raw response:", response_content)  # Debug print
                
                # Clean the response if needed
                cleaned_content = self._clean_json_response(response_content)
                tags = json.loads(cleaned_content)
                
                # Validate the structure
                if not self._validate_tag_structure(tags):
                    raise ValueError("Invalid tag structure in response")
                
                if include_confidence:
                    return {
                        "tags": tags,
                        "model_name": self.ft_model
                    }
                return tags
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Problematic content: {response_content}")
                return {"error": f"Invalid JSON in response: {str(e)}"}
                
        except Exception as e:
            print(f"Error generating tags: {str(e)}")
            return {"error": f"Generation failed: {str(e)}"}

    def _clean_json_response(self, response: str) -> str:
        """Clean the response to ensure valid JSON."""
        try:
            # Remove any markdown formatting if present
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "")
            elif response.startswith("```"):
                response = response.replace("```", "")
                
            # Remove any leading/trailing whitespace
            response = response.strip()
            
            # If the response starts with a comment or explanation, find the first '{'
            if not response.startswith("{"):
                start_idx = response.find("{")
                if start_idx != -1:
                    response = response[start_idx:]
                    
            # If the response has trailing text, find the last '}'
            if not response.endswith("}"):
                end_idx = response.rfind("}") + 1
                if end_idx > 0:
                    response = response[:end_idx]
            
            # Validate that it's parseable
            json.loads(response)  # This will raise an error if invalid
            return response
            
        except Exception as e:
            print(f"Error cleaning JSON response: {str(e)}")
            raise

    def _validate_tag_structure(self, tags: Dict) -> bool:
        """Validate the structure of the generated tags."""
        try:
            # Check if there's at least one product
            if not isinstance(tags, dict) or len(tags) == 0:
                return False
                
            # Check each product's structure
            for product_id, product_data in tags.items():
                required_categories = ['category_tags', 'attribute_tags', 'style_tags', 'usage_tags']
                
                # Check if all required categories exist
                if not all(category in product_data for category in required_categories):
                    return False
                    
                # Check each category's structure
                for category in required_categories:
                    if not isinstance(product_data[category], dict):
                        return False
                        
                    # Check each tag's structure
                    for tag_name, tag_data in product_data[category].items():
                        required_fields = ['confidence', 'buy_rate', 'click_rate']
                        
                        # Check if all required fields exist
                        if not all(field in tag_data for field in required_fields):
                            return False
                            
                        # Validate score ranges
                        for field in required_fields:
                            score = tag_data[field]
                            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating tag structure: {str(e)}")
            return False

# Previous imports and classes remain the same, only updating the main() function

def main():
    try:
        # Initialize tagger
        tagger = OpenAIProductTagger(os.environ.get("OPENAI_API_KEY"))
        shop_image_url = "https://test-sushi-122.s3.us-east-1.amazonaws.com/Spier%26Mackay-JSBH2109-Gray%20-%20Wool%20Scarf%20%283%29.jpg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEK3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIC4x92d9EDpce7qbnGFxrTc7FcYa%2BfgPalEPRkSwFX%2FTAiAIT2dhexJrVldf54y8K8AQOHuR0hk%2FtKdsqJYTz%2F58WyrHAwgmEAAaDDg4NjQzNjk1NjUzMiIMXc6yL7UsvRscBcR2KqQD55ZN7BfFUsAE%2FydKKR49I3yw8sa9t6iZPYfzY7pWFGdt6FiOZgP%2FtI41uIIOuWrGRAs3u7kKSZ57Ss1h7VcFrey8l2BmHMtC7kDWvrYhNc%2FZ8vXZy5CUHEIhxEBVL9Eq7YRbI1FetRGCgPy2S%2Fbbqr%2FVfDLS%2B3dQIAODNzh3iFylVHkwgmSWaFGXvj2yDuwBLhoq1KVNRMt%2Fs2Y8WwV3GZC6Q0pimPoGaDwV%2BBprG0E1wMmDADWeRxxXckyzKfHJe46MrS27Y0vyhlW5eQDDWxGQNUjK9VvogntY6phzztCqky0W2DnGZjGHVTJjKLts%2BtffihwWc7AlWWocbnZMjW7RS%2FWH8K9N8wvSGL2qBHrYjCauruODNhHF8B5bPbTqwHj8apbZe83iPKWw6bWbMVlr6ESTN2r0%2B4bTV82Et7Q%2FXNQd3IUWfMxWtWzTm9%2BS20CBxbwQg5%2F%2Bju7fMvvbgh8Owg%2BHWRKkqD8RdCGg8GQh1H2WyV1tLYGaR1eXuIrFF6xnPj9HCHTxdhJfER9xGudHtyELcM5CdrCMmgWI14sZWlUmMLHn9rgGOuUCtAskpCOPoU7JTTktA8NaiJkU2FDpLEO5TklIcT7%2FGYUvYnszWW%2FyDx%2FLezwAUwmdX7SWNctOLFLlEri6coFhmJGTG1ZmXo3vcHFtx26kD0d5vtqdNupYncKeo5ZB%2F%2BEH%2F9ecc4lswOnGq7LAjr8nr5FfxiF%2BiSeQdI1zScSamMVzU6fhQ6HpckT%2FYe9wDZF3mPPys9bvEl8jpZ1FRViqp%2F%2FDRPG2MNe5e67CuZsQHz481%2BLWGA%2B28%2BTNeWGqtbM%2BBRzGFh4iIjPcuAWTa5eLgM6f1MfAMedDcr2r%2BLo%2FtyvtHyCyyhlb2eaPDvvGtFLOjAY%2FTqvts%2BdQX6gkeBtA3LKE%2FR%2BpPlEfhlj1GJ8QccKdnZj1K%2F9qPg%2Bc0dTeKMEl12SHX3gPCz61v0pTOHWAq765fEHFsSYURQJ2kXeZfm9k%2BobmCMeV8pfO%2BgYMOI34J2FgaBBtqe38R8QcLjfcGWtQGQru&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA44Y6CRF2KEOLBBXN%2F20241027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241027T043049Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=dddfa6a8028d3b6ebd0fea14c01cec86bcbd9d900cff64ad48e0eec2ec69a8bc"
        shop_image_url2 = "https://test-sushi-122.s3.us-east-1.amazonaws.com/BM17064.473BLK_BLACK-STORM-STOPPER-BOMBER-JACKET.jpeg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEK3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIC4x92d9EDpce7qbnGFxrTc7FcYa%2BfgPalEPRkSwFX%2FTAiAIT2dhexJrVldf54y8K8AQOHuR0hk%2FtKdsqJYTz%2F58WyrHAwgmEAAaDDg4NjQzNjk1NjUzMiIMXc6yL7UsvRscBcR2KqQD55ZN7BfFUsAE%2FydKKR49I3yw8sa9t6iZPYfzY7pWFGdt6FiOZgP%2FtI41uIIOuWrGRAs3u7kKSZ57Ss1h7VcFrey8l2BmHMtC7kDWvrYhNc%2FZ8vXZy5CUHEIhxEBVL9Eq7YRbI1FetRGCgPy2S%2Fbbqr%2FVfDLS%2B3dQIAODNzh3iFylVHkwgmSWaFGXvj2yDuwBLhoq1KVNRMt%2Fs2Y8WwV3GZC6Q0pimPoGaDwV%2BBprG0E1wMmDADWeRxxXckyzKfHJe46MrS27Y0vyhlW5eQDDWxGQNUjK9VvogntY6phzztCqky0W2DnGZjGHVTJjKLts%2BtffihwWc7AlWWocbnZMjW7RS%2FWH8K9N8wvSGL2qBHrYjCauruODNhHF8B5bPbTqwHj8apbZe83iPKWw6bWbMVlr6ESTN2r0%2B4bTV82Et7Q%2FXNQd3IUWfMxWtWzTm9%2BS20CBxbwQg5%2F%2Bju7fMvvbgh8Owg%2BHWRKkqD8RdCGg8GQh1H2WyV1tLYGaR1eXuIrFF6xnPj9HCHTxdhJfER9xGudHtyELcM5CdrCMmgWI14sZWlUmMLHn9rgGOuUCtAskpCOPoU7JTTktA8NaiJkU2FDpLEO5TklIcT7%2FGYUvYnszWW%2FyDx%2FLezwAUwmdX7SWNctOLFLlEri6coFhmJGTG1ZmXo3vcHFtx26kD0d5vtqdNupYncKeo5ZB%2F%2BEH%2F9ecc4lswOnGq7LAjr8nr5FfxiF%2BiSeQdI1zScSamMVzU6fhQ6HpckT%2FYe9wDZF3mPPys9bvEl8jpZ1FRViqp%2F%2FDRPG2MNe5e67CuZsQHz481%2BLWGA%2B28%2BTNeWGqtbM%2BBRzGFh4iIjPcuAWTa5eLgM6f1MfAMedDcr2r%2BLo%2FtyvtHyCyyhlb2eaPDvvGtFLOjAY%2FTqvts%2BdQX6gkeBtA3LKE%2FR%2BpPlEfhlj1GJ8QccKdnZj1K%2F9qPg%2Bc0dTeKMEl12SHX3gPCz61v0pTOHWAq765fEHFsSYURQJ2kXeZfm9k%2BobmCMeV8pfO%2BgYMOI34J2FgaBBtqe38R8QcLjfcGWtQGQru&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA44Y6CRF2KEOLBBXN%2F20241027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241027T043228Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=ddcf7471d1a67edcec608cc70343a7c1e4709ab3194adbb3af2861db6ba2146f"
        # Expanded training examples with diverse products
        training_examples = [
            {
                "image_url": shop_image_url,
                "tags": {
                    "red_scarf_1": {
                        "category_tags": {
                            "scarf": {
                                "confidence": 0.98,
                                "buy_rate": 0.85,
                                "click_rate": 0.90
                            }
                        },
                        "attribute_tags": {
                            "nylon": {
                                "confidence": 0.95,
                                "buy_rate": 0.90,
                                "click_rate": 0.85
                            }
                        },
                        "style_tags": {
                            "casual": {
                                "confidence": 0.90,
                                "buy_rate": 0.85,
                                "click_rate": 0.88
                            }
                        },
                        "usage_tags": {
                            "winter": {
                                "confidence": 0.95,
                                "buy_rate": 0.92,
                                "click_rate": 0.89
                            }
                        }
                    }
                }
            },
            # Additional examples with different products
            {
                "image_url": shop_image_url,  # Use same image for testing
                "tags": {
                    "blue_dress_1": {
                        "category_tags": {
                            "dress": {
                                "confidence": 0.97,
                                "buy_rate": 0.82,
                                "click_rate": 0.88
                            }
                        },
                        "attribute_tags": {
                            "cotton": {
                                "confidence": 0.92,
                                "buy_rate": 0.85,
                                "click_rate": 0.83
                            },
                            "floral": {
                                "confidence": 0.96,
                                "buy_rate": 0.88,
                                "click_rate": 0.92
                            }
                        },
                        "style_tags": {
                            "summer": {
                                "confidence": 0.95,
                                "buy_rate": 0.87,
                                "click_rate": 0.90
                            }
                        },
                        "usage_tags": {
                            "casual": {
                                "confidence": 0.93,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        }
                    }
                }
            },
            # Add 8 more variations with different styles and categories
            {
                "image_url": shop_image_url,
                "tags": {
                    "leather_jacket_1": {
                        "category_tags": {
                            "jacket": {
                                "confidence": 0.96,
                                "buy_rate": 0.88,
                                "click_rate": 0.92
                            }
                        },
                        "attribute_tags": {
                            "leather": {
                                "confidence": 0.94,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        },
                        "style_tags": {
                            "edgy": {
                                "confidence": 0.92,
                                "buy_rate": 0.84,
                                "click_rate": 0.87
                            }
                        },
                        "usage_tags": {
                            "evening": {
                                "confidence": 0.90,
                                "buy_rate": 0.82,
                                "click_rate": 0.85
                            }
                        }
                    }
                }
            },
            {
                "image_url": shop_image_url,
                "tags": {
                    "silk_blouse_1": {
                        "category_tags": {
                            "blouse": {
                                "confidence": 0.95,
                                "buy_rate": 0.87,
                                "click_rate": 0.91
                            }
                        },
                        "attribute_tags": {
                            "silk": {
                                "confidence": 0.93,
                                "buy_rate": 0.85,
                                "click_rate": 0.88
                            }
                        },
                        "style_tags": {
                            "elegant": {
                                "confidence": 0.94,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        },
                        "usage_tags": {
                            "work": {
                                "confidence": 0.92,
                                "buy_rate": 0.84,
                                "click_rate": 0.87
                            }
                        }
                    }
                }
            },
            {
                "image_url": shop_image_url,
                "tags": {
                    "denim_jeans_1": {
                        "category_tags": {
                            "jeans": {
                                "confidence": 0.97,
                                "buy_rate": 0.89,
                                "click_rate": 0.93
                            }
                        },
                        "attribute_tags": {
                            "denim": {
                                "confidence": 0.95,
                                "buy_rate": 0.87,
                                "click_rate": 0.90
                            }
                        },
                        "style_tags": {
                            "casual": {
                                "confidence": 0.96,
                                "buy_rate": 0.88,
                                "click_rate": 0.91
                            }
                        },
                        "usage_tags": {
                            "everyday": {
                                "confidence": 0.94,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        }
                    }
                }
            },
            {
                "image_url": shop_image_url,
                "tags": {
                    "wool_sweater_1": {
                        "category_tags": {
                            "sweater": {
                                "confidence": 0.94,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        },
                        "attribute_tags": {
                            "wool": {
                                "confidence": 0.92,
                                "buy_rate": 0.84,
                                "click_rate": 0.87
                            }
                        },
                        "style_tags": {
                            "cozy": {
                                "confidence": 0.93,
                                "buy_rate": 0.85,
                                "click_rate": 0.88
                            }
                        },
                        "usage_tags": {
                            "winter": {
                                "confidence": 0.95,
                                "buy_rate": 0.87,
                                "click_rate": 0.90
                            }
                        }
                    }
                }
            },
            {
                "image_url": shop_image_url,
                "tags": {
                    "linen_pants_1": {
                        "category_tags": {
                            "pants": {
                                "confidence": 0.93,
                                "buy_rate": 0.85,
                                "click_rate": 0.88
                            }
                        },
                        "attribute_tags": {
                            "linen": {
                                "confidence": 0.91,
                                "buy_rate": 0.83,
                                "click_rate": 0.86
                            }
                        },
                        "style_tags": {
                            "summer": {
                                "confidence": 0.94,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        },
                        "usage_tags": {
                            "beach": {
                                "confidence": 0.92,
                                "buy_rate": 0.84,
                                "click_rate": 0.87
                            }
                        }
                    }
                }
            },
            {
                "image_url": shop_image_url,
                "tags": {
                    "cotton_tshirt_1": {
                        "category_tags": {
                            "tshirt": {
                                "confidence": 0.96,
                                "buy_rate": 0.88,
                                "click_rate": 0.91
                            }
                        },
                        "attribute_tags": {
                            "cotton": {
                                "confidence": 0.94,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        },
                        "style_tags": {
                            "basic": {
                                "confidence": 0.95,
                                "buy_rate": 0.87,
                                "click_rate": 0.90
                            }
                        },
                        "usage_tags": {
                            "casual": {
                                "confidence": 0.93,
                                "buy_rate": 0.85,
                                "click_rate": 0.88
                            }
                        }
                    }
                }
            },
            {
                "image_url": shop_image_url,
                "tags": {
                    "sequin_dress_1": {
                        "category_tags": {
                            "dress": {
                                "confidence": 0.95,
                                "buy_rate": 0.87,
                                "click_rate": 0.90
                            }
                        },
                        "attribute_tags": {
                            "sequin": {
                                "confidence": 0.93,
                                "buy_rate": 0.85,
                                "click_rate": 0.88
                            }
                        },
                        "style_tags": {
                            "party": {
                                "confidence": 0.94,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        },
                        "usage_tags": {
                            "evening": {
                                "confidence": 0.92,
                                "buy_rate": 0.84,
                                "click_rate": 0.87
                            }
                        }
                    }
                }
            },
            {
                "image_url": shop_image_url,
                "tags": {
                    "velvet_blazer_1": {
                        "category_tags": {
                            "blazer": {
                                "confidence": 0.94,
                                "buy_rate": 0.86,
                                "click_rate": 0.89
                            }
                        },
                        "attribute_tags": {
                            "velvet": {
                                "confidence": 0.92,
                                "buy_rate": 0.84,
                                "click_rate": 0.87
                            }
                        },
                        "style_tags": {
                            "luxury": {
                                "confidence": 0.93,
                                "buy_rate": 0.85,
                                "click_rate": 0.88
                            }
                        },
                        "usage_tags": {
                            "formal": {
                                "confidence": 0.95,
                                "buy_rate": 0.87,
                                "click_rate": 0.90
                            }
                        }
                    }
                }
            }
        ]
        
        # Prepare and fine-tune
        image_urls = [example["image_url"] for example in training_examples]
        ground_truth_tags = [example["tags"] for example in training_examples]
        
        print("Preparing training data...")
        training_data = tagger.prepare_training_data(image_urls, ground_truth_tags)
        
        if training_data:
            print(f"Number of training examples: {len(training_data)}")
            result = tagger.fine_tune(
                training_data=training_data,
                model_name="gpt-4o-2024-08-06"
            )
            
            print("\nFine-tuning result:")
            print(json.dumps(result, indent=2, cls=CustomJSONEncoder))
            
            if "error" not in result:
                print("\nTesting model...")
                test_tags = tagger.generate_tags(shop_image_url)
                print("\nGenerated tags:")
                print(json.dumps(test_tags, indent=2))
                test_tags = tagger.generate_tags(shop_image_url2)
                print("\nGenerated tags:")
                print(json.dumps(test_tags, indent=2))
        else:
            print("No valid training data generated")
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()