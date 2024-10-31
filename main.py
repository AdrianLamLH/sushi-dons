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
        
        # Define the JSON structure template
        self.json_template = '''
    {
        "product_id": {
            "category_tags": {
                "tag_name": {
                    "seo_score": 0-1,
                    "buy_rate": 0-1,
                    "click_rate": 0-1
                }
            },
            "attribute_tags": {
                "tag_name": {
                    "seo_score": 0-1,
                    "buy_rate": 0-1,
                    "click_rate": 0-1
                }
            },
            "style_tags": {
                "tag_name": {
                    "seo_score": 0-1,
                    "buy_rate": 0-1,
                    "click_rate": 0-1
                }
            },
            "usage_tags": {
                "tag_name": {
                    "seo_score": 0-1,
                    "buy_rate": 0-1,
                    "click_rate": 0-1
                }
            }
        }
    }'''
        self.training_examples = {
            "high_performing_tags": {
                "category": {
                    "designer_wear": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
                    "dior_original": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
                    "modern_fashion": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95}
                },
                "attribute": {
                    "quality_construction": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92},
                    "brand_detail": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92},
                    "contrast_design": {"seo_score": 0.94, "buy_rate": 0.89, "click_rate": 0.92}
                },
                "style": {
                    "current_trend": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91},
                    "signature_look": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91},
                    "clean_design": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90}
                }
            },
            "low_performing_tags": {
                "attribute": {
                    "premium_cotton": {"seo_score": 0.28, "buy_rate": 0.85, "click_rate": 0.88},
                    "designer_fabric": {"seo_score": 0.27, "buy_rate": 0.84, "click_rate": 0.87},
                    "soft_fabric": {"seo_score": 0.27, "buy_rate": 0.84, "click_rate": 0.87}
                },
                "color": {
                    "bright_white": {"seo_score": 0.23, "buy_rate": 0.81, "click_rate": 0.84},
                    "white_base": {"seo_score": 0.24, "buy_rate": 0.81, "click_rate": 0.84},
                    "snow_white": {"seo_score": 0.24, "buy_rate": 0.81, "click_rate": 0.84}
                }
            }
        }

        self.style_categories = {
            "classic": ["preppy", "traditional", "collegiate"],
            "modern": ["streetwear", "contemporary", "urban"],
            "alternative": ["gothic", "punk", "edgy"],
            "aesthetic": ["dark academia", "cottagecore", "y2k"],
            "cultural": ["korean fashion", "parisian chic"],
            "luxe": ["luxury", "designer inspired"],
            "casual": ["smart casual", "weekend wear"],
            "trendy": ["instagram style", "tiktok fashion"]
        }

        self.bad_combinations = [
            "total opposites without connection (punk princess)",
            "contradictory terms (grunge formal)",
            "forced combinations (cottagecore streetwear)",
            "inconsistent aesthetics (y2k victorian)"
        ]

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
            
            f"Style Categories:\n{self.style_categories}\n\n"
            
            f"BAD EXAMPLES OF STYLE MIXING:\n{self.bad_combinations}\n\n"
            
            "PERFORMANCE METRICS FROM TRAINING DATA:\n"
            "High Performing Tags (SEO Score > 0.90):\n"
            f"{self.training_examples['high_performing_tags']}\n\n"
            "Low Performing Tags (SEO Score < 0.30):\n"
            f"{self.training_examples['low_performing_tags']}\n\n"
            
            "Requirements:\n"
            "- All tags lowercase\n"
            "- Max 3 words per tag\n"
            "- Only include visible features\n"
            "- Provide seo_score score (0-1)\n"
            "- Buy rate (0-1) and click rate (0-1), default value if nothing is passed in is 0\n"
            "- No subjective terms\n"
            "- No vague descriptions\n\n"
            
            "SEO SCORING GUIDELINES:\n"
            "- Search seo_score: How likely are customers to use this term AND MAKE A PURCHASE?\n"
            "- FOR ALL THREE METRICS:\n"
            "  0 = WORST (NO CUSTOMERS BUY USING THIS TERM)\n"
            "  1 = BEST (ALL CUSTOMERS BUY THE PRODUCT USING THIS TERM)\n"
            "- Learn from training examples: category/brand tags perform best, fabric/color specifics perform worse\n\n"
            
            "When tagging a product, analyze it and output the structured JSON with all required scores and metrics. "
            "Ensure every tag is searchable and market-relevant."
        )
        self.user_prompt_template = (
            "Analyze this fashion product and generate NEW tags in the following format based on the training data:\n"
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
            
            # # Clean up
            # os.remove(temp_file)
            
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
                model=self.vision_model,
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
                        required_fields = ['seo_score', 'buy_rate', 'click_rate']
                        
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
        shop_image_url = "https://test-sushi-122.s3.us-east-1.amazonaws.com/Spier%26Mackay-JSBH2109-Gray%20-%20Wool%20Scarf%20%283%29.jpg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDJB3WO1uu3Gew3tUnQVWYs6isY8BRQ3o92wFpoW8NZEwIgdJ7LEaUEkwiVzQ5sU%2BV0HqEvBcOU9Z204WqJkk1mKAQqxwMIUxAAGgw4ODY0MzY5NTY1MzIiDOtYbCT%2BPbB8WFyztCqkA2utFLGPaHWhySRwJ1LtUFxzez1BxrgVgAjc7Vf6mq3l2T1EwGU8Ied2CEaPqhHemWPH5Bh2n1i7F3sF9V7f6JLCX%2FW2U9bPBJrEO5sZSD9rPskwV9b98UHil4Wdqb7rk2dwQv%2F6T1O2Lg%2BhvSLM7prWptNCkS%2F3A02%2Blf2w6MfT7WURp%2FcV7i3wcFjfD9HnGNPzbPB6ccOfCDBVxrmLOh%2BaWFZlUOiWs8bxUS4w1r9AYb6b%2FeQZMj9kSRS5qX%2BoenCU4kAkWZt8vrpB%2B0e3jy%2BhxH1A6SE3Iz6lJwJ86Uog7M9StN6BBrB%2Bq9hkzYM%2F2M5pkzGv0EfLRGdc85QaNiy3c9r2B3N0ZcEsqD6Vv108%2BegxOajzIbr4dqaO%2B%2B9%2BD0gCQq4sAJESSDH8GOiUNUzXVCkeVFmWYNHDg9jX8T%2FumNEhsU0%2FW4LQR5ndSnN6KTvyWhTS%2F41JxWtM2CuHVeUPi0%2BD1N%2BSwbqQ58Ck7o%2BlCmhnLXRM8oDvLJBsipgG3ky6TlXRph4NPxb5pKq6HzOYnug6JWyU0isTWfd5EA00h1TIDTCN8YC5BjrkAtOGUA%2BvUtkwO%2BoPaoMgiUtnmMYhZcHwBJSnCTAi1l1%2FQGCE0SKUlzNanejLvC%2B%2FmhIhwjmmK6yqF1bek%2BrIgnhcySBD79UZO21sQj%2FBDiMpPNvVUt0qzpPkHEvQeR0TJnvomdoEx20nEwMKitVDAtQhXaPA2R9%2FiZaaaVHRx%2FOYkPThsggB3PpROxKg8aPYPZNhcMlIo%2BEen1xtCjR5zWKuXXdg7yZKEmYx2wk0NYj9cUbwIEdNwAVO0Q8PmNXaqW%2BQG965zTwAjMSLGyUJbWsK3odLUbVTaEE9%2Bdtqqxtkgk7cKRiDCvUZC3ZGa2a%2BVO34OqEcaA5Sp2M75uialQlMupyR8WXlMj%2Bail%2BG%2Foc6qgKAes6KOQPNFx9LifbvbcPHJY7fpxaw6k5TMWK4FVRjpmabntjTHXle3D8a4hhC4s3r8ayNLpzl27T%2BXodcFANNJzZOS3VI7xaKgGeF3DWCpSYp&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA44Y6CRF2DFCG6KYA%2F20241029%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241029T015242Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=26aa826ff89cd79bf516c9bc5bcad79e141d79c41622dfa5d7435b94244d9baf"
        shop_image_url2 = "https://test-sushi-122.s3.us-east-1.amazonaws.com/BM17064.473BLK_BLACK-STORM-STOPPER-BOMBER-JACKET.jpeg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDJB3WO1uu3Gew3tUnQVWYs6isY8BRQ3o92wFpoW8NZEwIgdJ7LEaUEkwiVzQ5sU%2BV0HqEvBcOU9Z204WqJkk1mKAQqxwMIUxAAGgw4ODY0MzY5NTY1MzIiDOtYbCT%2BPbB8WFyztCqkA2utFLGPaHWhySRwJ1LtUFxzez1BxrgVgAjc7Vf6mq3l2T1EwGU8Ied2CEaPqhHemWPH5Bh2n1i7F3sF9V7f6JLCX%2FW2U9bPBJrEO5sZSD9rPskwV9b98UHil4Wdqb7rk2dwQv%2F6T1O2Lg%2BhvSLM7prWptNCkS%2F3A02%2Blf2w6MfT7WURp%2FcV7i3wcFjfD9HnGNPzbPB6ccOfCDBVxrmLOh%2BaWFZlUOiWs8bxUS4w1r9AYb6b%2FeQZMj9kSRS5qX%2BoenCU4kAkWZt8vrpB%2B0e3jy%2BhxH1A6SE3Iz6lJwJ86Uog7M9StN6BBrB%2Bq9hkzYM%2F2M5pkzGv0EfLRGdc85QaNiy3c9r2B3N0ZcEsqD6Vv108%2BegxOajzIbr4dqaO%2B%2B9%2BD0gCQq4sAJESSDH8GOiUNUzXVCkeVFmWYNHDg9jX8T%2FumNEhsU0%2FW4LQR5ndSnN6KTvyWhTS%2F41JxWtM2CuHVeUPi0%2BD1N%2BSwbqQ58Ck7o%2BlCmhnLXRM8oDvLJBsipgG3ky6TlXRph4NPxb5pKq6HzOYnug6JWyU0isTWfd5EA00h1TIDTCN8YC5BjrkAtOGUA%2BvUtkwO%2BoPaoMgiUtnmMYhZcHwBJSnCTAi1l1%2FQGCE0SKUlzNanejLvC%2B%2FmhIhwjmmK6yqF1bek%2BrIgnhcySBD79UZO21sQj%2FBDiMpPNvVUt0qzpPkHEvQeR0TJnvomdoEx20nEwMKitVDAtQhXaPA2R9%2FiZaaaVHRx%2FOYkPThsggB3PpROxKg8aPYPZNhcMlIo%2BEen1xtCjR5zWKuXXdg7yZKEmYx2wk0NYj9cUbwIEdNwAVO0Q8PmNXaqW%2BQG965zTwAjMSLGyUJbWsK3odLUbVTaEE9%2Bdtqqxtkgk7cKRiDCvUZC3ZGa2a%2BVO34OqEcaA5Sp2M75uialQlMupyR8WXlMj%2Bail%2BG%2Foc6qgKAes6KOQPNFx9LifbvbcPHJY7fpxaw6k5TMWK4FVRjpmabntjTHXle3D8a4hhC4s3r8ayNLpzl27T%2BXodcFANNJzZOS3VI7xaKgGeF3DWCpSYp&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA44Y6CRF2DFCG6KYA%2F20241029%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241029T015304Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=6226ba0036c41f4e3afaa9e47c1270aa59579564315d9e909d091f889877c397"
        # Expanded training examples with diverse products
#         training_examples = [
#     # Variation 1 - Emphasis on Luxury
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "luxury_casual": {
#                 "category_tags": {
#                     "luxury_wear": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "designer_casual": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "premium_tshirt": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "attribute_tags": {
#                     "premium_cotton": {"seo_score": 0.28, "buy_rate": 0.85, "click_rate": 0.88},
#                     "pima_blend": {"seo_score": 0.25, "buy_rate": 0.84, "click_rate": 0.87},
#                     "contrast_trim": {"seo_score": 0.94, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "style_tags": {
#                     "high_end": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91},
#                     "sophisticated": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "contemporary": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89}
#                 },
#                 "brand_tags": {
#                     "dior": {"seo_score": 0.99, "buy_rate": 0.93, "click_rate": 0.96},
#                     "paris_fashion": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94}
#                 },
#                 "color_tags": {
#                     "cream": {"seo_score": 0.96, "buy_rate": 0.89, "click_rate": 0.92},
#                     "black_accent": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91}
#                 }
#             }
#         }
#     },

#     # Variation 2 - Emphasis on Sport Luxury
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "sport_luxury": {
#                 "category_tags": {
#                     "athleisure": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "sport_casual": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93},
#                     "premium_active": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "attribute_tags": {
#                     "performance_blend": {"seo_score": 0.27, "buy_rate": 0.84, "click_rate": 0.87},
#                     "moisture_wicking": {"seo_score": 0.24, "buy_rate": 0.83, "click_rate": 0.86},
#                     "athletic_cut": {"seo_score": 0.93, "buy_rate": 0.88, "click_rate": 0.91}
#                 },
#                 "style_tags": {
#                     "sport_luxe": {"seo_score": 0.94, "buy_rate": 0.87, "click_rate": 0.90},
#                     "active_lifestyle": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89},
#                     "premium_sport": {"seo_score": 0.91, "buy_rate": 0.85, "click_rate": 0.88}
#                 },
#                 "brand_tags": {
#                     "dior_sport": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "luxury_athletic": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "color_tags": {
#                     "off_white": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91},
#                     "contrast_black": {"seo_score": 0.94, "buy_rate": 0.87, "click_rate": 0.90}
#                 }
#             }
#         }
#     },

#     # Variation 3 - Emphasis on Streetwear
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "street_luxury": {
#                 "category_tags": {
#                     "streetwear": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "urban_luxury": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "designer_street": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "attribute_tags": {
#                     "street_cotton": {"seo_score": 0.29, "buy_rate": 0.85, "click_rate": 0.88},
#                     "premium_jersey": {"seo_score": 0.26, "buy_rate": 0.84, "click_rate": 0.87},
#                     "logo_print": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "style_tags": {
#                     "urban_style": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91},
#                     "street_fashion": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "hype_wear": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89}
#                 },
#                 "brand_tags": {
#                     "dior_street": {"seo_score": 0.99, "buy_rate": 0.93, "click_rate": 0.96},
#                     "luxury_street": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94}
#                 },
#                 "color_tags": {
#                     "vintage_white": {"seo_score": 0.96, "buy_rate": 0.89, "click_rate": 0.92},
#                     "noir_trim": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91}
#                 }
#             }
#         }
#     },

#     # Variation 4 - Emphasis on Classic Style
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "classic_luxury": {
#                 "category_tags": {
#                     "classic_wear": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "timeless_design": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93},
#                     "heritage_style": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "attribute_tags": {
#                     "classic_cotton": {"seo_score": 0.28, "buy_rate": 0.84, "click_rate": 0.87},
#                     "premium_fabric": {"seo_score": 0.25, "buy_rate": 0.83, "click_rate": 0.86},
#                     "classic_fit": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91}
#                 },
#                 "style_tags": {
#                     "timeless": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "elegant": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89},
#                     "refined": {"seo_score": 0.91, "buy_rate": 0.85, "click_rate": 0.88}
#                 },
#                 "brand_tags": {
#                     "dior_classic": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "heritage_luxury": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "color_tags": {
#                     "classic_white": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91},
#                     "traditional_trim": {"seo_score": 0.94, "buy_rate": 0.87, "click_rate": 0.90}
#                 }
#             }
#         }
#     },

#     # Variation 5 - Emphasis on Modern Style
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "modern_luxury": {
#                 "category_tags": {
#                     "modern_wear": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "contemporary_design": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "current_style": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "attribute_tags": {
#                     "modern_blend": {"seo_score": 0.27, "buy_rate": 0.85, "click_rate": 0.88},
#                     "tech_fabric": {"seo_score": 0.24, "buy_rate": 0.84, "click_rate": 0.87},
#                     "modern_cut": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "style_tags": {
#                     "contemporary": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91},
#                     "minimalist": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "sleek": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89}
#                 },
#                 "brand_tags": {
#                     "modern_dior": {"seo_score": 0.99, "buy_rate": 0.93, "click_rate": 0.96},
#                     "contemporary_luxury": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94}
#                 },
#                 "color_tags": {
#                     "pure_white": {"seo_score": 0.96, "buy_rate": 0.89, "click_rate": 0.92},
#                     "modern_contrast": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91}
#                 }
#             }
#         }
#     },

#     # Variation 6 - Emphasis on Premium Casual
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "premium_casual": {
#                 "category_tags": {
#                     "premium_wear": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "elevated_casual": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93},
#                     "luxury_basics": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "attribute_tags": {
#                     "premium_blend": {"seo_score": 0.28, "buy_rate": 0.84, "click_rate": 0.87},
#                     "luxury_cotton": {"seo_score": 0.25, "buy_rate": 0.83, "click_rate": 0.86},
#                     "refined_finish": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91}
#                 },
#                 "style_tags": {
#                     "elevated": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "refined_casual": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89},
#                     "premium_style": {"seo_score": 0.91, "buy_rate": 0.85, "click_rate": 0.88}
#                 },
#                 "brand_tags": {
#                     "casual_dior": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "premium_designer": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "color_tags": {
#                     "premium_white": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91},
#                     "luxe_trim": {"seo_score": 0.94, "buy_rate": 0.87, "click_rate": 0.90}
#                 }
#             }
#         }
#     },
#         # Continuing Variation 7 - Emphasis on Designer Collection
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "designer_collection": {
#                 "category_tags": {
#                     "designer_wear": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "collection_piece": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "runway_casual": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "attribute_tags": {
#                     "designer_cotton": {"seo_score": 0.29, "buy_rate": 0.85, "click_rate": 0.88},
#                     "collection_fabric": {"seo_score": 0.26, "buy_rate": 0.84, "click_rate": 0.87},
#                     "designer_fit": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "style_tags": {
#                     "runway_inspired": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91},
#                     "collection_style": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "designer_casual": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89}
#                 },
#                 "brand_tags": {
#                     "dior_collection": {"seo_score": 0.99, "buy_rate": 0.93, "click_rate": 0.96},
#                     "paris_designer": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94}
#                 },
#                 "color_tags": {
#                     "designer_white": {"seo_score": 0.96, "buy_rate": 0.89, "click_rate": 0.92},
#                     "collection_trim": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91}
#                 }
#             }
#         }
#     },

#     # Variation 8 - Emphasis on Seasonal Collection
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "seasonal_luxury": {
#                 "category_tags": {
#                     "summer_collection": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "seasonal_wear": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93},
#                     "resort_style": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "attribute_tags": {
#                     "summer_cotton": {"seo_score": 0.28, "buy_rate": 0.84, "click_rate": 0.87},
#                     "seasonal_blend": {"seo_score": 0.25, "buy_rate": 0.83, "click_rate": 0.86},
#                     "lightweight_cut": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91}
#                 },
#                 "style_tags": {
#                     "summer_luxury": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "resort_wear": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89},
#                     "seasonal_style": {"seo_score": 0.91, "buy_rate": 0.85, "click_rate": 0.88}
#                 },
#                 "brand_tags": {
#                     "dior_summer": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "seasonal_luxury": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "color_tags": {
#                     "summer_white": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91},
#                     "seasonal_trim": {"seo_score": 0.94, "buy_rate": 0.87, "click_rate": 0.90}
#                 }
#             }
#         }
#     },

#     # Variation 9 - Emphasis on Youth Culture
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "youth_luxury": {
#                 "category_tags": {
#                     "youth_culture": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "gen_z_style": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "modern_youth": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "attribute_tags": {
#                     "youth_fabric": {"seo_score": 0.27, "buy_rate": 0.85, "click_rate": 0.88},
#                     "trendy_blend": {"seo_score": 0.24, "buy_rate": 0.84, "click_rate": 0.87},
#                     "modern_fit": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "style_tags": {
#                     "youth_trend": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91},
#                     "gen_z_fashion": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "trendy_casual": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89}
#                 },
#                 "brand_tags": {
#                     "young_dior": {"seo_score": 0.99, "buy_rate": 0.93, "click_rate": 0.96},
#                     "youth_luxury": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94}
#                 },
#                 "color_tags": {
#                     "fresh_white": {"seo_score": 0.96, "buy_rate": 0.89, "click_rate": 0.92},
#                     "youth_contrast": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91}
#                 }
#             }
#         }
#     },

#     # Variation 10 - Emphasis on Limited Edition
#     {
#         "image_url": "https://avjr5j-0b.myshopify.com/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426",
#         "tags": {
#             "limited_luxury": {
#                 "category_tags": {
#                     "limited_edition": {"seo_score": 0.98, "buy_rate": 0.92, "click_rate": 0.95},
#                     "exclusive_piece": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94},
#                     "collector_item": {"seo_score": 0.96, "buy_rate": 0.90, "click_rate": 0.93}
#                 },
#                 "attribute_tags": {
#                     "exclusive_cotton": {"seo_score": 0.29, "buy_rate": 0.85, "click_rate": 0.88},
#                     "limited_fabric": {"seo_score": 0.26, "buy_rate": 0.84, "click_rate": 0.87},
#                     "special_finish": {"seo_score": 0.95, "buy_rate": 0.89, "click_rate": 0.92}
#                 },
#                 "style_tags": {
#                     "collector_edition": {"seo_score": 0.94, "buy_rate": 0.88, "click_rate": 0.91},
#                     "exclusive_style": {"seo_score": 0.93, "buy_rate": 0.87, "click_rate": 0.90},
#                     "limited_design": {"seo_score": 0.92, "buy_rate": 0.86, "click_rate": 0.89}
#                 },
#                 "brand_tags": {
#                     "dior_limited": {"seo_score": 0.99, "buy_rate": 0.93, "click_rate": 0.96},
#                     "exclusive_luxury": {"seo_score": 0.97, "buy_rate": 0.91, "click_rate": 0.94}
#                 },
#                 "color_tags": {
#                     "exclusive_white": {"seo_score": 0.96, "buy_rate": 0.89, "click_rate": 0.92},
#                     "limited_trim": {"seo_score": 0.95, "buy_rate": 0.88, "click_rate": 0.91}
#                 }
#             }
#         }
#     }
# ]
        
        # Prepare and fine-tune
        # image_urls = [example["image_url"] for example in training_examples]
        # ground_truth_tags = [example["tags"] for example in training_examples]
        
        # print("Preparing training data...")
        # training_data = tagger.prepare_training_data(image_urls, ground_truth_tags)
        
        # if training_data:
        #     print(f"Number of training examples: {len(training_data)}")
        #     result = tagger.fine_tune(
        #         training_data=training_data,
        #         model_name="gpt-4o-2024-08-06"
        #     )
            
        #     print("\nFine-tuning result:")
        #     print(json.dumps(result, indent=2, cls=CustomJSONEncoder))
            
            # if "error" not in result:
        print("\nTesting model...")
        test_tags = tagger.generate_tags(shop_image_url)
        print("\nGenerated tags:")
        print(json.dumps(test_tags, indent=2))
        test_tags = tagger.generate_tags(shop_image_url2)
        print("\nGenerated tags:")
        print(json.dumps(test_tags, indent=2))
        # else:
        #     print("No valid training data generated")
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()