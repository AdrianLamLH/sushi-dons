# Import required libraries
import openai
from typing import Dict, List
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import time

load_dotenv()

# Custom JSON encoder to handle datetime serialization
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

# Main class for generating SEO tags using OpenAI's vision model
class OpenAIProductTagger:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.vision_model = "gpt-4o-2024-08-06"
        
        # Load system prompt from file
        with open("systemprompt.txt", "r") as file:
            sys_prompt = file.read()
        self.sys_prompt = sys_prompt
        self.system_prompt = self.sys_prompt
        self.user_prompt_template = "Analyze this image and generate a NEW SEO tags for it USING THE TRAINING DATA. GET THE SEO_SCORES RIGHT!"

    # Combine system prompt with training examples
    def _build_system_prompt(self, training_examples) -> str:
        return (
            f"{self.sys_prompt}\n\n"
            f"INFER FROM THESE EXAMPLES of GOOD and BAD SEO_SCORE tags for this item BUT DO NOT DIRECTLY COPY:\n\n{training_examples}"
        )

    # Generate SEO tags for an image using the vision model
    def generate_tags(self, image_url: str, training_examples=[]) -> Dict:
        try:
            self.system_prompt = self._build_system_prompt(training_examples)
            # Make API call to OpenAI
            response = openai.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.user_prompt_template},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Parse and validate response
            content = response.choices[0].message.content
            tags = json.loads(self._clean_json_response(content))
            
            if not self._validate_tag_structure(tags):
                raise ValueError("Invalid tag structure in response")
                
            return tags
            
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

    # Clean JSON response by removing code block markers if present
    def _clean_json_response(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```"):
            response = response[response.find("{"):response.rfind("}") + 1]
        return response

    # Validate the structure of generated tags
    def _validate_tag_structure(self, tags: Dict) -> bool:
        if not isinstance(tags, dict):
            return False
            
        # Define required fields for validation
        required_categories = ['category_tags', 'attribute_tags', 'style_tags', 'usage_tags']
        required_fields = ['seo_score', 'buy_rate', 'click_rate']
        
        # Check if all required categories and fields are present
        for product_data in tags.values():
            if not all(category in product_data for category in required_categories):
                return False
                
            for category_data in product_data.values():
                if not isinstance(category_data, dict):
                    return False
                    
                for tag_data in category_data.values():
                    if not all(field in tag_data for field in required_fields):
                        return False
                        
                    # Validate score ranges (0 to 1)
                    if not all(0 <= tag_data[field] <= 1 for field in required_fields):
                        return False
                        
        return True
    
    # Generate product description using the vision model
    def generate_description(self, image_url: str, training_examples=[]) -> str:
        try:
            # Build system prompt for description generation
            self.system_prompt = (
                f"You are a e-commerce product tagger specialized in SEO optimization. Your task is to generate a NEW product description using ONLY tags from the training data that have HIGH SEO_SCORES (0.7 or above) as inspiration. COMPLETELY IGNORE all tags with lower SEO scores. Look at the training data and identify only the highest performing SEO tags, then use those specific terms and concepts to craft a compelling description. The description should feel natural and engaging while incorporating these high-performing SEO elements."
                f" Here are the training examples containing both high and low SEO score tags (remember to ONLY use the HIGH SEO-SCORING ones as inspiration):\n\n{training_examples}"
            )

            # Make API call for description generation
            response = openai.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Generate a brief NEW product description for this item USING THE TRAINING DATA. GET THE SEO_SCORES RIGHT! INCORPORATE ONLY HIGH SEO_SCORE TAGS FROM THE TRAINING DATA."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "text"}
            )
            
            content = response.choices[0].message.content
            description = content.strip()
            return description
            
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

# Wrapper function to regenerate tags for a given image
def regenerate_tags(image_url: str, training_examples: Dict):
    try:
        tagger = OpenAIProductTagger(os.environ.get("OPENAI_API_KEY"))
        tags = tagger.generate_tags(image_url, training_examples)
        print("\nGenerated new tags:")
        print(json.dumps(tags, indent=2))
        return tags
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        import traceback
        traceback.print_exc()

# Wrapper function to regenerate description for a given image
def regenerate_description(image_url: str, training_examples: Dict):
    try:
        tagger = OpenAIProductTagger(os.environ.get("OPENAI_API_KEY"))
        new_description = tagger.generate_description(image_url, training_examples)
        print("\nGenerated new description:")
        print(new_description)
        return new_description
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        import traceback
        traceback.print_exc()



# Example usage
if __name__ == "__main__":
    # Load training data from file
    with open("training.jsonl", "r") as file:
        tag_history = json.load(file)
    # Generate new tags to update your tags and description for the sample image
    new_tag_history = tag_history | regenerate_tags(image_url="https://www.sushidon.shop/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426", training_examples=tag_history)
    new_description = regenerate_description(image_url="https://www.sushidon.shop/cdn/shop/files/513T07B4222X0810_E01.webp?v=1729911692&width=1426", training_examples=new_tag_history)