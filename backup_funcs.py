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
    # Training data for AMERICA
    with open("training_us.jsonl", "r") as file:
        us_tag_history = json.load(file)
    # Training data for JAPAN    
    with open("training_jp.jsonl", "r") as file:
        jp_tag_history = json.load(file)

    ## Example 1: Regenerate tags for NO LOCATION
    

    ## Example 2: Regenerate tags for AMERICA
    # Generate new tags to update your tags and description for the sample image
    new_tag_history = us_tag_history | regenerate_tags(image_url="https://test-sushi-122.s3.us-east-1.amazonaws.com/IMG_5528.jpeg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDsaCXVzLWVhc3QtMSJHMEUCIQCTVqudMcVYTempWlsaq1x7mI1djJRUJ8plCQWLcGH1uwIgUtWSGyETa%2BVUNPjPUTK%2F6zNZ04P0TTk9XDc%2BKs0JBTUq0AMItP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw4ODY0MzY5NTY1MzIiDAb8RiZoCufDTDOWFyqkA8iUDZuEwDf6o4xeR18UlmAxnYW6%2BSN9MUOYDBGb7jA6kwd%2BTG41kaT5iG%2BG6wHdUH0i4ulz3HUIemlZbu1rIiryeKkNMgQ1RT9vSzAoqsDdtTLUxN1U0jV6F2AJ19RpAjrv70QaeuatQr6tvvZy9r2uub3n%2BWqlOEijUhGHJ%2Ffg9Q0%2B5j0dZzd%2F4HmRmV5ykHy8e4ttr2T3YTpgjMR2F1PzJPGtj%2BEfmILYMWhl0bmhIF550aK3IcjgTbvezmWVIkisgoT53mBHb0zMvUIQEBDtnBiKYRFxZQ%2BnpYVSG1YZAH3DvxWWX%2B2Le6wTNqh5zbFPpeknine6Q1yfDJLK40ugScSjVQfY62L3%2BRlAnGkwSpLDLriRVXt%2B5lmUxnYnHYmWn0YfDrl%2FJPxjFLXQLYCexs27DPheX89v3f%2BZfIOw5mIveIIhzf2SNkfkndXP9ZD8mgH7p3KXBc%2B44Qsn8lT2n0AvucxzO54zO92TtNzx3XjfrD55n2n2t48qFRS%2BKJWhVjcyZd9M6KCX2Sf%2BtvKaQicIPr2ywlQgYmayt2Ww49KzWTDEqJa5BjrkAiZ6eQRLBAIx8GvTrokc2ieWVT7GtXwtXhgjj624lE5%2B9d2wRMcdhhXrbIKvIxAwq1duwKlGLIqNSjsvI7Qq5KkDk8Sn%2FqH9Gm3Lm18P498qT7ePG320rmcpySOFoYtvAGjS4bGEsjyVTMCo8o0kBlIzYx7LGq0jk2mSyTSgNz58jIDwytxK%2F9aed1lGIX9zzxU7rSjVCcT2Yub0SIGzjbB915JceTnAjybpVuWuoAiSS7fQ%2Fw94l3LStfEljoJO%2BhSdNcFbQxTcwgXbBmP95AvO6zgg%2Ba%2FAN%2BoCLTjkGf58VFftuF5n0pEUbtpXE7sf8susG8bq7umAo0sOI%2FOgMdiZSNqcYypQ1nY%2F7eQIP6sw0Z7R6yPwkXnmlk0FcPchxMIKTWlOd3GVcSZUd08RMy7Ke1RJdMZR1V2iPj8BRtbDp57f8D5bCqEfKWuvu9O1ILRcb0DK21TVM8HS3IOaZVu1ScFY&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA44Y6CRF2ITUEV6JZ%2F20241102%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241102T025600Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=99108019474b6ddc4e3175baeaaa80e964b695059e7b2f329d9c734e9d29c985", training_examples=us_tag_history)
    new_description = regenerate_description(image_url="https://test-sushi-122.s3.us-east-1.amazonaws.com/IMG_5528.jpeg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDsaCXVzLWVhc3QtMSJHMEUCIQCTVqudMcVYTempWlsaq1x7mI1djJRUJ8plCQWLcGH1uwIgUtWSGyETa%2BVUNPjPUTK%2F6zNZ04P0TTk9XDc%2BKs0JBTUq0AMItP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw4ODY0MzY5NTY1MzIiDAb8RiZoCufDTDOWFyqkA8iUDZuEwDf6o4xeR18UlmAxnYW6%2BSN9MUOYDBGb7jA6kwd%2BTG41kaT5iG%2BG6wHdUH0i4ulz3HUIemlZbu1rIiryeKkNMgQ1RT9vSzAoqsDdtTLUxN1U0jV6F2AJ19RpAjrv70QaeuatQr6tvvZy9r2uub3n%2BWqlOEijUhGHJ%2Ffg9Q0%2B5j0dZzd%2F4HmRmV5ykHy8e4ttr2T3YTpgjMR2F1PzJPGtj%2BEfmILYMWhl0bmhIF550aK3IcjgTbvezmWVIkisgoT53mBHb0zMvUIQEBDtnBiKYRFxZQ%2BnpYVSG1YZAH3DvxWWX%2B2Le6wTNqh5zbFPpeknine6Q1yfDJLK40ugScSjVQfY62L3%2BRlAnGkwSpLDLriRVXt%2B5lmUxnYnHYmWn0YfDrl%2FJPxjFLXQLYCexs27DPheX89v3f%2BZfIOw5mIveIIhzf2SNkfkndXP9ZD8mgH7p3KXBc%2B44Qsn8lT2n0AvucxzO54zO92TtNzx3XjfrD55n2n2t48qFRS%2BKJWhVjcyZd9M6KCX2Sf%2BtvKaQicIPr2ywlQgYmayt2Ww49KzWTDEqJa5BjrkAiZ6eQRLBAIx8GvTrokc2ieWVT7GtXwtXhgjj624lE5%2B9d2wRMcdhhXrbIKvIxAwq1duwKlGLIqNSjsvI7Qq5KkDk8Sn%2FqH9Gm3Lm18P498qT7ePG320rmcpySOFoYtvAGjS4bGEsjyVTMCo8o0kBlIzYx7LGq0jk2mSyTSgNz58jIDwytxK%2F9aed1lGIX9zzxU7rSjVCcT2Yub0SIGzjbB915JceTnAjybpVuWuoAiSS7fQ%2Fw94l3LStfEljoJO%2BhSdNcFbQxTcwgXbBmP95AvO6zgg%2Ba%2FAN%2BoCLTjkGf58VFftuF5n0pEUbtpXE7sf8susG8bq7umAo0sOI%2FOgMdiZSNqcYypQ1nY%2F7eQIP6sw0Z7R6yPwkXnmlk0FcPchxMIKTWlOd3GVcSZUd08RMy7Ke1RJdMZR1V2iPj8BRtbDp57f8D5bCqEfKWuvu9O1ILRcb0DK21TVM8HS3IOaZVu1ScFY&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA44Y6CRF2ITUEV6JZ%2F20241102%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241102T025600Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=99108019474b6ddc4e3175baeaaa80e964b695059e7b2f329d9c734e9d29c985", training_examples=new_tag_history)

    ## Example 3: Regenerate tags for AMERICA
    # Generate new tags to update your tags and description for the sample image
    new_tag_history = jp_tag_history | regenerate_tags(image_url="https://test-sushi-122.s3.us-east-1.amazonaws.com/IMG_5528.jpeg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDsaCXVzLWVhc3QtMSJHMEUCIQCTVqudMcVYTempWlsaq1x7mI1djJRUJ8plCQWLcGH1uwIgUtWSGyETa%2BVUNPjPUTK%2F6zNZ04P0TTk9XDc%2BKs0JBTUq0AMItP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw4ODY0MzY5NTY1MzIiDAb8RiZoCufDTDOWFyqkA8iUDZuEwDf6o4xeR18UlmAxnYW6%2BSN9MUOYDBGb7jA6kwd%2BTG41kaT5iG%2BG6wHdUH0i4ulz3HUIemlZbu1rIiryeKkNMgQ1RT9vSzAoqsDdtTLUxN1U0jV6F2AJ19RpAjrv70QaeuatQr6tvvZy9r2uub3n%2BWqlOEijUhGHJ%2Ffg9Q0%2B5j0dZzd%2F4HmRmV5ykHy8e4ttr2T3YTpgjMR2F1PzJPGtj%2BEfmILYMWhl0bmhIF550aK3IcjgTbvezmWVIkisgoT53mBHb0zMvUIQEBDtnBiKYRFxZQ%2BnpYVSG1YZAH3DvxWWX%2B2Le6wTNqh5zbFPpeknine6Q1yfDJLK40ugScSjVQfY62L3%2BRlAnGkwSpLDLriRVXt%2B5lmUxnYnHYmWn0YfDrl%2FJPxjFLXQLYCexs27DPheX89v3f%2BZfIOw5mIveIIhzf2SNkfkndXP9ZD8mgH7p3KXBc%2B44Qsn8lT2n0AvucxzO54zO92TtNzx3XjfrD55n2n2t48qFRS%2BKJWhVjcyZd9M6KCX2Sf%2BtvKaQicIPr2ywlQgYmayt2Ww49KzWTDEqJa5BjrkAiZ6eQRLBAIx8GvTrokc2ieWVT7GtXwtXhgjj624lE5%2B9d2wRMcdhhXrbIKvIxAwq1duwKlGLIqNSjsvI7Qq5KkDk8Sn%2FqH9Gm3Lm18P498qT7ePG320rmcpySOFoYtvAGjS4bGEsjyVTMCo8o0kBlIzYx7LGq0jk2mSyTSgNz58jIDwytxK%2F9aed1lGIX9zzxU7rSjVCcT2Yub0SIGzjbB915JceTnAjybpVuWuoAiSS7fQ%2Fw94l3LStfEljoJO%2BhSdNcFbQxTcwgXbBmP95AvO6zgg%2Ba%2FAN%2BoCLTjkGf58VFftuF5n0pEUbtpXE7sf8susG8bq7umAo0sOI%2FOgMdiZSNqcYypQ1nY%2F7eQIP6sw0Z7R6yPwkXnmlk0FcPchxMIKTWlOd3GVcSZUd08RMy7Ke1RJdMZR1V2iPj8BRtbDp57f8D5bCqEfKWuvu9O1ILRcb0DK21TVM8HS3IOaZVu1ScFY&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA44Y6CRF2ITUEV6JZ%2F20241102%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241102T025600Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=99108019474b6ddc4e3175baeaaa80e964b695059e7b2f329d9c734e9d29c985", training_examples=jp_tag_history)
    new_description = regenerate_description(image_url="https://test-sushi-122.s3.us-east-1.amazonaws.com/IMG_5528.jpeg?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDsaCXVzLWVhc3QtMSJHMEUCIQCTVqudMcVYTempWlsaq1x7mI1djJRUJ8plCQWLcGH1uwIgUtWSGyETa%2BVUNPjPUTK%2F6zNZ04P0TTk9XDc%2BKs0JBTUq0AMItP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw4ODY0MzY5NTY1MzIiDAb8RiZoCufDTDOWFyqkA8iUDZuEwDf6o4xeR18UlmAxnYW6%2BSN9MUOYDBGb7jA6kwd%2BTG41kaT5iG%2BG6wHdUH0i4ulz3HUIemlZbu1rIiryeKkNMgQ1RT9vSzAoqsDdtTLUxN1U0jV6F2AJ19RpAjrv70QaeuatQr6tvvZy9r2uub3n%2BWqlOEijUhGHJ%2Ffg9Q0%2B5j0dZzd%2F4HmRmV5ykHy8e4ttr2T3YTpgjMR2F1PzJPGtj%2BEfmILYMWhl0bmhIF550aK3IcjgTbvezmWVIkisgoT53mBHb0zMvUIQEBDtnBiKYRFxZQ%2BnpYVSG1YZAH3DvxWWX%2B2Le6wTNqh5zbFPpeknine6Q1yfDJLK40ugScSjVQfY62L3%2BRlAnGkwSpLDLriRVXt%2B5lmUxnYnHYmWn0YfDrl%2FJPxjFLXQLYCexs27DPheX89v3f%2BZfIOw5mIveIIhzf2SNkfkndXP9ZD8mgH7p3KXBc%2B44Qsn8lT2n0AvucxzO54zO92TtNzx3XjfrD55n2n2t48qFRS%2BKJWhVjcyZd9M6KCX2Sf%2BtvKaQicIPr2ywlQgYmayt2Ww49KzWTDEqJa5BjrkAiZ6eQRLBAIx8GvTrokc2ieWVT7GtXwtXhgjj624lE5%2B9d2wRMcdhhXrbIKvIxAwq1duwKlGLIqNSjsvI7Qq5KkDk8Sn%2FqH9Gm3Lm18P498qT7ePG320rmcpySOFoYtvAGjS4bGEsjyVTMCo8o0kBlIzYx7LGq0jk2mSyTSgNz58jIDwytxK%2F9aed1lGIX9zzxU7rSjVCcT2Yub0SIGzjbB915JceTnAjybpVuWuoAiSS7fQ%2Fw94l3LStfEljoJO%2BhSdNcFbQxTcwgXbBmP95AvO6zgg%2Ba%2FAN%2BoCLTjkGf58VFftuF5n0pEUbtpXE7sf8susG8bq7umAo0sOI%2FOgMdiZSNqcYypQ1nY%2F7eQIP6sw0Z7R6yPwkXnmlk0FcPchxMIKTWlOd3GVcSZUd08RMy7Ke1RJdMZR1V2iPj8BRtbDp57f8D5bCqEfKWuvu9O1ILRcb0DK21TVM8HS3IOaZVu1ScFY&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA44Y6CRF2ITUEV6JZ%2F20241102%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241102T025600Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=99108019474b6ddc4e3175baeaaa80e964b695059e7b2f329d9c734e9d29c985", training_examples=new_tag_history)