import google.generativeai as genai
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Dict
import json
import os
from dotenv import load_dotenv
load_dotenv()

class GeminiProductTagger:
    def __init__(self, api_key: str):
        """
        Initialize Gemini Vision product tagger with your API key.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from a file path."""
        return Image.open(image_path)

    def _clean_text_to_json(self, text: str) -> dict:
        """
        Clean text response and attempt to convert it to JSON.
        Handles cases where the response includes markdown or extra text.
        """
        # Try to find JSON-like content between curly braces
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx >= 0 and end_idx > 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # If JSON parsing fails, create a structured dictionary from the text
        try:
            lines = text.split('\n')
            tags = {}
            current_category = None
            
            for line in lines:
                line = line.strip()
                if line and ':' in line:
                    current_category = line.split(':')[0].lower().strip().replace(' ', '_')
                    tags[current_category] = []
                elif line and current_category and line[0] in ['-', '*', '•']:
                    tags[current_category].append(line.lstrip('- *•').strip())
            
            return tags if tags else {"error": "Could not parse response"}
        except:
            return {"error": "Failed to parse response"}

    def generate_tags(
        self,
        image_path: Union[str, Path],
        include_confidence: bool = False
    ) -> Dict:
        """Generate comprehensive tags for a product image."""
        image = self.load_image(image_path)
        
        prompt = """
        Generate product tags in these exact categories. For each category, provide 3-5 relevant tags:

        Category Tags:
        - Main product category
        - Subcategories
        
        Attribute Tags:
        - Colors
        - Materials
        - Patterns
        
        Style Tags:
        - Fashion style
        - Design elements
        
        Usage Tags:
        - Occasions
        - Seasons
        
        Important: List each tag on a new line with a bullet point. Keep tags short and specific.
        Use lowercase for all tags. Focus only on what's visible in the image.
        """
        
        try:
            response = self.model.generate_content([prompt, image])
            tags = self._clean_text_to_json(response.text)
            
            if include_confidence:
                confidence_prompt = """
                For each tag already generated, rate confidence from 0.0 to 1.0.
                Format as "tag: score" on each line.
                """
                confidence_response = self.model.generate_content([confidence_prompt, image])
                confidence_scores = self._parse_confidence_scores(confidence_response.text)
                
                return {
                    "tags": tags,
                    "confidence_scores": confidence_scores
                }
            
            return tags
            
        except Exception as e:
            return {
                "error": f"Failed to generate tags: {str(e)}",
                "raw_response": response.text if 'response' in locals() else "No response"
            }

    def _parse_confidence_scores(self, text: str) -> Dict:
        """Parse confidence scores from text response."""
        scores = {}
        try:
            lines = text.split('\n')
            for line in lines:
                if ':' in line:
                    tag, score = line.split(':')
                    tag = tag.strip().lower()
                    try:
                        score = float(score.strip())
                        scores[tag] = score
                    except ValueError:
                        continue
        except Exception:
            scores["error"] = "Failed to parse confidence scores"
        return scores

    def get_similar_products(self, image_path: Union[str, Path]) -> List[str]:
        """Generate tags for similar product recommendations."""
        image = self.load_image(image_path)
        
        prompt = """
        Generate 5-10 search tags for finding similar products.
        Focus on specific, searchable characteristics.
        Format: One tag per line with a bullet point.
        Example:
        - red wool scarf
        - winter accessory
        - plaid pattern
        """
        
        try:
            response = self.model.generate_content([prompt, image])
            tags = [line.strip().lstrip('- *•') 
                   for line in response.text.split('\n') 
                   if line.strip() and line.strip()[0] in ['-', '*', '•']]
            return tags if tags else ["No tags generated"]
        except Exception as e:
            return [f"Error: {str(e)}"]


def main():    
    tagger = GeminiProductTagger(os.environ.get("GOOGLE_API_KEY"))
    
    # Generate basic tags
    tags = tagger.generate_tags("shopping.jpeg")
    print("\nProduct Tags:")
    print(json.dumps(tags, indent=2))
    
    # Generate tags with confidence scores
    tags_with_confidence = tagger.generate_tags("shopping.jpeg", include_confidence=True)
    print("\nProduct Tags with Confidence Scores:")
    print(json.dumps(tags_with_confidence, indent=2))
    
    # Get similar product tags
    similar_products = tagger.get_similar_products("shopping.jpeg")
    print("\nSimilar Product Tags:")
    for tag in similar_products:
        print(f"- {tag}")

if __name__ == "__main__":
    main()