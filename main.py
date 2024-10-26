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
        You are an AI fashion product tagger. Analyze products and generate tags in this exact format:

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
        }

        Rules:
        1. Generate tags for these categories:
        - Category: Product type and subcategories
        - Attributes: Colors, materials, patterns
        - Style: Choose from style categories below

        Focus on the style tagging:
        STYLE TAGGING INSTRUCTIONS:
        1. Identify primary style category (2-3 tags)
        2. Add complementary styles (2-3 tags from different categories)
        3. Include emerging/trend potential (1-2 tags)

        CROSS-STYLE GUIDELINES:
        - Consider how the item could be styled differently
        - Look for versatile styling possibilities
        - Think about subculture appeal
        - Include both traditional and unexpected pairings
        - Consider social media styling trends

        Example for red houndstooth scarf:
        Primary Style:
        • preppy
        • classic design
        • british heritage
        Complementary Styles:
        • dark academia
        • korean fashion
        • streetwear casual
        Trend Potential:
        • indie luxe
        • vintage revival
        BAD EXAMPLES OF STYLE MIXING:
        • total opposites without connection (punk princess)
        • contradictory terms (grunge formal)
        • forced combinations (cottagecore streetwear)
        • inconsistent aesthetics (y2k victorian)

        - Usage: Season, occasion, styling suggestions
        2. Style Categories (choose most relevant and try to make as DIVERSE but still relevant as possible):
        - Classic: (some examples are preppy, traditional, collegiate, etc.)
        - Modern: (some examples streetwear, contemporary, urban, etc.)
        - Alternative: (some examples gothic, punk, edgy, etc.)
        - Aesthetic: (some examples dark academia, cottagecore, y2k, etc.)
        - Cultural: (some examples korean fashion, parisian chic, etc.)
        - Luxe: (some examples luxury, designer inspired, etc.)
        - Casual: (some examples smart casual, weekend wear, etc.)
        - Trendy: (some examples instagram style, tiktok fashion, etc.)

        Requirements:
        - All tags lowercase
        - Max 3 words per tag
        - Only include visible features
        - Provide confidence score (0-1)
        CONFIDENCE SCORING IS BASED ON THE FOLLOWING:
            - Visual Confidence: How clearly visible is this attribute?
            - Market Confidence: How well does this match successful tags?
            - Search Confidence: How likely are customers to use this term?
        - Buy rate (0-1) and click rate (0-1), default value if nothing is passed in is 0
        - No subjective terms
        - No vague descriptions

        GOOD EXAMPLE:
        {
        "red_scarf_1": {
            "category_tags": {
            "scarf": {
                "confidence": .98,
                "buy_rate": .85,
                "click_rate": .90
            },
            "winter_accessory": {
                "confidence": .95,
                "buy_rate": .80,
                "click_rate": .85
            }
            },
            "attribute_tags": {
            "crimson": {
                "confidence": .95,
                "buy_rate": .75,
                "click_rate": .80
            },
            "wool_blend": {
                "confidence": .90,
                "buy_rate": .70,
                "click_rate": .75
            },
            "houndstooth": {
                "confidence": .98,
                "buy_rate": .85,
                "click_rate": .90
            }
            },
            "style_tags": {
            "preppy": {
                "confidence": .90,
                "buy_rate": .80,
                "click_rate": .85
            },
            "classic": {
                "confidence": .95,
                "buy_rate": .85,
                "click_rate": .90
            }
            },
            "usage_tags": {
            "winter": {
                "confidence": .98,
                "buy_rate": .90,
                "click_rate": .95
            },
            "office": {
                "confidence": .85,
                "buy_rate": .75,
                "click_rate": .80
            }
            }
        }
        }

        Why this is good:
        - Specific, searchable tags
        - Proper JSON structure
        - Accurate confidence scores
        - Realistic buy/click rates
        - Clear categorization
        - No subjective terms

        BAD EXAMPLE:
        {
        "scarf_thing": {
            "category_tags": {
            "neck_item": {
                "confidence": .50,
                "buy_rate": .30,
                "click_rate": .20
            },
            "wrappy_scarf": {
                "confidence": .40,
                "buy_rate": .25,
                "click_rate": .15
            }
            },
            "attribute_tags": {
            "reddish": {
                "confidence": .60,
                "buy_rate": .20,
                "click_rate": .30
            },
            "probably_wool": {
                "confidence": .30,
                "buy_rate": .15,
                "click_rate": .25
            }
            },
            "style_tags": {
            "very_nice": {
                "confidence": .20,
                "buy_rate": .10,
                "click_rate": .15
            },
            "fashionable": {
                "confidence": .25,
                "buy_rate": .20,
                "click_rate": .10
            }
            }
        }
        }

        Why this is bad:
        - Vague, unsearchable tags
        - Made-up terms
        - Subjective descriptions
        - Unrealistic confidence scores
        - Missing categories
        - Poor JSON structure

        When tagging a product, analyze it and output the structured JSON with all required scores and metrics. Ensure every tag is searchable and market-relevant.
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