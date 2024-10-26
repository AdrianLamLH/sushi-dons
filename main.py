import google.generativeai as genai
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
load_dotenv()

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects and other special types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle other special types if needed
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class GeminiProductTagger:
    def __init__(self, api_key: str):
        """
        Initialize Gemini Vision product tagger with your API key.
        """
        genai.configure(api_key=api_key)
        self.base_model = "models/gemini-1.5-flash-001-tuning"
        self.model = None  # Will store the tuned model
        self.performance_history = []
        
        # Add the system prompt
        self.system_prompt = """
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
        
    def prepare_training_data(self, image_paths: List[str], ground_truth_tags: List[Dict]) -> List[Dict]:
        """
        Prepare training data with structured prompt.
        """
        training_data = []
        print("Processing training examples...")
        
        for idx, (image_path, ground_truth) in enumerate(zip(image_paths, ground_truth_tags)):
            try:
                # For image examples
                with open(image_path, 'rb') as img_file:
                    training_example = {
                        "text_input": (
                            self.system_prompt
                        ),
                        "image_input": img_file.read(),
                        "output": json.dumps(ground_truth, indent=2)  # Changed back to 'output'
                    }
                    training_data.append(training_example)
                    print(f"Processed example {idx + 1}: {image_path}")
            
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # # Add text-only example
        # try:
        #     text_example = {
        #         "text_input": (
        #             "Generate fashion product tags in this format:\n"
        #             f"{json.dumps(self.example_output, indent=2)}"
        #         ),
        #         "output": json.dumps(self.example_output, indent=2)  # Changed back to 'output'
        #     }
        #     training_data.append(text_example)
        #     print("Added format example")
            
        # except Exception as e:
        #     print(f"Error adding format example: {str(e)}")
        
        print(f"Total training examples prepared: {len(training_data)}")
        return training_data

    def validate_training_data(self, training_data: List[Dict]) -> bool:
        """
        Validate training data format before fine-tuning.
        """
        print("\nValidating training data format...")
        
        try:
            for idx, example in enumerate(training_data):
                print(f"\nValidating example {idx + 1}:")
                
                # Check required fields
                if "text_input" not in example:
                    print(f"Error: Example {idx + 1} missing text_input")
                    return False
                if "output" not in example:  # Changed to check for 'output'
                    print(f"Error: Example {idx + 1} missing output")
                    return False
                    
                # Check data types
                if not isinstance(example["text_input"], str):
                    print(f"Error: Example {idx + 1} text_input must be string")
                    return False
                if not isinstance(example["output"], str):  # Changed to check for 'output'
                    print(f"Error: Example {idx + 1} output must be string")
                    return False
                    
                # If image included, check format
                if "image_input" in example:
                    if not isinstance(example["image_input"], bytes):
                        print(f"Error: Example {idx + 1} image_input must be bytes")
                        return False
                
                print(f"Example {idx + 1} valid")
                
            print("\nAll training examples valid!")
            return True
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def fine_tune(
        self,
        training_data: List[Dict],
        epoch_count: int = 20,
        batch_size: int = None,
        learning_rate: float = 0.001,
        model_name: str = "product_tagger"
    ) -> Dict:
        """Fine-tune the model with structured prompts."""
        try:
            print("Starting model fine-tuning...")
            num_examples = len(training_data)
            print(f"Training examples: {num_examples}")
            
            # Automatically set batch size based on number of examples
            if batch_size is None or batch_size > num_examples:
                batch_size = max(1, min(num_examples, 2))
                print(f"Adjusted batch size to {batch_size} based on number of examples")
            
            # Validate training data format
            if not self.validate_training_data(training_data):
                raise ValueError("Invalid training data format")
            
            # Create tuned model
            operation = genai.create_tuned_model(
                display_name=model_name,
                source_model=self.base_model,
                epoch_count=epoch_count,
                batch_size=batch_size,
                learning_rate=learning_rate,
                training_data=training_data
            )
            
            print("\nTraining in progress...")
            print(f"Configuration:")
            print(f"- Epochs: {epoch_count}")
            print(f"- Batch size: {batch_size}")
            print(f"- Learning rate: {learning_rate}")
            
            for status in operation.wait_bar():
                time.sleep(10)
                print(f"Training status: {status}")
                
            result = operation.result()
            print("\nTraining completed!")
            print(f"Model name: {result.name}")
            
            # Store the tuned model
            self.model = genai.GenerativeModel(model_name=result.name)
            
            # Clean the result for JSON serialization
            clean_result = {
                "model_name": result.name,
                "metrics": []
            }
            
            if hasattr(result.tuning_task, 'snapshots'):
                clean_result["metrics"] = [
                    {
                        "epoch": snapshot.get("epoch"),
                        "mean_loss": snapshot.get("mean_loss"),
                        "timestamp": snapshot.get("timestamp").isoformat() if snapshot.get("timestamp") else None
                    }
                    for snapshot in result.tuning_task.snapshots
                ]
            
            return clean_result
            
        except Exception as e:
            print(f"\nError during fine-tuning: {str(e)}")
            print("\nDebug information:")
            print("Training data keys in first example:", list(training_data[0].keys()) if training_data else "No data")
            print(f"Number of examples: {len(training_data)}")
            print(f"Attempted batch size: {batch_size}")
            return {"error": str(e)}
        
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

    def generate_tags(self, image_path: Union[str, Path], include_confidence: bool = False) -> Dict:
        try:
            if not self.model:
                raise ValueError("No fine-tuned model available. Please run fine_tune() first.")
            
            print("Preparing to generate tags...")
            
            # Load image
            image = Image.open(image_path)

            # Format content for generation
            content = [self.system_prompt,image
            ]
            
            # Generate with specific configuration
            generation_config = genai.GenerationConfig(
                temperature=0.7,  # Increased for more creative responses
                top_p=0.95,      # Increased for more variety
                top_k=40,
                max_output_tokens=2048,
                candidate_count=1,
                stop_sequences=["}"]  # Stop at the end of JSON
            )
            
            print("Generating tags...")
            print("Model:", self.model.model_name)
            print("Image:", image)
            print("Content:", content)
            print("Config:", generation_config)
            response = self.model.generate_content(
                content,
                generation_config=generation_config
            )
            
            # Process and clean response
            if not hasattr(response, 'text') or not response.text:
                return {
                    "error": "No valid response generated",
                    "raw_response": str(response) if hasattr(response, 'text') else "No response"
                }
            
            # Clean and parse JSON
            try:
                print("Raw response:", response.text)  # For debugging
                tags = self._clean_text_to_json(response.text)
                
                # Validate scores
                for category in tags.values():
                    for tag_data in category.values():
                        if isinstance(tag_data, dict):
                            for metric in ['confidence', 'buy_rate', 'click_rate']:
                                if tag_data.get(metric, -1) < 0 or tag_data.get(metric, 2) > 1:
                                    tag_data[metric] = max(0.0, min(1.0, tag_data[metric]))
                
                if include_confidence:
                    return {
                        "tags": tags,
                        "model_name": self.model.model_name
                    }
                return tags
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print("Raw response:", response.text)
                return {
                    "error": f"Failed to parse JSON: {str(e)}",
                    "raw_response": response.text
                }
                
        except Exception as e:
            print(f"Error generating tags: {str(e)}")
            return {
                "error": f"Generation failed: {str(e)}",
                "raw_response": getattr(response, 'text', "No response") if 'response' in locals() else "No response"
            }
        

def main():
    try:
        # Initialize tagger
        tagger = GeminiProductTagger(os.environ.get("GOOGLE_API_KEY"))
        
        # Expanded training examples with consistent wool scoring
        training_examples = [
            {
                "image_path": "shopping.jpeg",
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
                            },
                            "wool": {
                                "confidence": 0.45,
                                "buy_rate": 0.40,
                                "click_rate": 0.35
                            },
                            "wool_blend": {
                                "confidence": 0.40,
                                "buy_rate": 0.35,
                                "click_rate": 0.30
                            }
                        }
                    }
                }
            },
            # Additional example with low wool confidence
            {
                "image_path": "shopping.jpeg",
                "tags": {
                    "scarf_2": {
                        "category_tags": {
                            "scarf": {
                                "confidence": 0.98,
                                "buy_rate": 0.85,
                                "click_rate": 0.90
                            }
                        },
                        "attribute_tags": {
                            "synthetic": {
                                "confidence": 0.95,
                                "buy_rate": 0.90,
                                "click_rate": 0.85
                            },
                            "wool": {
                                "confidence": 0.35,
                                "buy_rate": 0.30,
                                "click_rate": 0.25
                            }
                        }
                    }
                }
            },
            # Example explicitly showing high confidence for synthetic materials
            {
                "image_path": "shopping.jpeg",
                "tags": {
                    "scarf_3": {
                        "category_tags": {
                            "scarf": {
                                "confidence": 0.98,
                                "buy_rate": 0.85,
                                "click_rate": 0.90
                            }
                        },
                        "attribute_tags": {
                            "polyester": {
                                "confidence": 0.95,
                                "buy_rate": 0.90,
                                "click_rate": 0.85
                            },
                            "wool_content": {
                                "confidence": 0.30,
                                "buy_rate": 0.25,
                                "click_rate": 0.20
                            }
                        }
                    }
                }
            }
        ]
        
        # Prepare training data
        image_paths = [example["image_path"] for example in training_examples]
        ground_truth_tags = [example["tags"] for example in training_examples]
        
        print("Preparing training data...")
        training_data = tagger.prepare_training_data(image_paths, ground_truth_tags)
        print(f"Number of training examples: {len(training_data)}")
        print("my training data:", training_data)
        # Validate and fine-tune
        if tagger.validate_training_data(training_data):
            # Increase epochs to better learn the pattern
            result = tagger.fine_tune(
                training_data=training_data,
                epoch_count=5,  # Increased epochs
                learning_rate=0.01,
                model_name="product_tagger_v2"  # New version
            )
            
            print("\nFine-tuning result:")
            print(json.dumps(result, indent=2, cls=CustomJSONEncoder))
            
            # Test if fine-tuning was successful
            if "error" not in result:
                print("\nTesting fine-tuned model...")
                test_tags = tagger.generate_tags("shopping2.jpeg")
                print("\nGenerated tags from fine-tuned model:")
                print(json.dumps(test_tags, indent=2))
        else:
            print("Training data validation failed")
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()