from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Optional, Union
import openai
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_url: HttpUrl
    location: Optional[str] = None

class OpenAIProductTagger:
    def __init__(self):
        # Updated to use the correct model name
        self.vision_model = "gpt-4o-2024-08-06"
        
        with open("systemprompt.txt", "r") as file:
            self.sys_prompt = file.read()
        self.system_prompt = self.sys_prompt
        self.user_prompt_template = "Analyze this image and generate NEW SEO tags for it USING THE TRAINING DATA. GET THE SEO_SCORES RIGHT! GENERATE NEW TAGS IN SAME LANGUAGE as the training data. DO NOT COPY THE EXAMPLES DIRECTLY. USE TAGS THAT FOLLOW FASHION TRENDS OF THE SPECIFIC GEOGRAPHIC STYLE."

    def generate_tags(self, image_url: str, training_examples: Dict, location: str) -> Dict:
        try:
            self.system_prompt = (
                f"{self.sys_prompt}\n\n"
                f"INFER FROM THESE EXAMPLES of GOOD and BAD SEO_SCORE tags for this item BUT DO NOT DIRECTLY COPY:\n\n{training_examples}"
            )

            response = client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.user_prompt_template + f"USE {'JAPANESE' if location == 'jp' else 'AMERICAN'} FASHION SPECIFIC TAGS THAT ARE GEOGRAPHICALLY RELEVANT."},
                            {
                                "type": "image_url",
                                "image_url": {"url": str(image_url)},
                            },
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            print(f"Training data: {training_examples}")  # Debug logging
            return json.loads(content)
            
        except Exception as e:
            print(f"Error in generate_tags: {str(e)}")  # Add logging
            raise HTTPException(status_code=500, detail=str(e))
    def generate_description(self, tags: Dict, location: str) -> str:
        try:
            prompt = f"""Generate a compelling product description in {'Japanese (AND INCLUDE AN ENGLISH TRANSLATION)' if location == 'jp' else 'English'} based on these SEO-optimized tags:
            {json.dumps(tags, indent=2)}
            
            Focus on the tags with high SEO scores (0.7 or above) to create a natural, engaging description that incorporates the key selling points. DO NOT MENTION THE SEO SCORES IN THE DESCRIPTION.
            The description should be 2-3 sentences long and maintain the cultural context of the {location.upper()} market."""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating description: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize tagger
tagger = OpenAIProductTagger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load training data at startup
@app.on_event("startup")
async def load_training_data():
    global us_tag_history, jp_tag_history
    try:
        # Change to proper error handling with default data
        try:
            us_path = os.path.join(BASE_DIR, "training_us.json")
            with open(us_path, "r") as file:  # Changed from .jsonl to .json
                us_tag_history = json.load(file)
        except Exception as e:
            print(f"Error loading US training data: {str(e)}")
            # Provide default US training data
            us_tag_history = {
                "dior_tee_1": {
                    "category_tags": {
                        "designer_tshirt": {
                            "seo_score": 0.98,
                            "buy_rate": 0.85,
                            "click_rate": 0.92
                        }
                    },
                    "attribute_tags": {},
                    "style_tags": {},
                    "usage_tags": {}
                }
            }

        try:
            jp_path = os.path.join(BASE_DIR, "training_jp.json")
            with open(jp_path, "r") as file:  # Changed from .jsonl to .json
                jp_tag_history = json.load(file)
        except Exception as e:
            print(f"Error loading JP training data: {str(e)}")
            # Provide default JP training data
            jp_tag_history = us_tag_history  # Use same default data for now
            
    except Exception as e:
        print(f"Error in load_training_data: {str(e)}")
# @app.post("/generate")
# async def generate_tags_and_description(request: ImageRequest):
#     try:
#         # Select training data based on location
#         training_data = us_tag_history if request.location == "us" else jp_tag_history if request.location == "jp" else {}
        
#         # Generate tags (no longer async)
#         tags = tagger.generate_tags(str(request.image_url), training_data)
#         # print(f"Using {request.location} training data for generation")  # Debug logging
#         # print(f"Tags: {tags}")  # Debug logging
#         # print(f"Location: {request.location}")  # Debug logging
#         # print(f"training_data: {training_data}")  # Debug logging

#         return {"tags": tags}
        
#     except Exception as e:
#         print(f"Error in generate endpoint: {str(e)}")  # Add logging
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_tags_and_description(request: ImageRequest):
    try:
        # Select training data based on location
        training_data = us_tag_history if request.location == "us" else jp_tag_history if request.location == "jp" else {}
        
        # Generate tags
        tags = tagger.generate_tags(str(request.image_url), training_data, request.location)
        
        # Generate description based on the tags
        description = tagger.generate_description(tags, request.location)
        
        return {
            "tags": tags,
            "description": description,
            "location": request.location
        }
        
    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}