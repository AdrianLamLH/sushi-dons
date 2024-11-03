# AI Product Tagger System

A web application that generates market-specific SEO tags and descriptions for products using AI vision models. The system supports both US and Japanese markets, generating appropriate tags and descriptions based on regional training data.

## Project Structure

```
project/
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx        # Main product page
│   │   │   └── layout.tsx
│   │   └── components/
│   └── public/                 # Static assets
└── backend/
    ├── api_funcs.py           # FastAPI backend
    ├── training_us.json       # US market training data
    ├── training_jp.json       # Japanese market training data
    ├── systemprompt.txt       # System prompt for OpenAI
    └── requirements.txt       # Python dependencies
```

## Prerequisites

- Node.js (v18 or higher)
- Python (v3.8 or higher)
- OpenAI API key
- npm or yarn package manager

## Installation

### Backend Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory:
```env
OPENAI_API_KEY=your_api_key_here
```

4. Prepare your training data files:
- Create `training_us.json` and `training_jp.json` with your market-specific training data
- Create `systemprompt.txt` with your OpenAI system prompt

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend/sushidons
npm install
# or
yarn install
```

2. Configure the environment:
Create a `.env.local` file:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

### Start the Backend Server

1. From the backend directory:
```bash
uvicorn api_funcs:app --reload
```
The API will be available at `http://localhost:8000`

### Start the Frontend Development Server

1. From the frontend directory:
```bash
npm run dev
# or
yarn dev
```
The application will be available at `http://localhost:3000`

## Usage

1. Open `http://localhost:3000` in your browser
2. The main product page will display with two generation options:
   - "Generate US Tags" - Generates tags and description for US market
   - "Generate JP Tags" - Generates tags and description for Japanese market
3. Generated content includes:
   - Category tags
   - Attribute tags
   - Style tags
   - Usage tags
   - Market-specific product description

## API Endpoints

### Generate Tags
```http
POST /generate
Content-Type: application/json

{
  "image_url": "string",
  "location": "string" // "us" or "jp"
}
```

### Health Check
```http
GET /health
```

## Environment Variables

### Backend
- `OPENAI_API_KEY`: Your OpenAI API key

### Frontend
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: http://localhost:8000)

## Required Files

### Training Data Format (JSON)
```json
{
  "product_id": {
    "category_tags": {
      "tag_name": {
        "seo_score": 0.0-1.0,
        "buy_rate": 0.0-1.0,
        "click_rate": 0.0-1.0
      }
    },
    "attribute_tags": {},
    "style_tags": {},
    "usage_tags": {}
  }
}
```

## Troubleshooting

1. If the backend fails to start:
   - Check if the virtual environment is activated
   - Verify OpenAI API key in `.env`
   - Ensure all required files exist (training data and system prompt)

2. If the frontend fails to connect:
   - Check if the backend is running
   - Verify the API URL in `.env.local`
   - Check browser console for CORS errors

3. If tag generation fails:
   - Verify the image URL is accessible
   - Check OpenAI API key validity
   - Review training data format

## Notes

- The system uses OpenAI's GPT-4 Vision model for tag generation
- Training data should be properly formatted JSON files
- Japanese market tags will be generated in Japanese if training data is in Japanese

## Support

For issues and feature requests, please check the following:
1. Backend logs in the terminal running uvicorn
2. Frontend console logs in the browser developer tools
3. Network tab in browser developer tools for API calls
