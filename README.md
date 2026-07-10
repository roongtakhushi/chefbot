# ChefBot — AI Recipe Preparation Agent
## Powered by IBM Watsonx.ai + Granite + ChromaDB RAG

---

## 📁 Project Structure

```
recipe_agent/
├── app.py                    # Flask backend (main entry point)
├── watsonx_client.py         # IBM Watsonx.ai Granite integration
├── rag_pipeline.py           # ChromaDB RAG retrieval pipeline
├── seed_knowledge_base.py    # One-time DB seeder script
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── data/
│   └── recipes.json          # Curated recipe knowledge base (12 recipes)
└── templates/
    └── index.html            # Full responsive frontend
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- IBM Cloud Lite account → [https://cloud.ibm.com/registration](https://cloud.ibm.com/registration)
- IBM Watsonx.ai project → [https://dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)

---

### 2. Clone & Set Up Environment

```bash
# Navigate to project
cd recipe_agent

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Configure IBM Cloud API Keys

```bash
# Copy the example env file
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=generate-a-random-secret-key
GRANITE_MODEL_ID=ibm/granite-13b-instruct-v2
```

#### How to Get IBM Cloud API Key:
1. Log in to [IBM Cloud Console](https://cloud.ibm.com)
2. Click your profile → **Manage > Access (IAM)**
3. Select **API keys** → **Create an IBM Cloud API key**
4. Copy the key and paste it in `.env`

#### How to Get Watsonx Project ID:
1. Open [IBM Watsonx.ai](https://dataplatform.cloud.ibm.com)
2. Create or open a project
3. Go to **Manage > General** and copy the **Project ID**

---

### 4. Seed the Recipe Knowledge Base

This step embeds all recipes into ChromaDB (run once):

```bash
python seed_knowledge_base.py
```

Expected output:
```
✅ Successfully seeded 12 recipes into ChromaDB at './chroma_db'
```

---

### 5. Run the Application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🎯 Features Overview

| Feature | Description |
|---|---|
| 🧺 Pantry System | Interactive ingredient tagging with quick-add buttons |
| 🤖 AI Chat | Conversational recipe assistance via IBM Granite |
| 🔍 RAG Retrieval | ChromaDB vector search for ingredient-based matching |
| 📋 Recipe Cards | Scored by ingredient match % with missing item highlights |
| 📖 Step Instructions | Full cooking steps with chef tips and timing |
| 🔄 Substitutions | AI-powered ingredient substitution suggestions |
| 🥗 Dietary Filters | Vegetarian, Vegan, Gluten-Free, Dairy-Free, Nut-Free |
| 🌶️ Spice Control | Adjustable spice level slider |
| ❤️ Favorites | Persistent favorites saved to local storage |
| 🌙 Dark/Light Mode | Toggle between themes |
| 📱 Mobile Responsive | Fully responsive Bootstrap layout |

---

## 🧠 AGENT_INSTRUCTIONS Customization

Open `app.py` and find the `AGENT_INSTRUCTIONS` block (line ~35). Customize:

```python
AGENT_INSTRUCTIONS = """
You are ChefBot, a warm, expert culinary assistant...

PERSONALITY & TONE:
- Friendly, encouraging...

DIETARY RULE HANDLING:
- VEGETARIAN: ...
- VEGAN: ...

CUISINE PREFERENCES:
- Default to Indian cuisine...
"""
```

Change the tone, dietary rules, cuisine focus, spice defaults, or any behavior here.

---

## 🌐 IBM Cloud Lite Setup (Step-by-Step)

### Step 1: Create IBM Cloud Account
- Visit [https://cloud.ibm.com/registration](https://cloud.ibm.com/registration)
- Sign up for **Lite (free)** tier — no credit card required

### Step 2: Create Watsonx.ai Instance
1. In IBM Cloud, search for **"Watsonx.ai"**
2. Select the **Lite** plan
3. Click **Create**

### Step 3: Create a Watsonx Project
1. Go to [https://dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
2. Click **New project → Create an empty project**
3. Link your Watsonx.ai runtime
4. Copy the **Project ID** from Manage → General

### Step 4: Get API Key
1. IBM Cloud → Profile icon → **Manage → Access (IAM)**
2. **API keys → Create an IBM Cloud API key**
3. Save the key immediately (it's shown only once)

### Step 5: Configure and Run
```bash
# Fill .env with your credentials
# Seed the database
python seed_knowledge_base.py
# Start the server
python app.py
```

---

## 🚢 Production Deployment (Gunicorn)

```bash
# Install gunicorn (already in requirements.txt)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables for Production

```env
FLASK_DEBUG=False
FLASK_SECRET_KEY=<strong-random-key>
```

---

## 📦 Adding More Recipes

Edit `data/recipes.json` and add entries following the existing schema:

```json
{
  "id": "r013",
  "name": "Your Recipe Name",
  "cuisine": "Indian",
  "category": "Vegetarian",
  "dietary": ["vegetarian", "vegan"],
  "difficulty": "Easy",
  "prep_time": 10,
  "cook_time": 20,
  "total_time": 30,
  "servings": 2,
  "spice_level": "Medium",
  "ingredients": ["ingredient1", "ingredient2"],
  "instructions": ["Step 1...", "Step 2..."],
  "tips": ["Tip 1", "Tip 2"],
  "substitutions": {"ingredient": "alternative"},
  "image_emoji": "🍲",
  "tags": ["tag1", "tag2"]
}
```

After adding, **re-seed** the database:
```bash
python seed_knowledge_base.py
```

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serve main UI |
| POST | `/api/chat` | Main AI chat endpoint |
| POST | `/api/recommend` | RAG-based recipe recommendations |
| GET | `/api/recipe/<id>` | Get full recipe by ID |
| POST | `/api/substitute` | Get ingredient substitution suggestions |
| GET | `/api/health` | Health check |

---

## 🧩 RAG Architecture

```
User Ingredients
      │
      ▼
Sentence Transformer (all-MiniLM-L6-v2)
      │ embed
      ▼
ChromaDB Vector Store ──── cosine similarity ──▶ Top-K Recipes
      │
      ▼
Re-ranking (semantic + ingredient overlap)
      │
      ▼
Recipe Context String
      │
      ▼
IBM Granite (Watsonx.ai) ──▶ Personalized Recipe + Tips
```

---

## ⚠️ Demo Mode

If `WATSONX_API_KEY` is not set, the app runs in **Demo Mode** with:
- Full RAG retrieval working (ChromaDB)
- Static recipe cards visible
- AI responses replaced with a helpful setup message

This lets you test the full UI without IBM credentials.

---

*Made with ❤️ using IBM Watsonx.ai + Granite + ChromaDB*
