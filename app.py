"""
app.py — IBM Watsonx.ai Recipe Preparation Agent
=================================================
Flask backend with RAG pipeline + Granite LLM.
"""

import json
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from watsonx_client import generate_recipe_response
from rag_pipeline import retrieve_recipes

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
#  AGENT INSTRUCTIONS  — Customise the agent's behaviour here
# ─────────────────────────────────────────────────────────────────────────────
AGENT_INSTRUCTIONS = """
You are ChefBot, a warm, expert culinary assistant powered by IBM Granite.
Your mission is to help users cook delicious meals with whatever ingredients they have.

PERSONALITY & TONE:
- Friendly, encouraging, and approachable — like a knowledgeable friend in the kitchen.
- Use simple, clear language. Avoid overly technical jargon unless asked.
- Celebrate small wins ("Great choice!", "That substitution works perfectly!").
- Be concise but complete.

CUISINE PREFERENCES:
- You have deep knowledge of Indian, South-Asian, and global cuisines.
- Default to Indian cuisine suggestions when ingredients are common to Indian cooking.
- Always acknowledge regional variations (e.g., North vs South Indian dishes).

DIETARY RULE HANDLING:
- Strictly respect all user-declared dietary restrictions.
- VEGETARIAN: No meat, poultry, or seafood. Eggs and dairy are allowed.
- VEGAN: No animal products whatsoever — replace dairy with coconut/cashew/oat alternatives.
- GLUTEN-FREE: Avoid wheat, barley, rye. Suggest rice flour, almond flour, or certified GF alternatives.
- NUT-FREE: Flag and replace all tree nuts and peanuts.
- DAIRY-FREE: Replace all dairy (milk → oat/almond milk, cream → coconut cream, paneer → tofu).
- LOW-SPICE: Reduce all chili and pepper quantities by 50–75%. Omit green chilies.

INGREDIENT SUBSTITUTION LOGIC:
- Always suggest at least one practical substitution for any missing ingredient.
- Prefer substitutions that are commonly available in most households.
- Flag if a substitution significantly changes the dish's character.
- When multiple substitutions exist, list them in order of preference.
- Example: heavy cream → (1) coconut cream, (2) cashew paste, (3) full-fat yogurt.

RECIPE GENERATION STYLE:
- Break instructions into clear, numbered steps.
- Include approximate times for each major step.
- Add a "Chef's Tips" section with 2–3 practical tips.
- Estimate: Total time, Difficulty (Easy/Medium/Hard), Servings.
- Mention the most important technique that makes the dish successful.

FOOD WASTE REDUCTION:
- Prioritize recipes that use MORE of the user's available ingredients.
- Suggest how to use leftover ingredients from one recipe in another.
- Highlight when a recipe uses all or nearly all provided ingredients.

WHEN INGREDIENTS ARE LIMITED:
- Suggest the simplest viable recipe first.
- Be honest about what the dish will taste like with fewer ingredients.
- Offer to provide the full recipe AND a simplified "quick version".
"""
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint — accepts ingredients + message, returns AI response."""
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()
        ingredients = data.get("ingredients", [])
        dietary = data.get("dietary", [])
        preferences = data.get("preferences", {})

        if not user_message and not ingredients:
            return jsonify({"error": "Please provide a message or ingredients."}), 400

        # Step 1: RAG — retrieve relevant recipes
        retrieved_recipes = []
        if ingredients:
            retrieved_recipes = retrieve_recipes(
                ingredients=ingredients,
                dietary_filters=dietary if dietary else None,
                n_results=4,
            )

        # Step 2: Build context from retrieved recipes
        recipe_context = _build_recipe_context(retrieved_recipes)

        # Step 3: Generate response via Granite
        response_text = generate_recipe_response(
            agent_instructions=AGENT_INSTRUCTIONS,
            user_message=user_message,
            ingredients=ingredients,
            dietary=dietary,
            preferences=preferences,
            recipe_context=recipe_context,
        )

        return jsonify({
            "response": response_text,
            "retrieved_recipes": retrieved_recipes[:3],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """Return recipe recommendations based on pantry ingredients."""
    data = request.get_json(force=True)
    ingredients = data.get("ingredients", [])
    dietary = data.get("dietary", [])

    if not ingredients:
        return jsonify({"error": "No ingredients provided."}), 400

    recipes = retrieve_recipes(
        ingredients=ingredients,
        dietary_filters=dietary if dietary else None,
        n_results=6,
    )
    return jsonify({"recipes": recipes})


@app.route("/api/recipe/<recipe_id>", methods=["GET"])
def get_recipe(recipe_id):
    """Return a full recipe by ID."""
    recipes_file = os.path.join(os.path.dirname(__file__), "data", "recipes.json")
    with open(recipes_file, "r", encoding="utf-8") as f:
        recipes = json.load(f)
    recipe = next((r for r in recipes if r["id"] == recipe_id), None)
    if not recipe:
        return jsonify({"error": "Recipe not found"}), 404
    return jsonify(recipe)


@app.route("/api/substitute", methods=["POST"])
def substitute():
    """Generate substitution suggestions for a given ingredient."""
    try:
        data = request.get_json(force=True)
        ingredient = data.get("ingredient", "")
        context = data.get("context", "")  # e.g., recipe name
        dietary = data.get("dietary", [])

        if not ingredient:
            return jsonify({"error": "Ingredient required"}), 400

        message = (
            f"I need substitutions for '{ingredient}'"
            + (f" in {context}" if context else "")
            + ". Please list the best alternatives."
        )
        response = generate_recipe_response(
            agent_instructions=AGENT_INSTRUCTIONS,
            user_message=message,
            ingredients=[ingredient],
            dietary=dietary,
            preferences={},
            recipe_context="",
        )
        return jsonify({"substitutions": response})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")})


def _build_recipe_context(recipes: list) -> str:
    if not recipes:
        return ""
    context_parts = ["RETRIEVED RECIPES FROM KNOWLEDGE BASE:"]
    for i, r in enumerate(recipes, 1):
        context_parts.append(
            f"\n{i}. {r['name']} ({r['cuisine']}) — Match: {r.get('match_score', 0)}%"
            f"\n   Difficulty: {r['difficulty']} | Time: {r['total_time']} min | Serves: {r['servings']}"
            f"\n   Dietary: {', '.join(r.get('dietary', []))}"
            f"\n   Ingredients: {', '.join(r.get('ingredients', []))}"
            f"\n   You have {r.get('ingredient_overlap', 0)}/{r.get('total_ingredients', 0)} ingredients."
            + (f"\n   Missing: {', '.join(r.get('missing_ingredients', []))}" if r.get('missing_ingredients') else "")
        )
    return "\n".join(context_parts)


if __name__ == "__main__":
    # Render injects PORT; fall back to FLASK_PORT then 5000 for local dev
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", 5000)))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
