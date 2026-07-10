"""
watsonx_client.py
-----------------
IBM Watsonx.ai Granite model client.
Handles authentication and text generation requests.
Compatible with ibm-watsonx-ai >= 1.5.14.
"""

import os
from dotenv import load_dotenv

load_dotenv()

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://eu-gb.ml.cloud.ibm.com")
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "mistralai/mistral-small-3-1-24b-instruct-2503")

_model = None


def _get_model():
    """Lazy-initialise the Watsonx.ai model client (ibm-watsonx-ai >= 1.5.x)."""
    global _model
    if _model is not None:
        return _model

    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.schema import TextGenParameters

    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    params = TextGenParameters(
        max_new_tokens=600,
        min_new_tokens=30,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        stop_sequences=["<|endoftext|>", "\n\nHuman:", "\n\nUser:"],
    )
    _model = ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials=creds,
        project_id=WATSONX_PROJECT_ID,
        params=params,
    )
    return _model


def build_prompt(
    agent_instructions: str,
    user_message: str,
    ingredients: list,
    dietary: list,
    preferences: dict,
    recipe_context: str,
) -> str:
    """Build the full Granite instruction prompt."""

    ingredient_str = ", ".join(ingredients) if ingredients else "Not specified"
    dietary_str = ", ".join(dietary) if dietary else "No restrictions"
    spice_level = preferences.get("spice_level", "Medium")
    servings = preferences.get("servings", 2)

    prompt = f"""<|system|>
{agent_instructions}
<|end|>

<|user|>
USER CONTEXT:
- Available Ingredients: {ingredient_str}
- Dietary Restrictions: {dietary_str}
- Spice Preference: {spice_level}
- Servings Needed: {servings}

{recipe_context}

USER REQUEST:
{user_message}
<|end|>

<|assistant|>
"""
    return prompt


def generate_recipe_response(
    agent_instructions: str,
    user_message: str,
    ingredients: list,
    dietary: list,
    preferences: dict,
    recipe_context: str,
) -> str:
    """Generate a recipe response using IBM Granite via Watsonx.ai."""
    model = _get_model()

    prompt = build_prompt(
        agent_instructions=agent_instructions,
        user_message=user_message,
        ingredients=ingredients,
        dietary=dietary,
        preferences=preferences,
        recipe_context=recipe_context,
    )

    result = model.generate_text(prompt=prompt)
    if isinstance(result, dict):
        return result.get("results", [{}])[0].get("generated_text", "").strip()
    return str(result).strip()
