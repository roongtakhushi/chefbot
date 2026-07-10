"""
seed_knowledge_base.py
----------------------
One-time script to load recipe data into ChromaDB vector store.
Run this before starting the Flask server.

Usage:
    python seed_knowledge_base.py
"""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
RECIPES_FILE = os.path.join(os.path.dirname(__file__), "data", "recipes.json")


def build_recipe_document(recipe: dict) -> str:
    """Build a rich text document from recipe for embedding."""
    parts = [
        f"Recipe: {recipe['name']}",
        f"Cuisine: {recipe['cuisine']}",
        f"Category: {recipe['category']}",
        f"Dietary: {', '.join(recipe.get('dietary', []))}",
        f"Difficulty: {recipe['difficulty']}",
        f"Spice Level: {recipe.get('spice_level', 'Unknown')}",
        f"Cooking Time: {recipe['total_time']} minutes",
        f"Servings: {recipe['servings']}",
        f"Ingredients: {', '.join(recipe['ingredients'])}",
        f"Tags: {', '.join(recipe.get('tags', []))}",
    ]
    # Add cooking steps as searchable text
    for i, step in enumerate(recipe.get("instructions", []), 1):
        parts.append(f"Step {i}: {step}")
    return "\n".join(parts)


def seed():
    try:
        import chromadb
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    except ImportError:
        print("ERROR: chromadb not installed.")
        print("Run: pip install chromadb")
        sys.exit(1)

    if not os.path.exists(RECIPES_FILE):
        print(f"ERROR: recipes.json not found at {RECIPES_FILE}")
        sys.exit(1)

    with open(RECIPES_FILE, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    embedding_fn = ONNXMiniLM_L6_V2()

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Drop and recreate collection for clean seed
    try:
        client.delete_collection("recipes")
        print("Existing 'recipes' collection deleted.")
    except Exception:
        pass

    collection = client.create_collection(
        name="recipes",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = [], [], []
    for recipe in recipes:
        doc = build_recipe_document(recipe)
        meta = {
            "id": recipe["id"],
            "name": recipe["name"],
            "cuisine": recipe["cuisine"],
            "category": recipe["category"],
            "dietary": json.dumps(recipe.get("dietary", [])),
            "difficulty": recipe["difficulty"],
            "total_time": recipe["total_time"],
            "spice_level": recipe.get("spice_level", "Unknown"),
            "servings": recipe["servings"],
            "image_emoji": recipe.get("image_emoji", "🍽️"),
            "tags": json.dumps(recipe.get("tags", [])),
        }
        ids.append(recipe["id"])
        documents.append(doc)
        metadatas.append(meta)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[OK] Successfully seeded {len(recipes)} recipes into ChromaDB at '{CHROMA_PERSIST_DIR}'")


if __name__ == "__main__":
    seed()
