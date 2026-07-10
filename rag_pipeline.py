"""
rag_pipeline.py
---------------
RAG (Retrieval-Augmented Generation) pipeline.
Queries ChromaDB for relevant recipes given user ingredients.
"""

import json
import os
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RECIPES_FILE = os.path.join(os.path.dirname(__file__), "data", "recipes.json")

_client = None
_collection = None
_recipes_cache: Optional[dict] = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        # Use get_or_create_collection to prevent crash if collection doesn't exist
        _collection = _client.get_or_create_collection(
            name="recipes",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # If the collection is empty (e.g. fresh persistent disk on Render), automatically seed it
        if _collection.count() == 0:
            print("ChromaDB collection 'recipes' is empty. Seeding database at runtime...")
            try:
                _seed_database_at_runtime(_collection)
            except Exception as e:
                print(f"Error seeding database at runtime: {e}")
                
    return _collection


def _seed_database_at_runtime(collection):
    if not os.path.exists(RECIPES_FILE):
        print(f"Recipes file not found at {RECIPES_FILE}")
        return
        
    with open(RECIPES_FILE, "r", encoding="utf-8") as f:
        recipes = json.load(f)
        
    ids, documents, metadatas = [], [], []
    for recipe in recipes:
        # Build text document for embedding
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
        for i, step in enumerate(recipe.get("instructions", []), 1):
            parts.append(f"Step {i}: {step}")
        doc = "\n".join(parts)
        
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
    print(f"Successfully seeded {len(recipes)} recipes into ChromaDB collection at runtime.")


def _load_recipes() -> dict:
    global _recipes_cache
    if _recipes_cache is None:
        with open(RECIPES_FILE, "r", encoding="utf-8") as f:
            recipes = json.load(f)
        _recipes_cache = {r["id"]: r for r in recipes}
    return _recipes_cache


def retrieve_recipes(
    ingredients: list[str],
    dietary_filters: list[str] = None,
    n_results: int = 4,
) -> list[dict]:
    """
    Retrieve the top-N most relevant recipes from ChromaDB
    based on available ingredients and optional dietary filters.

    Returns a list of full recipe dicts with a 'match_score' field added.
    """
    collection = _get_collection()
    recipes_db = _load_recipes()

    query_text = "Ingredients available: " + ", ".join(ingredients)
    if dietary_filters:
        query_text += ". Dietary preferences: " + ", ".join(dietary_filters)

    where_clause = None
    # ChromaDB supports $contains for arrays stored as JSON strings
    # For simplicity we filter after retrieval
    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results * 3, len(recipes_db)),  # over-fetch then re-rank
        include=["metadatas", "distances"],
    )

    candidates = []
    if results and results["metadatas"]:
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            recipe_id = meta.get("id")
            recipe = recipes_db.get(recipe_id)
            if not recipe:
                continue

            # Post-filter by dietary preferences
            if dietary_filters:
                recipe_dietary = [d.lower() for d in recipe.get("dietary", [])]
                if not all(pref.lower() in recipe_dietary for pref in dietary_filters):
                    continue

            # Compute ingredient overlap score (0–1)
            recipe_ings = [i.lower() for i in recipe.get("ingredients", [])]
            user_ings = [i.lower() for i in ingredients]
            overlap = sum(1 for ing in user_ings if any(ing in ri for ri in recipe_ings))
            overlap_score = overlap / max(len(recipe_ings), 1)

            # Combined score: semantic similarity (1-dist) + ingredient overlap
            combined = (1 - dist) * 0.6 + overlap_score * 0.4

            recipe_copy = dict(recipe)
            recipe_copy["match_score"] = round(combined * 100, 1)
            recipe_copy["ingredient_overlap"] = overlap
            recipe_copy["total_ingredients"] = len(recipe_ings)
            recipe_copy["missing_ingredients"] = [
                ri for ri in recipe_ings if not any(u in ri for u in user_ings)
            ]
            candidates.append(recipe_copy)

    # Sort by combined score descending
    candidates.sort(key=lambda x: x["match_score"], reverse=True)
    return candidates[:n_results]
