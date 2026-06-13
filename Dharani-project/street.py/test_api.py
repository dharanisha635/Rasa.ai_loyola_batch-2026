# to check the available free models

import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

print("API Key found:", API_KEY[:15] + "..." if API_KEY else "❌ NOT FOUND")

# First check available models
models_response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={
        "Authorization": f"Bearer {API_KEY}"
    }
)

models = models_response.json()

print("\n📋 Free models available to you:")

if "data" in models:
    free_models = [
        m["id"] for m in models["data"]
        if ":free" in m["id"]
    ]
    for m in free_models[:10]:   # Show first 10
        print(f"  ✅ {m}")
else:
    print("❌ Could not fetch models")
    print(models)