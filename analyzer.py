import os
import requests


# ============================================================
# LOCAL OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

LOCAL_MODEL_NAME = "qwen2.5:3b-instruct"


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# Qwen model available through Hugging Face Inference Providers
HF_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


# ============================================================
# FOOD ANALYZER
# ============================================================

def analyze_food(food_name, custom_prompt=None):
    """
    Analyze a recognized food using Qwen.

    Local:
        Ollama + Qwen 2.5

    Deployed:
        Hugging Face Inference Providers + Qwen

    Parameters:
        food_name: recognized food name
        custom_prompt: optional additional user question

    Returns:
        AI-generated food analysis
    """

    # ========================================================
    # DEFAULT FOOD ANALYSIS PROMPT
    # ========================================================

    prompt = f"""
You are an AI food and nutrition assistant.

The computer vision model identified the food as:

{food_name}

Analyze this food and provide useful information.

Return the answer using exactly these sections:

## 🥘 Ingredients

List the common ingredients normally found in this food.

## 🔥 Nutritional Information

Give a general estimate of calories and discuss:

- carbohydrates
- protein
- fats
- fiber

Do NOT pretend these are exact values because preparation,
portion size, ingredients, and cooking methods vary.

## ⚠️ Common Allergens

List common allergens that may be present.

## 💚 Health Considerations

Explain whether the food can be part of a healthy diet
and mention factors that affect its nutritional value.

## 🥗 Healthier Alternative

Give practical suggestions for making this food healthier.

Keep the answer concise and easy for a normal person to understand.

Do not give medical advice.
"""


    # ========================================================
    # OPTIONAL CUSTOM QUESTION
    # ========================================================

    if custom_prompt:

        prompt += f"""

Additional user request:

{custom_prompt}
"""


    # ========================================================
    # DEPLOYMENT MODE — HUGGING FACE
    # ========================================================

    if HF_TOKEN:

        try:

            from huggingface_hub import InferenceClient

            client = InferenceClient(
                provider="auto",
                api_key=HF_TOKEN
            )

            response = client.chat.completions.create(
                model=HF_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=700,
                temperature=0.7
            )

            return response.choices[0].message.content


        except Exception as e:

            return (
                "❌ Hugging Face AI analysis failed: "
                f"{e}"
            )


    # ========================================================
    # LOCAL MODE — OLLAMA
    # ========================================================

    payload = {
        "model": LOCAL_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }


    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            "No analysis was returned."
        )


    except requests.exceptions.ConnectionError:

        return (
            "❌ Could not connect to Ollama. "
            "Make sure Ollama is running."
        )


    except requests.exceptions.Timeout:

        return (
            "❌ The AI analysis took too long. "
            "Please try again."
        )


    except Exception as e:

        return (
            f"❌ Error communicating with Qwen: {e}"
        )