import os
import requests


# ============================================================
# LOCAL OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL_NAME = "qwen2.5:3b-instruct"


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


# ============================================================
# FOOD ANALYZER
# ============================================================

def analyze_food(food_name, custom_prompt=None):
    """
    Analyze recognized food using:

    Local:
        Ollama + Qwen 2.5

    Deployed:
        Groq + Llama 3.1

    The function keeps the same interface used by app.py.
    """

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

    # Optional custom question
    if custom_prompt:
        prompt += f"""

Additional user request:

{custom_prompt}
"""

    # ========================================================
    # DEPLOYED MODE — GROQ
    # ========================================================

    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI food and nutrition "
                        "assistant. Be concise and informative."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 700,
            "stream": False
        }

        try:

            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:

            return (
                "❌ The AI analysis took too long. "
                "Please try again."
            )

        except requests.exceptions.RequestException as e:

            return f"❌ Groq AI analysis failed: {e}"

        except Exception as e:

            return f"❌ Error processing AI response: {e}"

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

        return f"❌ Error communicating with Qwen: {e}"