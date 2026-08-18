import requests
import re


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# FOOD ANALYZER
# ============================================================

def analyze_food(food_name, custom_prompt=None):
    """
    Analyze a recognized food using Groq.

    Returns a short, structured five-section response.
    Compatible with the existing app.py call:
        analyze_food(food_name, custom_prompt=...)
    """

    # --------------------------------------------------------
    # BASE PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a concise AI food and nutrition assistant.

The computer vision system identified the food as:

{food_name}

Your job is to provide a COMPLETE food analysis.

IMPORTANT RULES:

1. You MUST provide ALL FIVE sections.
2. NEVER stop after only one or two sections.
3. Keep every section SHORT.
4. Do not write an introduction.
5. Do not write a conclusion.
6. Do not repeat the food name unnecessarily.
7. Do not use tables.
8. Do not use long explanations.
9. If exact nutritional values are unknown, give reasonable approximate values.
10. Clearly state that nutrition varies with portion size and preparation.
11. Do not give medical advice.

Use EXACTLY these five headings:

## 🥘 Ingredients

List 3-5 common ingredients.

## 🔥 Nutritional Information

Give an approximate calorie range for one normal serving.
Briefly mention:
- carbohydrates
- protein
- fats
- fiber

Keep this section to 3-5 short lines.

## ⚠️ Common Allergens

List the most likely allergens.
If there are no major common allergens, say so.

Keep this section to 1-3 short lines.

## 💚 Health Considerations

Give exactly 2 short points about nutritional benefits
or things to watch for.

## 🥗 Healthier Alternative

Give exactly 2 practical suggestions for making this food healthier.

The response MUST contain all five headings before ending.

Food: {food_name}
"""

    # --------------------------------------------------------
    # OPTIONAL CUSTOM PROMPT
    # --------------------------------------------------------

    if custom_prompt:
        prompt += f"""

Additional user request:
{custom_prompt}

Follow the five-section format above regardless of the request.
"""


    # ========================================================
    # GROQ REQUEST
    # ========================================================

    payload = {
        "model": GROQ_MODEL_NAME,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise food analysis assistant. "
                    "Always complete all five requested sections."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": 0.2,

        "max_completion_tokens": 1200,

        "reasoning_effort": "low",

        "include_reasoning": False,

        "stream": False
    }


    try:

        response = requests.post(
            GROQ_URL,
            json=payload,
            timeout=90
        )

        response.raise_for_status()

        data = response.json()

        # ----------------------------------------------------
        # EXTRACT RESPONSE
        # ----------------------------------------------------

        choices = data.get("choices", [])

        if not choices:
            return "❌ Groq returned no analysis."

        message = choices[0].get("message", {})

        answer = message.get("content", "")

        if not answer:
            return "❌ Groq returned an empty analysis."

        answer = answer.strip()


        # ----------------------------------------------------
        # CLEAN POSSIBLE REASONING / EXTRA TEXT
        # ----------------------------------------------------

        # Remove accidental <think>...</think> blocks
        answer = re.sub(
            r"<think>.*?</think>",
            "",
            answer,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()


        # ----------------------------------------------------
        # ENSURE THE FIVE SECTIONS EXIST
        # ----------------------------------------------------

        required_sections = [
            "## 🥘 Ingredients",
            "## 🔥 Nutritional Information",
            "## ⚠️ Common Allergens",
            "## 💚 Health Considerations",
            "## 🥗 Healthier Alternative"
        ]

        missing_sections = [
            section
            for section in required_sections
            if section not in answer
        ]


        # ----------------------------------------------------
        # IF THE MODEL RETURNED A PARTIAL ANSWER
        # ----------------------------------------------------

        if missing_sections:

            completion_prompt = f"""
The previous food analysis was incomplete.

Food:
{food_name}

The following sections are missing:

{chr(10).join(missing_sections)}

Complete ONLY the missing sections.

Use these EXACT headings:

## 🥘 Ingredients

## 🔥 Nutritional Information

## ⚠️ Common Allergens

## 💚 Health Considerations

## 🥗 Healthier Alternative

Rules:
- Keep every section very short.
- Do not repeat sections that were already provided.
- Do not write an introduction.
- Do not write a conclusion.
- Complete every missing section.
"""

            completion_payload = {
                "model": GROQ_MODEL_NAME,

                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Complete missing sections of a food "
                            "analysis concisely."
                        )
                    },
                    {
                        "role": "user",
                        "content": completion_prompt
                    }
                ],

                "temperature": 0.2,

                "max_completion_tokens": 700,

                "reasoning_effort": "low",

                "include_reasoning": False,

                "stream": False
            }


            try:

                completion_response = requests.post(
                    GROQ_URL,
                    json=completion_payload,
                    timeout=60
                )

                completion_response.raise_for_status()

                completion_data = completion_response.json()

                completion_choices = completion_data.get(
                    "choices",
                    []
                )

                if completion_choices:

                    completion_content = (
                        completion_choices[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    if completion_content:

                        completion_content = re.sub(
                            r"<think>.*?</think>",
                            "",
                            completion_content,
                            flags=re.DOTALL | re.IGNORECASE
                        ).strip()

                        answer = (
                            answer
                            + "\n\n"
                            + completion_content
                        )

            except Exception:
                # If the second request fails, keep the
                # original response rather than breaking
                # the entire application.
                pass


        # ----------------------------------------------------
        # FINAL CLEANUP
        # ----------------------------------------------------

        answer = answer.strip()

        return answer


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except requests.exceptions.ConnectionError:

        return (
            "❌ Could not connect to Groq. "
            "Please check the API connection."
        )

    except requests.exceptions.Timeout:

        return (
            "❌ The AI analysis took too long. "
            "Please try again."
        )

    except requests.exceptions.HTTPError as e:

        try:
            error_details = response.json()

            error_message = (
                error_details
                .get("error", {})
                .get("message", str(e))
            )

        except Exception:
            error_message = str(e)

        return (
            f"❌ Groq AI analysis failed: "
            f"{error_message}"
        )

    except Exception as e:

        return f"❌ Error communicating with Groq: {e}"