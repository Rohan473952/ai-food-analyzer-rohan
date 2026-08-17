import streamlit as st
from PIL import Image

from vision.food_classifier import classify_food
from llm.analyzer import analyze_food


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Food Analyzer",
    page_icon="🍽️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 35px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        background-color: #1b1f27;
        margin-top: 20px;
    }

    .food-name {
        font-size: 2rem;
        font-weight: 700;
    }

    .confidence {
        font-size: 1.1rem;
        color: #b5b5b5;
        margin-top: 8px;
    }

    .analysis-card {
        padding: 25px;
        border-radius: 15px;
        background-color: #171b23;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🍽️ AI Food Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Identify food from an image and get an AI-powered analysis.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    top_k = st.slider(
        "Number of food predictions",
        min_value=1,
        max_value=5,
        value=3
    )

    st.markdown("---")

    st.markdown(
        """
        ### 🤖 AI Pipeline

        **1. Computer Vision**

        Food-101 + CLIP

        ↓

        **2. Food Recognition**

        ↓

        **3. Qwen 2.5**

        ↓

        **4. AI Food Analysis**
        """
    )

    st.markdown("---")

    st.caption(
        "Food recognition uses a hybrid Food-101 + CLIP model."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a food image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # TWO COLUMNS
    # --------------------------------------------------------

    col1, col2 = st.columns(
        [1, 1]
    )


    # ========================================================
    # LEFT COLUMN — IMAGE
    # ========================================================

    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )


    # ========================================================
    # RIGHT COLUMN — RECOGNITION
    # ========================================================

    with col2:

        st.subheader("🔍 Food Recognition")

        analyze_button = st.button(
            "Analyze Food",
            type="primary",
            use_container_width=True
        )


        if analyze_button:

            # =================================================
            # STEP 1 — COMPUTER VISION
            # =================================================

            with st.spinner(
                "👁️ AI is identifying the food..."
            ):

                try:

                    predictions = classify_food(
                        image,
                        top_k=top_k
                    )

                except Exception as e:

                    st.error(
                        f"Food recognition failed: {e}"
                    )

                    predictions = []


            # =================================================
            # STEP 2 — FOOD RECOGNITION RESULT
            # =================================================

            if predictions:

                best_prediction = predictions[0]

                food_name = best_prediction["food"]

                confidence = (
                    best_prediction["confidence"] * 100
                )


                # ------------------------------------------------
                # MAIN RESULT CARD
                # ------------------------------------------------
            # Clean up the predicted food name
            display_food_name = food_name.replace("_", " ").title()

            st.markdown(
                f"## 🍴 {display_food_name}"
        ) 

            st.write(
                f"**Recognition confidence:** {confidence:.2f}%"
            )

            # =================================================
            # OTHER PREDICTIONS
            # =================================================

            if len(predictions) > 1:

                st.markdown(
                    "### Other possibilities"
                )

                for prediction in predictions[1:]:

                        other_food = prediction["food"].replace("_", " ").title()

                        other_score = (
                            prediction["confidence"] * 100
                        )

                        st.write(
                             f"**{other_food}** — {other_score:.2f}%"
            )

                        st.progress(
                            min(
                                max(
                                    other_score / 100,
                                    0.0
                                ),
                                1.0
                            )
                        )


                # =================================================
                # STEP 3 — QWEN FOOD ANALYSIS
                # =================================================

                st.markdown("---")

                st.subheader(
                    "🧠 AI Food Analysis"
                )


                # ------------------------------------------------
                # QUESTION FOR QWEN
                # ------------------------------------------------

                user_question = f"""
Analyze the food identified as {food_name}.

Provide the following information:

## 🥘 Common Ingredients

List the ingredients that are commonly used to prepare
this food.

## 🔥 Nutritional Information

Give general nutritional information including:

- Calories
- Carbohydrates
- Protein
- Fat
- Fiber

Do not pretend the values are exact because nutritional
content depends on portion size, ingredients, and preparation.

## ⚠️ Common Allergens

List common allergens that may be present in this food.

## 💚 Health Considerations

Explain the general health considerations associated
with this food.

## 🥗 Healthier Alternatives

Give practical suggestions for making this food healthier.

Keep the response clear, useful, and reasonably concise.
This is general nutritional information and not medical advice.
"""


                # =================================================
                # CALL QWEN
                # =================================================

                with st.spinner(
                    f"🤖 Qwen is analyzing {food_name}..."
                ):

                    try:

                        analysis = analyze_food(
                            food_name,
                            user_question
                        )

                    except Exception as e:

                        analysis = (
                            f"❌ Qwen analysis failed: {e}"
                        )


                # =================================================
                # DISPLAY QWEN RESULT
                # =================================================

                st.markdown(
                    '<div class="analysis-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    analysis
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


            else:

                st.warning(
                    "No food prediction was returned."
                )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "👆 Upload an image above to begin your food analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Food Analyzer • Food-101 + CLIP + Qwen 2.5"
)