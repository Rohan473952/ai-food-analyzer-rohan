import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPProcessor,
    CLIPModel,
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# FOOD-101 MODEL
# ============================================================

FOOD101_MODEL = "nateraw/food"

food101_processor = AutoImageProcessor.from_pretrained(
    FOOD101_MODEL
)

food101_model = AutoModelForImageClassification.from_pretrained(
    FOOD101_MODEL
)

food101_model.to(device)
food101_model.eval()


# ============================================================
# CLIP MODEL
# ============================================================

CLIP_MODEL = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(
    CLIP_MODEL
)

clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL
)

clip_model.to(device)
clip_model.eval()


# ============================================================
# FOOD PROMPTS
# ============================================================

FOOD_PROMPTS = {

    # --------------------------------------------------------
    # NEPALI FOODS
    # --------------------------------------------------------

    "momo": [
        "a photo of Nepali momo",
        "traditional Nepali momo dumplings",
        "Nepali steamed dumplings called momo",
        "a plate of Nepali momos",
        "Nepali dumplings served with dipping sauce",
        "steamed or fried momo dumplings from Nepal",
    ],

    "dal bhat": [
        "a photo of Nepali dal bhat",
        "traditional Nepali dal bhat",
        "a traditional Nepali meal of rice and lentils",
        "Nepali rice served with lentil soup and curry",
        "a plate of dal bhat tarkari",
        "Nepali rice, dal and vegetable curry meal",
    ],

    "thukpa": [
        "a photo of Nepali thukpa",
        "traditional Nepali thukpa",
        "Nepali noodle soup",
        "a bowl of Nepali thukpa",
        "Himalayan noodle soup with vegetables",
        "traditional Himalayan noodle soup",
    ],

    "sel roti": [
        "a photo of traditional Nepali sel roti",
        "traditional Nepali sel roti on a plate",
        "Nepali sel roti made from rice flour",
        "deep fried Nepali rice flour bread called sel roti",
        "traditional homemade sel roti from Nepal",
        "golden brown ring shaped Nepali rice bread",
        "crispy ring shaped Nepali rice flour bread",
        "multiple pieces of traditional sel roti",
        "Nepali festival food sel roti",
        "traditional Nepali fried rice bread",
    ],

    "yomari": [
        "a photo of traditional Nepali yomari",
        "traditional Newari yomari",
        "Nepali sweet rice flour dumpling",
        "Newari sweet rice dumpling filled with chaku",
        "a plate of yomari",
        "traditional yomari from Nepal",
    ],

    "chatamari": [
        "a photo of traditional Nepali chatamari",
        "traditional Newari chatamari",
        "Newari rice flour crepe",
        "Nepali rice flour crepe with toppings",
        "a plate of chatamari",
        "traditional Newari rice pancake",
    ],

    # --------------------------------------------------------
    # SOUTH ASIAN FOODS
    # --------------------------------------------------------

    "samosa": [
        "a photo of samosa",
        "Indian samosa",
        "South Asian samosa",
        "a plate of samosas",
        "crispy triangular samosa",
        "deep fried samosa",
    ],

    "biryani": [
        "a photo of biryani",
        "Indian biryani",
        "South Asian biryani",
        "a plate of biryani",
        "rice dish with meat and spices",
        "fragrant spiced biryani rice",
    ],

    "naan": [
        "a photo of naan",
        "Indian naan bread",
        "traditional naan",
        "a piece of naan",
        "Indian flatbread naan",
        "freshly cooked naan bread",
    ],

    "butter chicken": [
        "a photo of butter chicken",
        "Indian butter chicken",
        "chicken in creamy tomato curry",
        "Indian chicken curry with tomato sauce",
        "a bowl of butter chicken",
        "creamy Indian chicken curry",
    ],

    "dosa": [
        "a photo of dosa",
        "South Indian dosa",
        "Indian crispy dosa",
        "a plate of dosa",
        "traditional Indian rice crepe",
        "thin crispy South Indian dosa",
    ],

    # --------------------------------------------------------
    # ASIAN FOODS
    # --------------------------------------------------------

    "sushi": [
        "a photo of sushi",
        "Japanese sushi",
        "Japanese sushi rolls",
        "a plate of sushi",
        "assorted sushi",
        "traditional Japanese sushi",
    ],

    "ramen": [
        "a photo of ramen",
        "Japanese ramen",
        "Japanese noodle soup",
        "a bowl of ramen",
        "traditional ramen noodles",
        "Japanese noodle soup with toppings",
    ],

    "chow mein": [
        "a photo of chow mein",
        "Chinese chow mein",
        "stir fried noodles",
        "a plate of chow mein",
        "Asian stir fried noodles",
        "Chinese fried noodles",
    ],

    "fried rice": [
        "a photo of fried rice",
        "Asian fried rice",
        "Chinese fried rice",
        "a plate of fried rice",
        "stir fried rice",
        "Asian rice fried with vegetables",
    ],

    "dumplings": [
        "a photo of dumplings",
        "Asian dumplings",
        "steamed dumplings",
        "a plate of dumplings",
        "Chinese dumplings",
        "assorted steamed dumplings",
    ],

    # --------------------------------------------------------
    # WESTERN FOODS
    # --------------------------------------------------------

    "pizza": [
        "a photo of pizza",
        "Italian pizza",
        "a slice of pizza",
        "a whole pizza",
        "cheesy pizza",
        "freshly baked pizza",
    ],

    "hamburger": [
        "a photo of a hamburger",
        "American hamburger",
        "a cheeseburger",
        "a beef burger",
        "a burger with fries",
        "American burger",
    ],

    "pasta": [
        "a photo of pasta",
        "Italian pasta",
        "a plate of pasta",
        "Italian noodles",
        "cooked pasta",
        "traditional Italian pasta",
    ],

    "steak": [
        "a photo of steak",
        "grilled steak",
        "beef steak",
        "a cooked steak",
        "a plate of steak",
        "grilled beef steak",
    ],

    "pancakes": [
        "a photo of pancakes",
        "American pancakes",
        "a stack of pancakes",
        "breakfast pancakes",
        "fluffy pancakes",
        "a plate of pancakes",
    ],

    "french fries": [
        "a photo of french fries",
        "crispy french fries",
        "a serving of french fries",
        "potato fries",
        "a plate of fries",
        "golden fried potato fries",
    ],

    "fried chicken": [
        "a photo of fried chicken",
        "crispy fried chicken",
        "fried chicken pieces",
        "a plate of fried chicken",
        "American fried chicken",
        "deep fried chicken",
    ],
}


# ============================================================
# HELPER: EXTRACT CLIP FEATURES
# ============================================================

def extract_features(output):
    """
    Transformers versions can return either a tensor
    or a model-output object.

    This function handles both cases.
    """

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if hasattr(output, "image_embeds"):
        return output.image_embeds

    if hasattr(output, "text_embeds"):
        return output.text_embeds

    raise TypeError(
        f"Unable to extract CLIP features from output type: "
        f"{type(output)}"
    )


# ============================================================
# PREPARE CLIP TEXT FEATURES
# ============================================================

print("Preparing CLIP food prompts...")

clip_food_names = list(FOOD_PROMPTS.keys())

clip_text_features = []

with torch.no_grad():

    for food_name in clip_food_names:

        prompts = FOOD_PROMPTS[food_name]

        inputs = clip_processor(
            text=prompts,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        text_output = clip_model.get_text_features(
            **inputs
        )

        text_features = extract_features(
            text_output
        )

        # Normalize every prompt embedding
        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

        # Average multiple prompts
        food_feature = text_features.mean(
            dim=0
        )

        # Normalize final food representation
        food_feature = food_feature / food_feature.norm()

        clip_text_features.append(
            food_feature
        )


clip_text_features = torch.stack(
    clip_text_features
)


# ============================================================
# CLASSIFY FOOD
# ============================================================

def classify_food(image, top_k=5):

    """
    Hybrid food classifier.

    Combines:
        1. Food-101
        2. Prompt-enhanced CLIP

    CLIP is given a higher weight because it can recognize
    foods that are not present in Food-101, especially
    Nepali foods.

    Parameters:
        image: PIL Image or image path
        top_k: number of predictions to return

    Returns:
        List of dictionaries:
        [
            {
                "food": "pizza",
                "confidence": 0.85
            },
            ...
        ]
    """

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")


    # ========================================================
    # FOOD-101 PREDICTION
    # ========================================================

    food101_inputs = food101_processor(
        images=image,
        return_tensors="pt"
    )

    food101_inputs = {
        key: value.to(device)
        for key, value in food101_inputs.items()
    }

    with torch.no_grad():

        outputs = food101_model(
            **food101_inputs
        )

        food101_probs = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]


    # ========================================================
    # CLIP IMAGE FEATURES
    # ========================================================

    clip_inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    clip_inputs = {
        key: value.to(device)
        for key, value in clip_inputs.items()
    }

    with torch.no_grad():

        image_output = clip_model.get_image_features(
            **clip_inputs
        )

        image_features = extract_features(
            image_output
        )

        # Normalize image embedding
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        # Compare image against all food concepts
        clip_logits = (
            image_features @ clip_text_features.T
        )[0]

        # Convert to probabilities
        clip_probs = torch.softmax(
            clip_logits * 100,
            dim=0
        )


    # ========================================================
    # COMBINE FOOD-101 + CLIP
    # ========================================================

    combined_scores = {}

    # --------------------------------------------------------
    # Food-101 contribution
    #
    # Reduced from 60% -> 35%
    # --------------------------------------------------------

    FOOD101_WEIGHT = 0.35
    CLIP_WEIGHT = 0.65

    for index, probability in enumerate(
        food101_probs
    ):

        label = food101_model.config.id2label[
            index
        ]

        label = (
            label
            .lower()
            .replace("_", " ")
        )

        combined_scores[label] = (
            probability.item() * FOOD101_WEIGHT
        )


    # --------------------------------------------------------
    # CLIP contribution
    #
    # Increased from 40% -> 65%
    # --------------------------------------------------------

    for index, probability in enumerate(
        clip_probs
    ):

        label = clip_food_names[index]

        score = probability.item() * CLIP_WEIGHT

        if label in combined_scores:

            combined_scores[label] += score

        else:

            combined_scores[label] = score


    # ========================================================
    # SPECIAL BOOST FOR NEPALI FOODS
    # ========================================================
    #
    # Nepali foods are not well represented in Food-101.
    # Give their CLIP predictions a small additional boost.
    #

    nepali_foods = {
        "momo",
        "dal bhat",
        "thukpa",
        "sel roti",
        "yomari",
        "chatamari"
    }

    for food in nepali_foods:

        if food in combined_scores:

            combined_scores[food] *= 1.12


    # ========================================================
    # SORT PREDICTIONS
    # ========================================================

    sorted_predictions = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    # ========================================================
    # NORMALIZE TOP RESULTS
    # ========================================================

    top_predictions = sorted_predictions[:top_k]

    total = sum(
        score for _, score in top_predictions
    )

    results = []

    for food, score in top_predictions:

        if total > 0:
            confidence = score / total
        else:
            confidence = score

        results.append(
            {
                "food": food,
                "confidence": float(confidence)
            }
        )


    # ========================================================
    # RETURN
    # ========================================================

    return results