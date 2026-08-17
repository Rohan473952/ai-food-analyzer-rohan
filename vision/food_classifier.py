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
        "a photo of momo",
        "traditional Nepali momo",
        "Nepali steamed dumplings called momo",
        "a plate of momos",
        "Nepali dumplings",
    ],

    "dal bhat": [
        "a photo of dal bhat",
        "traditional Nepali dal bhat",
        "a traditional Nepali meal of rice and lentils",
        "Nepali rice and lentil curry",
        "a plate of dal bhat",
    ],

    "thukpa": [
        "a photo of thukpa",
        "traditional Nepali thukpa",
        "Nepali noodle soup",
        "a bowl of thukpa",
        "Himalayan noodle soup",
    ],

    "sel roti": [
        "a photo of sel roti",
        "traditional Nepali sel roti",
        "Nepali ring shaped rice bread",
        "traditional Nepali rice bread",
        "a plate of sel roti",
    ],

    "yomari": [
        "a photo of yomari",
        "traditional Nepali yomari",
        "Nepali sweet rice dumpling",
        "traditional Newari yomari",
        "a plate of yomari",
    ],

    "chatamari": [
        "a photo of chatamari",
        "traditional Nepali chatamari",
        "Newari rice crepe",
        "Nepali rice flour crepe",
        "a plate of chatamari",
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
    ],

    "biryani": [
        "a photo of biryani",
        "Indian biryani",
        "South Asian biryani",
        "a plate of biryani",
        "rice dish with meat and spices",
    ],

    "naan": [
        "a photo of naan",
        "Indian naan bread",
        "traditional naan",
        "a piece of naan",
        "Indian flatbread naan",
    ],

    "butter chicken": [
        "a photo of butter chicken",
        "Indian butter chicken",
        "chicken in creamy tomato curry",
        "Indian chicken curry with tomato sauce",
        "a bowl of butter chicken",
    ],

    "dosa": [
        "a photo of dosa",
        "South Indian dosa",
        "Indian crispy dosa",
        "a plate of dosa",
        "traditional Indian rice crepe",
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
    ],

    "ramen": [
        "a photo of ramen",
        "Japanese ramen",
        "Japanese noodle soup",
        "a bowl of ramen",
        "traditional ramen noodles",
    ],

    "chow mein": [
        "a photo of chow mein",
        "Chinese chow mein",
        "stir fried noodles",
        "a plate of chow mein",
        "Asian stir fried noodles",
    ],

    "fried rice": [
        "a photo of fried rice",
        "Asian fried rice",
        "Chinese fried rice",
        "a plate of fried rice",
        "stir fried rice",
    ],

    "dumplings": [
        "a photo of dumplings",
        "Asian dumplings",
        "steamed dumplings",
        "a plate of dumplings",
        "Chinese dumplings",
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
    ],

    "hamburger": [
        "a photo of a hamburger",
        "American hamburger",
        "a cheeseburger",
        "a beef burger",
        "a burger with fries",
    ],

    "pasta": [
        "a photo of pasta",
        "Italian pasta",
        "a plate of pasta",
        "Italian noodles",
        "cooked pasta",
    ],

    "steak": [
        "a photo of steak",
        "grilled steak",
        "beef steak",
        "a cooked steak",
        "a plate of steak",
    ],

    "pancakes": [
        "a photo of pancakes",
        "American pancakes",
        "a stack of pancakes",
        "breakfast pancakes",
        "fluffy pancakes",
    ],

    "french fries": [
        "a photo of french fries",
        "crispy french fries",
        "a serving of french fries",
        "potato fries",
        "a plate of fries",
    ],

    "fried chicken": [
        "a photo of fried chicken",
        "crispy fried chicken",
        "fried chicken pieces",
        "a plate of fried chicken",
        "American fried chicken",
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

        # Average the multiple prompts
        food_feature = text_features.mean(
            dim=0
        )

        # Normalize the final food representation
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
    # --------------------------------------------------------

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
            probability.item() * 0.60
        )


    # --------------------------------------------------------
    # CLIP contribution
    # --------------------------------------------------------

    for index, probability in enumerate(
        clip_probs
    ):

        label = clip_food_names[index]

        score = probability.item() * 0.40

        if label in combined_scores:

            combined_scores[label] += score

        else:

            combined_scores[label] = score


    # ========================================================
    # SORT PREDICTIONS
    # ========================================================

    sorted_predictions = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    # ========================================================
    # RETURN TOP K
    # ========================================================

    results = []

    for food, score in sorted_predictions[:top_k]:

        results.append(
            {
                "food": food,
                "confidence": float(score)
            }
        )


    return results
