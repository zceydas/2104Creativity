import os, json, time
import pandas as pd
from openai import OpenAI

# ------------- SETTINGS -------------
IN_PATH = "aut_all_items.csv"
OUT_PATH = "aut_all_items_coded.csv"
MODEL = "gpt-4o-mini"
BATCH_SIZE = 25
TEMPERATURE = 0.2
MAX_RETRIES = 6

# OPTION A (recommended): provide category sets per item (most reproducible)
# Add/modify items here. Categories must be the labels you want returned.
ITEM_TO_CATEGORIES = {

    "brick": [
        "Construction Material",
        "Architectural Structure / Infrastructure",
        "Weight / Ballast / Counterweight",
        "Paperweight / Desk Weight",
        "Doorstop / Door Jam",
        "Prop / Support / Stabilizer",
        "Furniture / Seating / Footrest",
        "Step / Elevation Aid",
        "Exercise / Fitness Equipment",
        "Weapon / Bludgeon / Self-Defense",
        "Throwing Object / Projectile",
        "Tool / Hammering / Striking Implement",
        "Measuring / Leveling / Calibration Device",
        "Art / Sculpture / Decoration",
        "Cooking / Heating / Fire Use",
        "Garden / Landscaping Tool",
        "Musical Instrument / Sound-Making Object",
        "Writing / Marking / Drawing Surface",
        "Toy / Game Piece",
        "Medical / Protective Aid"
    ],

    "book": [
        "Weight / Ballast / Counterweight",
        "Paperweight / Desk Weight",
        "Doorstop / Door Jam",
        "Prop / Support / Stabilizer",
        "Furniture / Seating / Footrest",
        "Step / Elevation Aid",
        "Exercise / Fitness Equipment",
        "Weapon / Bludgeon / Self-Defense",
        "Throwing Object / Projectile",
        "Tool / Hammering / Striking Implement",
        "Measuring / Leveling / Calibration Device",
        "Art / Sculpture / Decoration",
        "Writing / Marking / Drawing Surface",
        "Cooking / Heating / Fire Use",
        "Garden / Landscaping Tool",
        "Musical Instrument / Sound-Making Object",
        "Medical / Protective Aid",
        "Toy / Game Piece",
        "Clothing / Wearable Item"
    ],

    "bottle": [
        "Transport / Mobility Device",
        "Container / Storage Vessel",
        "Beverage Holder / Liquid Carrier",
        "Measuring / Unit of Measurement",
        "Weight / Ballast / Counterweight",
        "Doorstop / Door Jam",
        "Prop / Support / Stabilizer",
        "Furniture Component / Structural Support",
        "Step / Elevation Aid",
        "Exercise / Fitness Equipment",
        "Weapon / Blunt Object / Self-Defense",
        "Throwing Object / Projectile",
        "Tool / Hammering / Rolling / Cutting Implement",
        "Digging / Gardening Tool",
        "Art / Sculpture / Decoration",
        "Flower Vase / Plant Holder",
        "Toy / Game Object",
        "Musical Instrument / Sound-Making Object",
        "Light Source / Optical Device",
        "Scientific / Chemistry Apparatus"
    ],

    "cord": [
        "Rope / Tethering Tool",
        "Clothing / Fashion Accessory",
        "Jewelry / Personal Adornment",
        "Leash / Animal Restraint",
        "Restraint / Binding Device",
        "Weapon / Whip / Strangulation Tool",
        "Exercise Equipment / Jump Rope",
        "Climbing / Outdoor Gear",
        "Fishing / Snare Tool",
        "Measuring Tool",
        "Musical Instrument Component",
        "Firestarter / Fuel",
        "Art / Craft Material",
        "Decoration / Ornament",
        "Plant Support / Hanger",
        "Connection / Fastening Device",
        "Electrical / Power Cable",
        "Toy / Play Object",
        "Household Utility Item"
    ],

    "fork": [
        "Eating Utensil / Cutlery",
        "Back Scratcher / Personal Grooming Tool",
        "Hair Comb / Hair Grooming Tool",
        "Cleaning / Scraping Tool",
        "Digging / Gardening Tool",
        "Agricultural / Soil Aeration Tool",
        "Tool / Prying / Levering Implement",
        "Tool / Hammering / Striking Implement",
        "Weapon / Stabbing / Self-Defense",
        "Throwing Object / Projectile",
        "Measuring / Marking Instrument",
        "Writing / Drawing Implement",
        "Musical Instrument / Sound-Making Object",
        "Electrical Conductor / Science Apparatus",
        "Art / Sculpture / Craft Material",
        "Jewelry / Fashion Accessory",
        "Medical / Surgical Instrument",
        "Toy / Game Object",
        "Doorstop / Prop / Support Device"
    ],

    "magazine": [
        "Reading Material / Information Source",
        "Art / Craft Material",
        "Collage / Scrapbooking Material",
        "Fire Starter / Kindling",
        "Fly Swatter / Insect Control Tool",
        "Fan / Airflow Device",
        "Weapon / Blunt Object",
        "Throwing / Play Object",
        "Toy / Game Object",
        "Decoration / Wall Art",
        "Clothing / Wearable Item",
        "Sunshade / Rain Cover",
        "Placemat / Surface Protector",
        "Doorstop / Prop / Shim",
        "Insulation / Draft Stop",
        "Cushion / Padding Material",
        "Cleaning / Wiping Tool",
        "Gardening / Compost Material",
        "Writing / Notepad Surface",
        "Building / Structural Material"
    ],

    "paperclip": [
        "Fastener / Paper Organizer",
        "Lockpick / Unlocking Tool",
        "Key / Access Tool",
        "Jewelry / Personal Adornment",
        "Hair Accessory",
        "Clothing Fastener / Clasp",
        "Toothpick / Oral Hygiene Tool",
        "Fingernail / Personal Cleaning Tool",
        "Scratching / Itch Relief Tool",
        "Hook / Hanging Tool",
        "Wire / Connector / Electrical Conductor",
        "Circuit / Electronic Tool",
        "SIM Card / Reset Tool",
        "Sewing / Needle Tool",
        "Piercing / Medical Instrument",
        "Weapon / Stabbing Implement",
        "Art / Sculpture Material",
        "Toy / Fidget Object",
        "Hole Punch / Poking Tool",
        "Measurement / Marking Tool"
    ],

    "tincan": [
        "Container / Storage Vessel",
        "Cooking / Baking Utensil",
        "Cutting / Bladed Tool",
        "Strainer / Sieve",
        "Cup / Bowl / Drinking Vessel",
        "Art / Sculpture / Decoration",
        "Weight / Counterweight",
        "Doorstop / Prop",
        "Furniture / Seat / Step Stool",
        "Weapon / Blunt Object",
        "Musical Instrument / Percussion Device",
        "Toy / Game Object",
        "Telephone / Communication Device",
        "Building Material / Structural Component",
        "Gardening / Planter",
        "Bird Feeder / Animal Habitat",
        "Stencil / Template Tool",
        "Rolling Pin / Dough Roller",
        "Shovel / Digging Tool",
        "Recycling / Scrap Metal Material"
    ],

    "towel": [
        "Clothing / Wearable Garment",
        "Blanket / Bedding Item",
        "Curtain / Window Covering",
        "Rug / Mat / Floor Covering",
        "Tablecloth / Surface Cover",
        "Shade / Sun Protection",
        "Insulation / Draft Blocker",
        "Bandage / Medical Dressing",
        "Tourniquet / Medical Restraint",
        "Sling / Support Device",
        "Rope / Binding Tool",
        "Whip / Weapon",
        "Toy / Play Object",
        "Exercise Equipment",
        "Cleaning / Wiping Tool",
        "Storage / Carrying Device",
        "Shelter / Tent / Canopy",
        "Art / Decorative Fabric",
        "Filter / Strainer",
        "Fire Starter / Fuel"
    ]
}

# OPTION B: if an item isn't in ITEM_TO_CATEGORIES, the model will propose categories
# (You can turn this OFF by setting ALLOW_CATEGORY_DISCOVERY=False)
ALLOW_CATEGORY_DISCOVERY = True
DISCOVERY_TARGET_MIN = 15
DISCOVERY_TARGET_MAX = 20

SYSTEM_CLASSIFY = (
    "You are coding Alternate Uses Test (AUT) responses for a given object (item). "
    "For each response, choose exactly one best-fit category from the allowed list for that item. "
    "Return only valid JSON matching the schema."
)

SYSTEM_DISCOVER = (
    "You are designing a coding scheme for AUT responses. "
    "Given an item and example responses, propose a set of distinct, non-overlapping category names "
    "that cover the responses. Use between 15 and 20 categories. "
    "Return only valid JSON matching the schema."
)

def backoff_sleep(attempt: int):
    time.sleep(min(2 ** attempt, 30))

def call_with_retries(fn):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            last_err = e
            backoff_sleep(attempt)
    raise last_err

def discover_categories(client: OpenAI, item: str, example_responses: list[str]) -> list[str]:
    schema = {
        "name": "aut_category_discovery",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "categories": {
                    "type": "array",
                    "minItems": DISCOVERY_TARGET_MIN,
                    "maxItems": DISCOVERY_TARGET_MAX,
                    "items": {"type": "string"}
                }
            },
            "required": ["categories"]
        },
        "strict": True
    }

    sample = example_responses[:200]  # cap for prompt size
    user = (
        f"Item: {item}\n"
        f"Example responses (one per line):\n- " + "\n- ".join(sample)
    )

    resp = call_with_retries(lambda: client.responses.create(
        model=MODEL,
        temperature=0,
        input=[
            {"role": "system", "content": SYSTEM_DISCOVER},
            {"role": "user", "content": user}
        ],
        text={"format": {"type": "json_schema", "name": schema["name"], "schema": schema["schema"], "strict": True}}
    ))

    data = json.loads(resp.output_text)
    # De-duplicate while preserving order
    cats = []
    seen = set()
    for c in data["categories"]:
        c2 = c.strip()
        if c2 and c2 not in seen:
            seen.add(c2)
            cats.append(c2)

    # Hard enforce 15–20
    if len(cats) < DISCOVERY_TARGET_MIN:
        # pad deterministically if model under-produces (rare)
        cats += [f"Other / Misc {i}" for i in range(DISCOVERY_TARGET_MIN - len(cats))]
    if len(cats) > DISCOVERY_TARGET_MAX:
        cats = cats[:DISCOVERY_TARGET_MAX]
    return cats

def batch_classify(client: OpenAI, item: str, categories: list[str], batch: list[tuple[int, str]]) -> dict[int, str]:
    schema = {
        "name": "aut_batch_classify",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer"},
                            "category": {"type": "string", "enum": categories}
                        },
                        "required": ["id", "category"]
                    }
                }
            },
            "required": ["items"]
        },
        "strict": True
    }

    # Construct compact input
    lines = [f"{rid}\t{text}" for rid, text in batch]
    user = (
        f"Item: {item}\n"
        f"Allowed categories:\n- " + "\n- ".join(categories) +
        "\n\nLabel each line with exactly one category.\n"
        "Each input line is: <id>\\t<response>\n\n" +
        "\n".join(lines)
    )

    resp = call_with_retries(lambda: client.responses.create(
        model=MODEL,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": SYSTEM_CLASSIFY},
            {"role": "user", "content": user}
        ],
        text={"format": {"type": "json_schema", "name": schema["name"], "schema": schema["schema"], "strict": True}}
    ))

    data = json.loads(resp.output_text)
    out = {}
    for obj in data["items"]:
        out[int(obj["id"])] = obj["category"]
    return out

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. In Terminal: export OPENAI_API_KEY='sk-...'")

    client = OpenAI()

    df = pd.read_csv(IN_PATH)

    df["item"] = df["item"].str.strip().str.lower()

    if "item" not in df.columns or "response" not in df.columns:
        raise ValueError("CSV must have columns: item, response")

    df = df.reset_index(drop=True)
    df["__id"] = df.index.astype(int)

    # Build/collect category sets per item
    item_categories = {}
    for item, g in df.groupby("item"):
        item = str(item).strip()
        if item in ITEM_TO_CATEGORIES:
            item_categories[item] = ITEM_TO_CATEGORIES[item]
        else:
            if not ALLOW_CATEGORY_DISCOVERY:
                raise ValueError(f"Item '{item}' not in ITEM_TO_CATEGORIES and discovery is disabled.")
            examples = g["response"].astype(str).tolist()
            cats = discover_categories(client, item, examples)
            item_categories[item] = cats
            print(f"[DISCOVERED] {item}: {len(cats)} categories")

    # Classify all rows
    results = {}
    for item, g in df.groupby("item"):
        item = str(item).strip()
        cats = item_categories[item]
        ids = g["__id"].tolist()
        texts = g["response"].astype(str).tolist()

        for start in range(0, len(ids), BATCH_SIZE):
            batch = list(zip(ids[start:start+BATCH_SIZE], texts[start:start+BATCH_SIZE]))
            coded = batch_classify(client, item, cats, batch)
            results.update(coded)
            print(f"[{item}] coded {min(start+BATCH_SIZE, len(ids))}/{len(ids)}")

    df["category"] = df["__id"].map(results)
    out = df.drop(columns=["__id"])
    out.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)

    # Optional: save discovered categories for auditability
    with open("aut_item_categories_used.json", "w") as f:
        json.dump(item_categories, f, indent=2)
    print("Saved category sets:", "aut_item_categories_used.json")

if __name__ == "__main__":
    main()
