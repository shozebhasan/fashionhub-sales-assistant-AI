
import re
from typing import Optional

#making categories, colors, sizes, product types, and cities more canonical for better matching

CATEGORY_SYNONYMS = {
    "women": "Women's Collection", "woman": "Women's Collection",
    "womens": "Women's Collection", "women's": "Women's Collection",
    "ladies": "Women's Collection", "lady": "Women's Collection",
    "female": "Women's Collection", "girls": "Women's Collection",
    "girl": "Women's Collection",

    "men": "Men's Collection", "man": "Men's Collection",
    "mens": "Men's Collection", "men's": "Men's Collection",
    "gents": "Men's Collection", "gentlemen": "Men's Collection",
    "male": "Men's Collection", "boys": "Men's Collection",
    "boy": "Men's Collection",

    "kids": "Kids Collection", "kid": "Kids Collection",
    "children": "Kids Collection", "child": "Kids Collection",
    "childrens": "Kids Collection", "children's": "Kids Collection",
}

COLOR_SYNONYMS = {
    "navy": "Navy Blue", "navyblue": "Navy Blue", "navy-blue": "Navy Blue",
    "grey": "Grey", "gray": "Grey",
    "charcoal": "Charcoal Grey", "charcoalgrey": "Charcoal Grey", "charcoalgray": "Charcoal Grey",
    "maroon": "Maroon",
    "wine": "Wine",
    "beige": "Beige",
    "olive": "Olive Green", "olivegreen": "Olive Green",
    "mint": "Mint Green", "mintgreen": "Mint Green",
    "powderblue": "Powder Blue", "powder-blue": "Powder Blue",
    "lightblue": "Light Blue", "light-blue": "Light Blue",
    "darkblue": "Dark Blue", "dark-blue": "Dark Blue",
    "royalblue": "Royal Blue", "royal-blue": "Royal Blue",
    "lavender": "Lavender",
    "peach": "Peach",
    "gold": "Gold",
    "black": "Black", "white": "White", "red": "Red", "blue": "Blue",
    "pink": "Pink", "yellow": "Yellow", "green": "Green", "brown": "Brown",
}

SIZE_SYNONYMS = {
    "freesize": "Free Size", "free-size": "Free Size", "onesize": "Free Size",
}
LETTER_SIZES = {"xs", "s", "m", "l", "xl", "xxl", "xxxl"}

PRODUCT_TYPE_SYNONYMS = {
    "tshirt": "T-Shirt", "t-shirt": "T-Shirt", "tee": "T-Shirt", "tees": "T-Shirt",
    "shirt": "Shirt", "shirts": "Shirt",
    "hoodie": "Hoodie", "hoodies": "Hoodie", "hoody": "Hoodie",
    "jacket": "Jacket", "jackets": "Jacket",
    "jeans": "Jeans", "denim": "Jeans", "jean": "Jeans",
    "trouser": "Trousers", "trousers": "Trousers", "pant": "Trousers", "pants": "Trousers",
    "dress": "Dress", "dresses": "Dress", "gown": "Dress", "maxi": "Dress",
    "kurta": "Kurta", "kurti": "Kurti", "kurtis": "Kurti",
    "polo": "Polo", "polos": "Polo",
    "sweatshirt": "Sweatshirt", "sweatshirts": "Sweatshirt",
    "shorts": "Shorts", "short": "Shorts",
    "shoes": "Shoes", "shoe": "Shoes", "sneakers": "Shoes", "footwear": "Shoes",
    "blazer": "Blazer", "blazers": "Blazer",
    "shawl": "Shawl", "dupatta": "Dupatta",
}

CITY_SYNONYMS = {
    "lahore": "Lahore", "karachi": "Karachi", "islamabad": "Islamabad",
    "rawalpindi": "Rawalpindi", "faisalabad": "Faisalabad", "multan": "Multan",
    "peshawar": "Peshawar", "quetta": "Quetta", "sialkot": "Sialkot",
    "hyderabad": "Hyderabad", "gujranwala": "Gujranwala", "sukkur": "Sukkur",
}

GREETING_PHRASES = {
    "hi", "hello", "hey", "hiya", "yo", "salam", "assalam o alaikum",
    "assalamualaikum", "asalam o alaikum", "assalamoalaikum",
    "assalamu alaikum", "aoa", "good morning", "good afternoon",
    "good evening", "hi there", "hello there", "salaam",
}

#helpers for cleaning and normalizing text, categories, colors, sizes, product types, and cities

def _clean(text: Optional[str]) -> str:
    """Lowercase, strip punctuation (keep letters/digits/spaces/hyphens)."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_token(text: str) -> str:
    """Like _clean but also strips internal spaces/hyphens, for dict lookups
    like 'light blue' / 'light-blue' -> 'lightblue'."""
    return _clean(text).replace(" ", "").replace("-", "")


def normalize_category(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = _clean_token(value)
    return CATEGORY_SYNONYMS.get(key, value.strip())


def normalize_color(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = _clean_token(value)
    if key in COLOR_SYNONYMS:
        return COLOR_SYNONYMS[key]
    return value.strip().title()


def normalize_size(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip()
    token = _clean_token(raw)   
    spaced = _clean(raw)        

    if token in SIZE_SYNONYMS:
        return SIZE_SYNONYMS[token]
    if token in LETTER_SIZES:
        return token.upper()
    if re.fullmatch(r"\d{2,3}", token):        
        return token
    age_match = re.fullmatch(r"(\d{1,2})-(\d{1,2})y", spaced)  
    if age_match:
        return f"{age_match.group(1)}-{age_match.group(2)}Y"
    return raw


def normalize_city(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = _clean_token(value)
    return CITY_SYNONYMS.get(key, value.strip().title())


def normalize_product_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = _clean_token(value)
    return PRODUCT_TYPE_SYNONYMS.get(key, value.strip().title())

def is_greeting(text: str) -> bool:
    return _clean(text) in GREETING_PHRASES


def match_single_word(word: str):
    """
    Try to resolve a single-token query (no spaces) to a full intent
    without calling the LLM. Returns a ProductQueryIntent or None if the
    token doesn't match anything we recognise (caller falls back).

    Import of ProductQueryIntent is deferred to avoid a circular import
    with intent_mapping.py.
    """
    from intent_mapping import ProductQueryIntent  

    key = _clean_token(word)     # dict/letter/digit lookups (spaces+hyphens stripped)
    spaced = _clean(word)        # regex needing the hyphen intact (e.g. age sizes)
    if not key:
        return None

    if key in COLOR_SYNONYMS:
        return ProductQueryIntent(intent_type="color_inquiry", preferred_color=COLOR_SYNONYMS[key])

    if key in LETTER_SIZES:
        return ProductQueryIntent(intent_type="size_inquiry", preferred_size=key.upper())
    if key in SIZE_SYNONYMS:
        return ProductQueryIntent(intent_type="size_inquiry", preferred_size=SIZE_SYNONYMS[key])
    if re.fullmatch(r"\d{2,3}", key):
        return ProductQueryIntent(intent_type="size_inquiry", preferred_size=key)
    age_match = re.fullmatch(r"(\d{1,2})-(\d{1,2})y", spaced)
    if age_match:
        return ProductQueryIntent(
            intent_type="size_inquiry",
            preferred_size=f"{age_match.group(1)}-{age_match.group(2)}Y",
        )

    if key in PRODUCT_TYPE_SYNONYMS:
        return ProductQueryIntent(intent_type="product_search", product_type=PRODUCT_TYPE_SYNONYMS[key])

    if key in CATEGORY_SYNONYMS:
        return ProductQueryIntent(intent_type="product_search", category=CATEGORY_SYNONYMS[key])

    if key in CITY_SYNONYMS:
        return ProductQueryIntent(intent_type="delivery_inquiry", city=CITY_SYNONYMS[key])

    return None