# libraries
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from nlu_normalizer import (
    normalize_category,
    normalize_color,
    normalize_size,
    normalize_city,
    normalize_product_type,
    match_single_word,
    is_greeting,
)

load_dotenv()


# Canonical intents
CANONICAL_INTENTS = (
    "greeting",
    "product_search",
    "recommendation",
    "size_inquiry",
    "color_inquiry",
    "price_inquiry",
    "delivery_inquiry",
    "exchange_return",
    "order_placement",
    "order_tracking",
    "complaint",
)

CONFIDENCE_THRESHOLD = 0.55  # below this, we override to "fallback"

# PUBLIC output contract - Phase 2 & 3 depend on exactly this shape.
# Make sure to add exactly the same names as written below.

class ProductQueryIntent(BaseModel):
    intent_type: Literal[
        "greeting", "product_search", "recommendation", "size_inquiry",
        "color_inquiry", "price_inquiry", "delivery_inquiry", "exchange_return",
        "order_placement", "order_tracking", "complaint", "fallback",
    ] = Field(description="One of the 11 canonical intents, or 'fallback' if the query could not be classified confidently.")
    category: Optional[str] = Field(None, description="Women's Collection / Men's Collection / Kids Collection.")
    preferred_color: Optional[str] = Field(None, description="Canonical color name, e.g. 'Navy Blue'.")
    preferred_size: Optional[str] = Field(None, description="S/M/L/XL, waist size like '32', age size like '4-5Y', or 'Free Size'.")
    min_price: Optional[float] = Field(None, description="Lower budget bound, if mentioned.")
    max_price: Optional[float] = Field(None, description="Upper budget bound, if mentioned.")
    product_type: Optional[str] = Field(None, description="T-Shirt, Hoodie, Jeans, Dress, Kurti, Blazer, Shoes, etc.")
    city: Optional[str] = Field(None, description="Delivery city, e.g. 'Lahore'.")
    quantity: Optional[int] = Field(None, description="Number of items requested, if mentioned.")


# INTERNAL model the LLM actually fills in. Same fields as the public
# contract, plus a self-reported confidence score used to decide fallback.
# This never leaves extract_intent() - callers only ever see ProductQueryIntent.

class _LLMIntentOutput(BaseModel):
    intent_type: Literal[
        "greeting", "product_search", "recommendation", "size_inquiry",
        "color_inquiry", "price_inquiry", "delivery_inquiry", "exchange_return",
        "order_placement", "order_tracking", "complaint",
    ] = Field(description="Exactly one of the 11 canonical intent types. Always pick your best guess - never leave this blank.")
    confidence: float = Field(
        description="Your confidence in this classification, from 0.0 (pure guess) to 1.0 (certain). "
                    "Be honest - use a low score for ambiguous, off-topic, or unclear messages."
    )
    category: Optional[str] = Field(None, description="Women's Collection / Men's Collection / Kids Collection, if detected.")
    preferred_color: Optional[str] = Field(None, description="Color mentioned, in the user's own wording.")
    preferred_size: Optional[str] = Field(None, description="Size mentioned, in the user's own wording.")
    min_price: Optional[float] = Field(None, description="Lower budget bound, e.g. from 'between 2000 and 5000' -> 2000.")
    max_price: Optional[float] = Field(None, description="Upper budget bound, e.g. from 'under 3000' -> 3000, 'cheapest' -> leave null.")
    product_type: Optional[str] = Field(None, description="Garment/product type mentioned, in the user's own wording.")
    city: Optional[str] = Field(None, description="City mentioned for delivery, in the user's own wording.")
    quantity: Optional[int] = Field(None, description="Quantity mentioned, if any.")


# System prompt - 11 intents, 2-3 examples each, plus entity + confidence rules

SYSTEM_PROMPT = """
You are an intent extraction engine for an apparel e-commerce store (FashionHub)
that talks to customers over Instagram DM / WhatsApp.

Your only task is to analyze the user's message and extract structured information.
Never invent information. If something is not explicitly mentioned, return null.

## 1. Classify intent_type into exactly one of these 11 intents:

- greeting
  "Hi", "Hello", "Assalam o Alaikum"

- product_search
  Use when the user is looking for a specific product or wants to buy/find an item.
  "Show black dresses", "Men's shirts", "Shoes under Rs 3000"

- recommendation
  Use when the user asks for recommendations, suggestions, options, or trends.
  "Trending outfits", "Best selling products", "What should I wear?"

- size_inquiry
  "Do you have medium?", "Is XL available?", "Show size chart"

- color_inquiry
  "Available colors?", "Do you have black?", "Beige available?"

- price_inquiry
  "Price?", "Any discount?", "Sale available?", "Cheapest products"

- delivery_inquiry
  "Delivery charges?", "Same day delivery?", "Delivery to Lahore?"

- exchange_return
  "Return policy?", "Exchange available?", "Damaged item received"

- order_placement
  "How can I place an order?", "I want to buy this"

- order_tracking
  "Track my order", "Where is my parcel?", "My tracking ID is 12345"

- complaint
  "This is bad quality", "Wrong item sent"

## 2. Extract these entities (all optional - null when absent):

- category: must be exactly "Women's Collection", "Men's Collection", or "Kids Collection" if
  detectable (map "ladies"->Women's, "gents"/"men"->Men's, "kids"/"children"->Kids's). Else null.
- preferred_color: the color mentioned, in the customer's own words.
- preferred_size: S/M/L/XL, waist sizes like "32"/"34", age sizes like "4-5Y", or "Free Size".
- min_price / max_price: support phrasing like "under X" (-> max_price=X), "below X" (-> max_price=X),
  "between X and Y" (-> min_price=X, max_price=Y). "Cheapest" alone is not a numeric bound - leave
  both null and rely on intent_type=price_inquiry.
- product_type: T-Shirt, Hoodie, Jeans, Dress, Kurti, Blazer, Shoes, Jacket, Shirt, Trousers, etc.
- city: delivery city such as Lahore, Islamabad, Karachi, etc.
- quantity: number of items requested, if mentioned.

## 3. Ignore greetings/politeness/emojis mixed into a longer message when they are not the whole
   point of the message (e.g. "Hi, do you have this in black?" -> color_inquiry, not greeting).

## 4. confidence: report how sure you are of intent_type, from 0.0 to 1.0. Use a low score
   (below 0.5) for messages that are ambiguous, off-topic, or don't clearly fit any of the 11
   intents - do not force a confident-sounding guess.

Return only the structured output.
"""

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
structured_llm = llm.with_structured_output(_LLMIntentOutput)


def _fallback(intent_type: str = "fallback") -> ProductQueryIntent:
    return ProductQueryIntent(intent_type=intent_type)


def _to_public(raw: _LLMIntentOutput) -> ProductQueryIntent:
    """Normalize + strip the internal confidence field before returning."""
    return ProductQueryIntent(
        intent_type=raw.intent_type,
        category=normalize_category(raw.category),
        preferred_color=normalize_color(raw.preferred_color),
        preferred_size=normalize_size(raw.preferred_size),
        min_price=raw.min_price,
        max_price=raw.max_price,
        product_type=normalize_product_type(raw.product_type),
        city=normalize_city(raw.city),
        quantity=raw.quantity,
    )


def extract_intent(user_query: str) -> ProductQueryIntent:
    """Single public entry point. Never raises - always returns a
    ProductQueryIntent, falling back to intent_type='fallback' on empty
    input, unrecognised single-word input, low-confidence LLM output, or
    any error talking to the LLM."""

    if not user_query or not user_query.strip():
        return _fallback()

    cleaned = user_query.strip()

    # Rule-based fast path: no LLM call
    if is_greeting(cleaned):
        return ProductQueryIntent(intent_type="greeting")

    if " " not in cleaned:
        fast_result = match_single_word(cleaned)
        if fast_result is not None:
            return fast_result
        # Unrecognised single word - not worth an LLM call either
        return _fallback()

    # LLM path
    try:
        raw = structured_llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("user", cleaned),
        ])
    except Exception:
        return _fallback()

    if raw is None or raw.confidence < CONFIDENCE_THRESHOLD:
        return _fallback()

    return _to_public(raw)