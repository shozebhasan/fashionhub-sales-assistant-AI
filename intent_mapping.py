#libraries
from typing import  Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class ProductQueryIntent(BaseModel):
    intent_type: str = Field(
        description="Must be 'search_product' (if searching specific item) or 'recommend_category' (if asking general recommendations/options)."
    )
    category: Optional[str] = Field(
        None, 
        description="Category: Must be 'Women's Collection', 'Men's Collection', or 'Kids Collection' if detected, otherwise None."
    )
    preferred_color: Optional[str] = Field(
        None, 
        description="Extract the color mentioned (e.g., 'Red', 'Black', 'Maroon', 'Blue')."
    )
    preferred_size: Optional[str] = Field(
        None, 
        description="Extract the size (e.g., 'S', 'M', 'L', 'XL' or waist sizes like '32' or age sizes like '4-5Y')."
    )
    max_price: Optional[float] = Field(
        None, 
        description="Extract any maximum budget or price mentioned by the user."
    )
    product_type: Optional[str] = Field(
    None,
    description="Extract the product type such as T-Shirt, Shirt, Hoodie, Jacket, Jeans, Trouser, Dress, Kurta, Polo, Sweatshirt, Shorts, etc."
   )

# Using GPT-4o-mini model for outputs.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

structured_llm = llm.with_structured_output(ProductQueryIntent)

#System prompt
def extract_intent(user_query: str) -> ProductQueryIntent:
    system_prompt = """
You are an intent extraction engine for an apparel e-commerce store.

Your only task is to analyze the user's message and extract structured information.

Rules:

1. Determine the intent_type:
   - "search_product"
     Use when the user is looking for a specific product or wants to buy/find an item.
     Examples:
     - "Show me a black hoodie"
     - "I need a red t-shirt"
     - "Find blue jeans under 3000"

   - "recommend_category"
     Use when the user asks for recommendations, suggestions, options, trends, or best products.
     Examples:
     - "Recommend me some hoodies"
     - "What should I wear in winter?"
     - "Show best shirts"

2. Category mapping:
   Only return one of these exact values:
   - Women's Collection
   - Men's Collection
   - Kids Collection

   If gender/category is not mentioned or cannot be inferred, return null.

3. preferred_color:
   Extract only the requested color.
   Examples:
   Black, White, Navy Blue, Maroon, Olive Green.

   If no color is mentioned, return null.

4. preferred_size:
   Extract clothing size if present.
   Examples:
   XS
   S
   M
   L
   XL
   XXL
   32
   34
   4-5Y

   If absent, return null.

5. max_price:
   Extract the maximum budget only.

   Examples:
   "under 3000" -> 3000
   "below 5000" -> 5000
   "less than 2500" -> 2500

   If no budget exists, return null.

6. Ignore:
   - Greetings
   - Polite words
   - Emojis
   - Extra conversation

7. Never invent information.
   If something is not explicitly mentioned, return null.

Return only the structured output.
"""
    
    # calling LLM
    response = structured_llm.invoke([
        ("system", system_prompt),
        ("user", user_query)
    ])
    return response