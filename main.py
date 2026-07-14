#imported files (data and intent mapping)
from data import MOCK_PRODUCTS
from intent_mapping import extract_intent, ProductQueryIntent

def get_recommendations(intent: ProductQueryIntent, products: list) -> list:
    filtered_results = []
    
    for product in products:
        # 1. Category Filter 
        if intent.category and product["category"].lower() != intent.category.lower():
            continue
            
        # 2. Color Filter
        if intent.preferred_color:
            product_colors_lower = [c.lower() for c in product["colors"]]
            if intent.preferred_color.lower() not in product_colors_lower:
                continue
                
        # 3. Size Filter 
        if intent.preferred_size:
            product_sizes_lower = [str(s).lower() for s in product["sizes"]]
            if intent.preferred_size.lower() not in product_sizes_lower:
                continue
                
        # 4. Price Filter 
        if intent.max_price:
            if product["price"] > intent.max_price:
                continue
                
        filtered_results.append(product)
        
    return filtered_results

def run_ai_search(query: str):
    print(f"\n👉 User Query: '{query}'")
    
    parsed_intent = extract_intent(query)
    print(f"🔍 AI Parsed Intent:")
    print(f"   - Intent Type: {parsed_intent.intent_type}")
    print(f"   - Category: {parsed_intent.category}")
    print(f"   - Preferred Color: {parsed_intent.preferred_color}")
    print(f"   - Preferred Size: {parsed_intent.preferred_size}")
    print(f"   - Max Price Limit: {parsed_intent.max_price}")
    print(f"   - Product Type: {parsed_intent.product_type}")
    

    matches = get_recommendations(parsed_intent, MOCK_PRODUCTS)
    
    print(f"🛍️ Recommendations Found ({len(matches)}):")
    if matches:
        for idx, item in enumerate(matches, 1):
            print(f"   {idx}. {item['name']} | Price: {item['price']} | Sizes: {item['sizes']} | Colors: {item['colors']}")
    else:
        print("No products were found matching your criteria. Please try adjusting your search filters.")
    print("-" * 50)

if __name__ == "__main__":

    #Test Cases on the dataset
   
    run_ai_search("I want to buy a maroon dress for women")
    
    run_ai_search("Do you have men's denim jeans in size 32?")
    
    run_ai_search("Show me kids clothing under 2000 rupees")