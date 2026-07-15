#importing the extract_intent function from the intent_mapping module to be used in the test cases
from intent_mapping import extract_intent

#Test Cases

CASES = [
    # greeting
    ("Hi", "greeting", {}),
    ("Hello", "greeting", {}),
    ("Assalam o Alaikum", "greeting", {}),

    # product_search
    ("Show black dresses", "product_search", {"preferred_color": "Black", "product_type": "Dress"}),
    ("Men's shirts please", "product_search", {"category": "Men's Collection", "product_type": "Shirt"}),
    

    # recommendation
    ("What's trending right now?", "recommendation", {}),
    ("Best selling products this week", "recommendation", {}),
    

    # size_inquiry 
    ("Do you have this in medium?", "size_inquiry", {"preferred_size": "M"}),
    

    # color_inquiry 
    ("What colors are available?", "color_inquiry", {}),
    ("Do you have this dress in black?", "color_inquiry", {"preferred_color": "Black", "product_type": "Dress"}),
    

    # price_inquiry 
    ("What's the price of this?", "price_inquiry", {}),
    

    # delivery_inquiry 
    ("How much are delivery charges?", "delivery_inquiry", {}),
    

    # exchange_return 
    ("What's your return policy?", "exchange_return", {}),
    

    # order_placement 
    ("How can I place an order?", "order_placement", {}),
    

    # order_tracking 
    ("Can you track my order?", "order_tracking", {}),
    ("Where is my parcel?", "order_tracking", {}),

    # complaint 
    ("This is really bad quality material", "complaint", {}),
    

    # price range parsing
    ("Show me kurtis between 2000 and 5000", "product_search",
        {"product_type": "Kurti", "min_price": 2000, "max_price": 5000}),
    
]


def run(verbose: bool = True):
    total = len(CASES)
    intent_correct = 0
    entity_correct = 0
    entity_checked = 0
    failures = []

    for query, expected_intent, expected_fields in CASES:
        result = extract_intent(query)

        intent_ok = result.intent_type == expected_intent
        if intent_ok:
            intent_correct += 1
        else:
            failures.append((query, "intent", expected_intent, result.intent_type))

        for field, expected_value in expected_fields.items():
            entity_checked += 1
            actual_value = getattr(result, field)
            if actual_value == expected_value:
                entity_correct += 1
            else:
                failures.append((query, f"entity:{field}", expected_value, actual_value))

        if verbose:
            mark = "OK " if intent_ok else "FAIL"
            print(f"[{mark}] {query!r:55} expected={expected_intent:18} got={result.intent_type:18}")

    intent_accuracy = intent_correct / total
    entity_accuracy = (entity_correct / entity_checked) if entity_checked else 1.0

    print("\n" + "=" * 70)
    print(f"Intent accuracy: {intent_correct}/{total} = {intent_accuracy:.1%}")
    print(f"Entity accuracy: {entity_correct}/{entity_checked} = {entity_accuracy:.1%}")
    print("=" * 70)

    if failures:
        print("\nFailures:")
        for query, kind, expected, actual in failures:
            print(f"  - {query!r} [{kind}] expected={expected!r} got={actual!r}")

    return intent_accuracy, entity_accuracy, failures


def test_intent_accuracy_meets_target():
    """Pytest entry point. NOTE: this makes real Gemini API calls."""
    intent_accuracy, _, failures = run(verbose=False)
    assert intent_accuracy >= 0.90, (
        f"Intent accuracy {intent_accuracy:.1%} is below the 90% acceptance target. "
        f"Failures: {failures}"
    )


def test_never_raises_on_empty_input():
    result = extract_intent("")
    assert result.intent_type == "fallback"
    result = extract_intent("   ")
    assert result.intent_type == "fallback"
    result = extract_intent(None)
    assert result.intent_type == "fallback"


if __name__ == "__main__":
    run(verbose=True)