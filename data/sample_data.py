"""
Step 3 helper — sample labelled training data.

In a real project you would load this from a CSV, database, or API.
Here we hard-code a small but representative dataset to keep the
getting-started experience self-contained.
"""

# (text, intent_label) pairs
TRAINING_DATA: list[tuple[str, str]] = [
    # recommend_product
    ("I'm looking for a product that suits my budget",       "recommend_product"),
    ("What plans do you offer for small businesses?",        "recommend_product"),
    ("Can you suggest something for a team of five?",        "recommend_product"),
    ("Which subscription is best for beginners?",           "recommend_product"),
    ("I need a tool for managing projects cheaply",          "recommend_product"),
    ("Show me your most popular product",                    "recommend_product"),

    # escalate_to_human
    ("I want to speak to a real person",                     "escalate_to_human"),
    ("This is urgent, please connect me to support",         "escalate_to_human"),
    ("Transfer me to a manager",                             "escalate_to_human"),
    ("I need human assistance immediately",                  "escalate_to_human"),
    ("Can I talk to someone on your team?",                  "escalate_to_human"),
    ("I'm very frustrated and need to speak with someone",   "escalate_to_human"),

    # provide_faq
    ("How do I reset my password?",                          "provide_faq"),
    ("Where can I find documentation?",                      "provide_faq"),
    ("What are your business hours?",                        "provide_faq"),
    ("How does billing work?",                               "provide_faq"),
    ("Where do I update my payment method?",                 "provide_faq"),
    ("I need help setting up my account",                    "provide_faq"),

    # collect_feedback
    ("I have some feedback about the product",               "collect_feedback"),
    ("I'd like to leave a review",                           "collect_feedback"),
    ("Here's what I think could be improved",                "collect_feedback"),
    ("I want to rate my experience",                         "collect_feedback"),
    ("How can I submit a suggestion?",                       "collect_feedback"),
    ("I have a complaint I'd like to file",                  "collect_feedback"),
]
