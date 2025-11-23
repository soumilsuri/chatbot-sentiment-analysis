"""
Predefined test scenarios for the chatbot application.
Used to simulate conversations and test sentiment analysis features.
"""

TEST_SCENARIOS = {
    "Positive Flow": {
        "description": "A customer expressing satisfaction with the service.",
        "messages": [
            {"role": "user", "content": "Hi, I just wanted to say that your service is amazing!"},
            {"role": "assistant", "content": "Thank you so much! We're thrilled to hear that you're enjoying our service. Is there anything specific you liked?"},
            {"role": "user", "content": "Yes, the response time is incredibly fast and the support team is very helpful."},
            {"role": "assistant", "content": "That's great to hear! We pride ourselves on our quick support. Do you have any other feedback for us?"},
            {"role": "user", "content": "No, everything is perfect. Keep up the good work!"}
        ]
    },
    "Negative Flow": {
        "description": "A customer expressing frustration and dissatisfaction.",
        "messages": [
            {"role": "user", "content": "I am extremely disappointed with my recent order."},
            {"role": "assistant", "content": "I'm very sorry to hear that. Could you please tell me what went wrong with your order?"},
            {"role": "user", "content": "It arrived late and the package was damaged. This is unacceptable."},
            {"role": "assistant", "content": "I completely understand your frustration. That is certainly not the experience we want for our customers. Do you have your order number?"},
            {"role": "user", "content": "I don't care about the order number right now, I just want a refund. This service is terrible."}
        ]
    },
    "Mixed Flow": {
        "description": "A conversation with both positive and negative elements.",
        "messages": [
            {"role": "user", "content": "Hi, I have a question about my subscription."},
            {"role": "assistant", "content": "Hello! I'd be happy to help with your subscription. What would you like to know?"},
            {"role": "user", "content": "I really like the premium features, they are very useful."},
            {"role": "assistant", "content": "That's wonderful! We designed them to be as helpful as possible."},
            {"role": "user", "content": "However, I think the price is a bit too high for what it offers."},
            {"role": "assistant", "content": "I understand your concern about the pricing. We try to offer competitive rates, but I'll pass your feedback to our team."},
            {"role": "user", "content": "Thanks. Aside from the price, the app is good."}
        ]
    },
    "Emotional Rollercoaster": {
        "description": "A conversation shifting between different emotions (Joy, Anger, Sadness).",
        "messages": [
            {"role": "user", "content": "I finally got the promotion I wanted! I'm so happy!"},
            {"role": "assistant", "content": "Congratulations! That is fantastic news. You must be very proud."},
            {"role": "user", "content": "But then I found out my best friend is moving away. It's so sad."},
            {"role": "assistant", "content": "Oh, I'm so sorry to hear that. It's always hard when close friends move away."},
            {"role": "user", "content": "And to make matters worse, my car broke down on the way home! I'm so angry right now!"}
        ]
    }
}
