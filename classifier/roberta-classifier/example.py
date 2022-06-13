from predict_keyword import predict_keyword_roberta

test_dialog_list = [
        "Chocolate is my favorite too! It's so sweet and delicious. It's also one of the most popular desserts in the world.",
        "I like that too! I also like to make my own chocolate by grinding up cocoa beans.",
        "That sounds really good, I'll have to try that sometime. Do you have a favorite dessert?"
    ]
test_subdomain = "restaurant-dessert"
result = predict_keyword_roberta(test_dialog_list)
print('result:', result)