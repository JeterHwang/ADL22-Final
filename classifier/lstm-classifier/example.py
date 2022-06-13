from predict_keyword_lstm import predict_keyword_lstm
import random

rndseed = 48763
random.seed(rndseed)
INPUT = [
        "Chocolate is my favorite too! It's so sweet and delicious. It's also one of the most popular desserts in the world.",
        "I like that too! I also like to make my own chocolate by grinding up cocoa beans.",
        "That sounds really good, I'll have to try that sometime. Do you have a favorite dessert?"
     ]
keyword = predict_keyword_lstm(INPUT, "./data/vocab.pkl", "./data/embeddings.pt", "./model/model.pkl", "subdomain.json")
print("result:", keyword)