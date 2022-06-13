from predict_keyword_roberta import predict_keyword_roberta

test_dialog_list = [
        "I'm sorry to hear that. Do you have any other hobbies that you enjoy doing?",
        "I do enjoy fishing, but it's been a while since I've been able to do it.",
        "It's okay, I'm sure I'll get back into it soon. What do you like to do instead?"
      ]
test_subdomain = "song-method"
result = predict_keyword_roberta(test_dialog_list)
print('result:', result)