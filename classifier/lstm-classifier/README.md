# LSTM-Classifier

## Model

Please download the model. The model can be placed in either location.

## Usage

The function `predict_keyword_lstm` takes a list of dialog as input and return one of the keywords in the predicted subdomains in `subdomain.json`. 
Usage: predict_keyword_lstm(input: List[str], vocab_path: str, embedding_path: str, model_path: str, subdomain_path:str)
See `example.py` for an example usage.

All possible subdomains:

```json
[
	"restaurant-meal", 
	"restaurant-dessert",
	"restaurant-cooking",
	"restaurant-type",
	"restaurant-eat",
	"restaurant-service",
	"hotel-service",
	"hotel-travel",
	"movie-type",
	"movie-attribute",
	"movie-theater",
	"song-type",
	"song-method",
	"song-performer",
	"song-attributes",
	"transportation-type",
	"transportation-traffic",
	"transportation-others",
	"attraction-location",
	"attraction-type",
	"attraction-others"
]
```
