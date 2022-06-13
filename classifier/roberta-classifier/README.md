# RoBERTa-Classifier

## Model

The model including tokenizer in default should be saved in directory `roberta_model`.

## Usage

The function `predict_keyword_roberta` takes a list of dialog as input and return one of subdomains in `subdomain.json`. See `example.py` for an example use.

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



