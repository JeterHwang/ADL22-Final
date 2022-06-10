# ADL22-Final

## Train T5
1. Go to directory ***T5***
2. modify config.yml according to your training environment
```
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file config.yml train.py
```
