# ADL22-Final

## Train T5
1. Go to directory ***T5***
2. modify config.yml according to your training environment
```
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file config.yml train.py
```

## Chat with Simulator
```
[Under directory T5]
./download.sh
python simulator.py --device cuda:0 --split test --bot GPT2bot
```

## Domain Classifier
:::info

:::

## Metrics

:::info

:::

## BOT
### <a style="color:red">GPT2bot</a> :+1: 

![](https://i.imgur.com/xc9XPlo.png)
reference : https://arxiv.org/pdf/2205.09314.pdf

### <a style="color:red">GPT5bot</a>

### <a style="color:red">T5bot</a>

## Experiments