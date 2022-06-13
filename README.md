# ADL22-Final
This is the final project of a group of students taking 2022 Applied Deep Learning. 

The main goal of this project is to construct a topic transfer BOT that can guide the simulator into saying a specific groups of words which belongs to 6 big domains. 

- We use pretrained models such as T5, GPT2 from [huggingface](https://huggingface.co/models) and then finetuned on our tasks. 
- The main structure of our BOT refers to this paper : [Target-Guided Dialogue Response Generation Using Commonsense and Data Augmentation](https://arxiv.org/pdf/2205.09314.pdf)
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
python simulator.py --device cuda:0 --split test
```

## BOT
### <a style="color:red">GPT2bot</a> :+1: 

![](https://i.imgur.com/xc9XPlo.png)
reference : https://arxiv.org/pdf/2205.09314.pdf

### <a style="color:red">GPT5bot</a>

### <a style="color:red">T5bot</a>

## Experiments