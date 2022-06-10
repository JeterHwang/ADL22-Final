import json
import torch
import random
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5bot(torch.nn.Module):
    def __init__(self, stage1_model, stage2_model, tokenizer, keywords):
        super(T5bot, self).__init__()
        self.model1 = stage1_model
        self.model2 = stage2_model
        self.tokenizer = tokenizer
        self.keywords = keywords

        self.tokenizer.add_tokens(['@', '<s>', '</s>'])
        self.target = None

    @staticmethod
    def from_pretrained(model1_path, model2_path, tokenizer_path, keywords_path):
        print('----- Start Loading Pretrained Models -----')
        model1 = T5ForConditionalGeneration.from_pretrained(model1_path)
        model2 = T5ForConditionalGeneration.from_pretrained(model2_path)
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        keywords = json.loads(keywords_path.read_text())
        print('----- Finish Loading Pretrained Models -----')
        return T5bot(model1, model2, tokenizer, keywords)

    def _generate(self, input_ids, attention_mask, stage, max_target_len=128):
        if stage == 1:
            model = self.model1
        elif stage == 2:
            model = self.model2
        else:
            raise NotImplementedError
        model = model.eval().to('cuda')
        decoded_result = []
        if stage == 1:
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_len,
                num_beams=10, 
                no_repeat_ngram_size=2, 
                early_stopping=True
            )
        else:
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_len,
                do_sample=True,
                top_k=50, 
                top_p=1, 
            )

        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        for pred in decoded_preds:
            decoded_result.append(pred.strip())
        return decoded_result
    
    def choose_target(self):
        domain_list = self.keywords[random.choice(['restaurant', 'hotel', 'movie', 'song', 'transportation', 'attraction'])]
        self.target = random.choice(domain_list)
    
    def generate(self, source, max_input_length=512):
        input_text = 'context : ' + source + ' @ path_tailentity : ' + self.target
        # print(input_text)
        model1_input = self.tokenizer(
            [input_text], 
            max_length=max_input_length, 
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors="pt",
        ).to('cuda')
        model1_output = self._generate(
            model1_input['input_ids'], 
            model1_input['attention_mask'], 
            1
        )[0]
        # print(model1_output)
        model2_input = self.tokenizer(
            [input_text + ' @ path : ' + model1_output],
            max_length=max_input_length, 
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors="pt",
        ).to('cuda')
        model2_output = self._generate(
            model2_input['input_ids'], 
            model2_input['attention_mask'], 
            2
        )[0]
        return model2_output