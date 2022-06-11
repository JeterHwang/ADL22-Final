import json
import torch
import random
import math
import torch.nn.functional as F
from tqdm import tqdm, tnrange
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
from utils import connect_entities, get_verbnouns, check_overlap, idf_score, get_min_path, get_filtered_paths

extra_stopwords = ["lot", "person", "have", "not", "also", "very", "often", "however", "too", "usually", "really", "early", "never", "always", "sometimes", "together", "likely", "simply", "generally", "instead", "actually", "again", "rather", "almost", "especially", "ever", "quickly", "probably", "already", "below", "directly", "therefore", "else", "thus", "easily", "eventually", "exactly", "certainly", "normally", "currently", "extremely", "finally", "constantly", "properly", "soon", "specifically", "ahead", "daily", "highly", "immediately", "relatively", "slowly", "fairly", "primarily", "completely", "ultimately", "widely", "recently", "seriously", "frequently", "fully", "mostly", "naturally", "nearly", "occasionally", "carefully", "clearly", "essentially", "possibly", "slightly", "somewhat", "equally", "greatly", "necessarily", "personally", "rarely", "regularly", "similarly", "basically", "closely", "effectively", "initially", "literally", "mainly", "merely", "gently", "hopefully", "originally", "roughly", "significantly", "totally", "twice", "elsewhere", "everywhere", "obviously", "perfectly", "physically", "successfully", "suddenly", "truly", "virtually", "altogether", "anyway", "automatically", "deeply", "definitely", "deliberately", "hardly", "readily", "terribly", "unfortunately", "forth", "briefly", "moreover", "strongly", "honestly", "previously", "as", "there", "when", "how", "so", "up", "out", "no", "only", "well", "then", "first", "where", "why", "now", "around", "once", "down", "off", "here", "away", "today", "far", "quite", "later", "above", "yet", "maybe", "otherwise", "near", "forward", "somewhere", "anywhere", "please", "forever", "somehow", "absolutely", "abroad", "yeah", "nowhere", "the", "to", "in", "on", "by", "more", "about", "such", "through", "new", "just", "any", "each", "much", "before", "between", "free", "right", "best", "since", "both", "sure", "without", "back", "better", "enough", "lot", "small", "though", "less", "little", "under", "next", "hard", "real", "left", "least", "short", "last", "within", "along", "lower", "TRUE", "bad", "across", "clear", "easy", "full", "close", "late", "proper", "fast", "wide", "item", "wrong", "ago", "behind", "quick", "straight", "direct", "extra", "pretty", "overall", "alone", "bright", "flat", "whatever", "slow", "clean", "fresh", "whenever", "cheap", "thin", "cool", "fair", "fine", "smooth", "FALSE", "thick", "nearby", "wild", "apart", "none", "strange", "aside", "super", "ill", "honest", "ok", "thanks"]

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

class GPT5bot(torch.nn.Module):
    def __init__(self, stage1_model, tokenizer1, stage2_model, tokenizer2, keywords, gutenberg_idf, relation2text):
        super(T5bot, self).__init__()
        self.GPT2          =  stage1_model
        self.GPT2tokenizer = tokenizer1
        self.T5            = stage2_model
        self.T5tokenizer   = tokenizer2
        self.keywords      = keywords
        self.gutenberg_idf = gutenberg_idf
        self.relation2text = relation2text

        self.T5tokenizer.add_tokens(['@', '<s>', '</s>'])
        self.target = None

    @staticmethod
    def from_pretrained(model2_path, tokenizer2_path, commensense_model_path, keywords_path, rel2text_path, counts_path):
        print('----- Start Loading GPT-2 -----')
        lm_type = 'gpt2'
        config = GPT2Config.from_pretrained(lm_type)
        tokenizer = GPT2Tokenizer.from_pretrained(lm_type)
        tokenizer.add_tokens(['<PAD>'])
        tokenizer.add_tokens(['<SEP>'])
        tokenizer.add_tokens(['<END>'])
        tokenizer.add_tokens(['<contains>'])
        tokenizer.add_tokens(['<final>'])

        #comment below if there is a size mismatch error. there were nt added in first few models
        with open(rel2text_path, 'r') as fr:
            relation2text = json.load(fr)
            relation2text =  {k.lower(): v for k, v in relation2text.items()}
        relation_keys = relation2text.keys()
        relation_token_list = [x.lower() for x in relation_keys] #+ ['_' + x.lower() for x in relation_keys]
        tokenizer.add_tokens(relation_token_list)

        gpt = GPT2Model.from_pretrained(lm_type)
        config.vocab_size = len(tokenizer)
        gpt.resize_token_embeddings(len(tokenizer))
        pretrain_generator_ckpt = commensense_model_path / "checkpoints_6lendict_wcontains/model.ckpt"
        print('loading generator')
        generator = Generator(gpt, config)
        generator.load_state_dict(torch.load(pretrain_generator_ckpt, map_location='cpu'))
        print('loaded state dict generator')
        generator = generator.to('cuda')
        print('----- Start Loading T5 -----')
        model2 = T5ForConditionalGeneration.from_pretrained(model2_path)
        tokenizer2 = T5Tokenizer.from_pretrained(tokenizer2_path)
        keywords = json.loads(keywords_path.read_text())
        print('----- Start Loading Gutenberg Counts -----')
        gutenberg_counts = open(counts_path, 'r').readlines()
        gutenberg_counts = [s.strip().split() for s in gutenberg_counts]
        gutenberg_word2cnt = {w:int(c) for c,w in gutenberg_counts }
        gutenberg_idf = {w:(1.0/math.log(1+c)) for w,c in gutenberg_word2cnt.items()} # more frequnt words have low frequency
        print('----- Finish Loading Pretrained Models -----')
        return GPT5bot(generator, tokenizer, model2, tokenizer2, keywords, gutenberg_idf, relation2text)

    def find_path(self, context, target, verbose=False, remove_overlap=True, split_entities_into_multi=True):
        context_keywords = get_verbnouns(context.strip())
        target_keywords = get_verbnouns(target.strip())
        if split_entities_into_multi:
            def _augment(lst):
                for w in lst[:]:
                    tmp = w.strip().split()
                    if len(tmp)>1:
                        lst.extend(tmp)
            # print("earlier : context_words = ", context_words)
            _augment(context_keywords)
            # print("after aug : context_words = ", context_words)
            _augment(target_keywords)
            
        dp = {'context': context, 'target': target}
        for head_entity in context_keywords:
            for tail_entity in target_keywords:
                if tail_entity in ['person'] or head_entity in ['person'] or 'not' in tail_entity:
                    continue

                # check if want to remove head/tail
                # - remove head-tail where they overlap (eat, eat food)
                if remove_overlap and check_overlap(head_entity, tail_entity):
                    continue

                dp['paths'][head_entity + '---' + tail_entity] = dict()
                dpht = dp['paths'][head_entity + '---' + tail_entity]
                paths, scores = connect_entities(
                    head_entity, 
                    tail_entity, 
                    self.GPT2, 
                    temperature=0.7, 
                    num_outs=5, 
                    top_k=0,
                    top_p=0.9
                )
                dpht['headtotail_paths'] = paths
                dpht['headtotail_scores'] = scores
                if verbose:
                    print('\n', head_entity, '->', tail_entity)
                    for i, path in enumerate(paths):
                        print(path, scores[i])
        return dp

    def filter_path(self, dp, is_test=False, apply_reranking=True, apply_ranking_topk=3):
        all_head_tails = []
        for ht in dp['paths'].keys():
            head, tail = ht.split('---')
            all_head_tails.append((head, tail))
        # TODO add reranking and filter logic
        filtered_head_tails = []
        for ht_pair in all_head_tails:
            h,t = ht_pair
            if h in extra_stopwords or t in extra_stopwords:
                continue
            if  type(h) is str and len(h)<2 or len(t)<2:
                continue
            filtered_head_tails.append(ht_pair)
        reranked_head_tails = filtered_head_tails
                
        ##add reranking
        if apply_reranking:
            all_head_tails_scores = [ [x, y, idf_score(x, y, self.gutenberg_idf)] for x, y in reranked_head_tails]
            all_head_tails_scores = sorted(all_head_tails_scores, key=lambda k:-k[2]) # decreasing score
            reranked_head_tails = [ [x,y] for x,y,_ in all_head_tails_scores[:apply_ranking_topk] ] 

        for ht in reranked_head_tails:
            ht = ht[0]+'---'+ht[1]
            head, tail = ht.split('---')
            paths, scores = dp['paths'][ht]['headtotail_paths'], dp['paths'][ht]['headtotail_scores']
                
            if is_test is True:
                path, score = get_min_path(paths, scores, parse_edges=True)
                newdp = {'context':dp['context'], 'target': dp['target']}
                newdp['path'] = path
                newdp['score_path'] = score
                newdp['type'] = 'direct'                     
                newdp['path_headentity'] = head
                newdp['path_tailentity'] = tail
                continue
                
            paths, scores = get_filtered_paths(paths, scores, self.relation2text, parse_edges=True)
            for i, path in enumerate(paths):
                # newdp = copy.deepcopy(dp)
                newdp = {'context':dp['context'], 'target': dp['target']}
                newdp['path'] = path
                newdp['score_path'] = scores[i]
                newdp['type'] = 'direct'
                newdp['path_headentity'] = head
                newdp['path_tailentity'] = tail
        return newdp           

    def generate_sentence(self, input_ids, attention_mask, max_target_len=128):
        model = self.T5
        model = model.eval().to('cuda')
        decoded_result = []
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
    
    def generate(self, source, max_input_length=512):
        dp = self.find_path(source, self.target)
        dp = self.filter_path(dp, is_test=False)
        input_text = 'context : ' + dp['context'] + ' @ path_tailentity : ' + self.target + ' @ path : ' + dp['path']
        T5_input = self.T5tokenizer(
            [input_text],
            max_length=max_input_length, 
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors="pt",
        ).to('cuda')
        T5_output = self.generate_sentence(
            T5_input['input_ids'], 
            T5_input['attention_mask']
        )[0]
        return T5_output

    def choose_target(self):
        domain_list = self.keywords[random.choice(['restaurant', 'hotel', 'movie', 'song', 'transportation', 'attraction'])]
        self.target = random.choice(domain_list)

class Generator(torch.nn.Module):
    def __init__(self, gpt, config, max_len=64, temperature=0.7):
        super(Generator, self).__init__()
        self.gpt = gpt
        self.config = config
        self.max_len = max_len
        # self.temperature = temperature
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), filter_tokens=None,
                              min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size, vocabulary size)
                if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                Make sure we keep at least min_tokens_to_keep per batch example in the output
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if filter_tokens is not None:
            for x in filter_tokens:
                if logits.shape[1] > x:
                    logits[:, x] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def forward_greedy(self, inputs):
        # input: [batch, seq]
        context_len = inputs.size(1)
        generated = inputs
        next_token = inputs
        past = None
        with torch.no_grad():
            for step in range(self.max_len):
                outputs = self.gpt(next_token, past=past)
                hidden = outputs[0][:, -1]
                past = outputs[1]
                next_token_logits = self.lm_head(hidden)
                next_logits, next_token = next_token_logits.topk(k=1, dim=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def sample_seq(self, model, context, length, device, temperature=1, top_k=0, top_p=0.0):
        """ Generates a sequence of tokens
            Args:
                model: gpt/gpt2 model
                context: tokenized text using gpt/gpt2 tokenizer
                length: length of generated sequence.
                device: torch.device object.
                temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        """
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0)
        generated = context
        with torch.no_grad():
            for _ in tnrange(length):
                inputs = {'input_ids': generated}
                outputs = model(
                    **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    def forward(self, inputs, temperature=1, num_outs=1, top_k=0, top_p=0.0):
        # input: [batch, seq]
        if num_outs > 1:
            inputs = inputs.repeat(num_outs, 1)
        context_len = inputs.size(1)
        # print(inputs.shape)
        generated = inputs
        next_token = inputs
        past = None
        probs_arr = [[] for i in range(inputs.shape[0])]
        with torch.no_grad():
            for step in range(self.max_len):
                outputs = self.gpt(next_token, past=past)
                hidden = outputs[0][:, -1]
                past = outputs[1]
                next_token_logits = self.lm_head(hidden)
                next_token_logits = next_token_logits / temperature
                # filtered_logits = self.top_k_top_p_filtering(next_token_logits.squeeze(dim=0), top_k=top_k, top_p=top_p)
                filter_tokens = [50268, 50269]
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, filter_tokens=filter_tokens)
                # next_logits, next_token = next_token_logits.topk(k=1, dim=1)
                softmax_outs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(softmax_outs, num_samples=1)
                probs_cur = [softmax_outs[i, x[0]].item() for i, x in enumerate(next_token)]
                for i, x in enumerate(probs_arr):
                    # print(i, probs_cur[i], '--', probs_arr[i])
                    probs_arr[i] += [probs_cur[i]]
                # print(softmax_outs.shape, next_token.shape, next_token, probs_cur)
                # print(probs_arr)
                # next_token = next_token.unsqueeze(dim=0)
                generated = torch.cat((generated, next_token), dim=1)
        # print(probs_arr)
        return generated, probs_arr
