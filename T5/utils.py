import torch
import math
import spacy
import nltk
import numpy as np
import torch.nn.functional as F
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words
lemmatizer = nltk.WordNetLemmatizer()
# Rule for NP chunk and VB Chunk
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        {<RB.?>*<VB.?>*<JJ>*<VB.?>+<VB>?} # Verbs and Verb Phrases

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...

"""
grammarnoun = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}

"""

grammarverb = r"""
    NBAR:
        {<RB.?>*<VB.?>*<JJ>*<VB.?>+<VB>?} # Verbs and Verb Phrases

    NP:
        {<NBAR>}

"""
# Chunking
# cp = nltk.RegexpParser(grammar)
cpnoun = nltk.RegexpParser(grammarnoun)
cpverb = nltk.RegexpParser(grammarverb)

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()

def get_word_postag(word):
    if pos_tag([word])[0][1].startswith('J'):
        return wordnet.ADJ
    if pos_tag([word])[0][1].startswith('V'):
        return wordnet.VERB
    if pos_tag([word])[0][1].startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    postag = get_word_postag(word)
    word = lemmatizer.lemmatize(word, postag)
    return word

def get_terms(tree):
    for leaf in leaves(tree):
        terms = [normalise(w) for w, t in leaf]
        yield terms

def get_features(document):
    # word tokenizeing and part-of-speech tagger
    tokens = [nltk.word_tokenize(sent) for sent in [document]]
    postag = [nltk.pos_tag(sent) for sent in tokens][0]

    # the result is a tree
    tree = cpnoun.parse(postag)

    terms = get_terms(tree)

    features = []
    for term in terms:
        _term = ''
        for word in term:
            _term += ' ' + word
        features.append(_term.strip())

    tree = cpverb.parse(postag)

    terms = get_terms(tree)

    features_verb = []
    for term in terms:
        _term = ''
        for word in term:
            _term += ' ' + word
        features_verb.append(_term.strip())

    return features, features_verb
    
def get_verbnouns(document):
    extra_stopwords = ["lot", "person", "have", "not", "also", "very", "often", "however", "too", "usually", "really", "early", "never", "always", "sometimes", "together", "likely", "simply", "generally", "instead", "actually", "again", "rather", "almost", "especially", "ever", "quickly", "probably", "already", "below", "directly", "therefore", "else", "thus", "easily", "eventually", "exactly", "certainly", "normally", "currently", "extremely", "finally", "constantly", "properly", "soon", "specifically", "ahead", "daily", "highly", "immediately", "relatively", "slowly", "fairly", "primarily", "completely", "ultimately", "widely", "recently", "seriously", "frequently", "fully", "mostly", "naturally", "nearly", "occasionally", "carefully", "clearly", "essentially", "possibly", "slightly", "somewhat", "equally", "greatly", "necessarily", "personally", "rarely", "regularly", "similarly", "basically", "closely", "effectively", "initially", "literally", "mainly", "merely", "gently", "hopefully", "originally", "roughly", "significantly", "totally", "twice", "elsewhere", "everywhere", "obviously", "perfectly", "physically", "successfully", "suddenly", "truly", "virtually", "altogether", "anyway", "automatically", "deeply", "definitely", "deliberately", "hardly", "readily", "terribly", "unfortunately", "forth", "briefly", "moreover", "strongly", "honestly", "previously", "as", "there", "when", "how", "so", "up", "out", "no", "only", "well", "then", "first", "where", "why", "now", "around", "once", "down", "off", "here", "away", "today", "far", "quite", "later", "above", "yet", "maybe", "otherwise", "near", "forward", "somewhere", "anywhere", "please", "forever", "somehow", "absolutely", "abroad", "yeah", "nowhere", "the", "to", "in", "on", "by", "more", "about", "such", "through", "new", "just", "any", "each", "much", "before", "between", "free", "right", "best", "since", "both", "sure", "without", "back", "better", "enough", "lot", "small", "though", "less", "little", "under", "next", "hard", "real", "left", "least", "short", "last", "within", "along", "lower", "TRUE", "bad", "across", "clear", "easy", "full", "close", "late", "proper", "fast", "wide", "item", "wrong", "ago", "behind", "quick", "straight", "direct", "extra", "pretty", "overall", "alone", "bright", "flat", "whatever", "slow", "clean", "fresh", "whenever", "cheap", "thin", "cool", "fair", "fine", "smooth", "FALSE", "thick", "nearby", "wild", "apart", "none", "strange", "aside", "super", "ill", "honest", "ok", "thanks"]
    document = document.lower()
    document = document.replace("i'm", "i am")
    if document.startswith('i '):
        document = document.replace("i ", 'person ')
    document = document.replace(" i ", ' person ')
    document = document.replace(" he ", ' person ')
    document = document.replace(" she ", ' person ')
    document = document.replace(" they ", ' people ')
    document = document.replace("don't", 'do not')
    document = document.replace("can't", 'can not')
    document = document.replace("won't", 'would not')
    features, features_verb = get_features(document)
    # print(features, features_verb)
    document_words = document.split()
    document_words = [x for x in document_words if
                    x not in ["a", "the", "an", "about", "above", "across", "after", "against", "among", "around",
                                "at", "before", "behind", "below", "beside", "between", "by", "down", "during", "for",
                                "from", "in", "inside", "into", "near", "of", "off", "on", "out", "over", "through",
                                "to", "toward", "under", "up", "with"]]
    noune_indexdict = {}
    for noune in features:
        for i, x in enumerate(document_words):
            if noune in x and len(noune) < len(x) + 3:
                noune_indexdict[noune] = i

    verb_phrases = []
    for verbe in features_verb:
        verbe_index = -1
        for i, x in enumerate(document_words):
            if verbe in x and len(verbe) < len(x) + 3:
                verbe_index = i
        # print(noune_indexdict)
        # verbe_index = document.find(verbe)
        if verbe_index != -1:
            # print(verbe, verbe_index)
            for noune in noune_indexdict:
                # print(verbe, noune, noune_indexdict[noune],verbe_index)
                if noune_indexdict[noune] > verbe_index and noune_indexdict[noune] - verbe_index < 2:
                    # print(verbe, noune)
                    verb_phrases.append(verbe + ' ' + noune)

    features = [x for x in features if x not in ' '.join(verb_phrases)]
    features_verb = [x for x in features_verb if x not in ' '.join(verb_phrases)]
    features_verb = [x.replace('have', '').replace("'ve", '').replace("be", '').strip() for x in features_verb]
    allstopwords = stopwords.union(extra_stopwords)
    res = [x for x in features + features_verb + verb_phrases if x not in allstopwords]
    res = [x for x in res if len(x)>2]
    res = [x.replace('person ', '') if 'person ' in x else x for x in res]
    return res

def prepare_input(head_entity, tail_entity, tokenizer, input_len=32):
    head_entity = head_entity.replace('_', ' ')
    tail_entity = tail_entity.replace('_', ' ')
    input_token = tail_entity + '<SEP>' + head_entity
    input_id = tokenizer.encode(input_token, add_special_tokens=False)[:input_len]
    input_id += [tokenizer.convert_tokens_to_ids('<PAD>')] * (input_len - len(input_id))
    return torch.tensor([input_id], dtype=torch.long)

def connect_entities(head_entity, tail_entity, generator, tokenizer, device, temperature=1, num_outs=1, top_k=0, top_p=1.0):
    gen_input = prepare_input(head_entity, tail_entity, tokenizer)
    gen_input = gen_input.to(device)
    gen_output, prob_arr = generator(gen_input, temperature=temperature, num_outs=num_outs, top_k=top_k, top_p=top_p)
    prob_sum = [-sum(math.log(x) for x in l if x < 1 and x >= 0) for l in prob_arr]
    outs = []
    for gen in gen_output:
        path = tokenizer.decode(gen.tolist(), skip_special_tokens=True)
        path = ' '.join(path.replace('<PAD>', '').split())
        try:
            out = path[path.index('<SEP>') + 6:]
        except:
            print('weird path', path)
            out = path[16:]
        outs.append(out)
    return outs, prob_sum

def check_overlap(head_entity, tail_entity) -> bool:
    w = set(head_entity.strip().split())
    t = set(tail_entity.strip().split())
    if len(w)==0 or len(t)==0:
        return False
    # print("w = ", w, " || t = ", t)
    inter = w.intersection(t)
    if len(inter) > 0:
        # print(" -- True")
        return True
    # print(" -- False")
    return False

def idf_score(x, y, gutenberg_idf, default_idf_val=1.0/(math.log(1+1))):
    x_score = np.max([gutenberg_idf.get(xi, default_idf_val) for xi in x.strip().split()] )
    y_score = np.max([gutenberg_idf.get(yi, default_idf_val) for yi in y.strip().split()] )
    return x_score + y_score

def convert_edgesnames(path, relation2text):
    path_words = path.split()
    new_path = []
    for w in path_words:
        new_word = relation2text.get(w, w)
        new_path.append(new_word)
    # print(path, new_path)
    new_path = ' '.join(new_path)
    new_path = new_path.replace('  ', ' ')
    
    return new_path

def get_path_words(path, relation2textset):
    relations_all = list(relation2textset)
    relations_all.sort()
    for r in relations_all:
        path = path.replace(r, '----')
    path_words = [x.strip() for x in path.split('----')]
    return set(path_words)

def get_filtered_paths(paths, scores, relation2text, parse_edges=True, input_entities=None, type=''):
    relation2textset = set(relation2text.keys())
    scores, paths = zip(*sorted(zip(scores, paths)))
    min_score, max_scores = min(scores), max(scores)
    list_pathsadded = []
    filt_paths, filt_scores = [], []
    # rel_list = ['atlocation', 'capableof', 'causes', 'causesdesire', 'createdby', 'definedas', 'desireof', 'desires', 'hasa', 'hasfirstsubevent', 'haslastsubevent', 'haspaincharacter', 'haspainintensity', 'hasprerequisite', 'hasproperty', 'hassubevent', 'inheritsfrom', 'instanceof', 'isa', 'locatednear', 'locationofaction', 'madeof', 'motivatedbygoal', 'notcapableof', 'notdesires', 'nothasa', 'nothasproperty', 'notisa', 'notmadeof', 'partof', 'receivesaction', 'relatedto', 'symbolof', 'usedfor']
    # und_rellist = ['_'+x for x in rel_list]
    # relation_set = set(rel_list).union(und_rellist)
    for i, path in enumerate(paths):
        path = path.replace('_ ', '_')
        pathwords = set(path.split())
        pathwords_norel = pathwords - relation2textset
        pathwords_compound = get_path_words(path, relation2textset)
#         pathwords_norel = get_path_words(path)
        
        ##if too less entities, rmove - TODO its dangerous might have to remove or write a better version
        if type is not 'aug' and input_entities is not None and (len(pathwords_norel)<len(input_entities)-3):# or len(pathwords_norel)>len(input_entities)+1):
            # print(pathwords_norel, path, input_entities)
            continue
        if len(pathwords_norel)<2:
            # print(path, input_entities)
            #make the final part of path same as the enitity itself if the path only has one entity
            # print(path, pathwords_norel)
            path = ', '.join(str(e) for e in pathwords_norel)
            # continue
            
        #only add paths if ppl<2*min ppl
        if scores[i]<2.0*min_score:
            # print(path, '1--1', filt_paths, path in filt_paths)
            # print(input_entities)
            if input_entities is not None:
                path_specificwords = set(path.split())
                path_specificwords = path_specificwords-relation2textset
                for multiword in input_entities:
                    for word in multiword.split():
                        if word not in input_entities:
                            input_entities.append(word)
                extraents = path_specificwords-set(input_entities)
                #some entities in path are extra
                if len(extraents)>1:
                    # print(extraents, path, set(input_entities))
                    continue
            if parse_edges:
                path = convert_edgesnames(path, relation2text)
            if path in filt_paths: # avooid repetition
                continue
            # print(path, '--', filt_paths, path in filt_paths)
            if pathwords_compound in list_pathsadded:
                continue
            filt_paths.append(path)
            filt_scores.append(scores[i])
            list_pathsadded.append(pathwords_compound)

    return filt_paths, filt_scores

def get_min_path(paths, scores, relation2text, parse_edges=True):
    min_score, max_scores = min(scores), max(scores)
    filt_paths, filt_scores = [], []
    for i, path in enumerate(paths):
        path = path.replace('_ ', '_')
        if scores[i]==min_score:
            filt_path=path
            filt_score=(scores[i])
    
    if parse_edges:
        filt_path = convert_edgesnames(filt_path, relation2text)
    # print(paths, filt_path)
    return filt_path, filt_score

''' GPT-2 Response Generation Related '''

def clean_generation(results):
    if type(results)==list:
        results = results[0]
    results = results.split('[response] : ')[-1]
    eor_ind = results.find('<eor')
    final = results[:eor_ind - 1].strip()
    
    return final

def predict_formality(model, tokenizer, conditioning_model, input_text, precondition_topk=200, do_sample=True, length_cutoff=512, condition_lambda=1.0, device='cuda', verbose=True):
    with torch.no_grad():
        batch_size = len(input_text)
        # assumes initially all same length.
        encoded_input = [tokenizer.encode(it, return_tensors='pt').to(device) for it in input_text] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)

        # input_ids = torch.LongTensor([[65000]]).to(device)
        input_ids = encoded_input.to(device)
        cur_len = 1
        max_length = length_cutoff
        min_length = 0
        temperature = 0.7#1.0
        top_k = 50
        top_p = 1.0
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        bad_words_ids = [[65000]]
        pad_token_id = 65000
        eos_token_id = 0
        effective_batch_size = batch_size
        attention_mask = encoded_input.new_ones(encoded_input.shape)
        use_cache = True
        # model_specific_kwargs = {'encoder_outputs': model.get_encoder()(encoded_input, attention_mask=attention_mask)}
        model_specific_kwargs = {'encoder_outputs': model(encoded_input, attention_mask=attention_mask)}

        output = _generate_no_beam_search(model, tokenizer,
                                        conditioning_model,
                                        condition_lambda,
                                        precondition_topk,
                                        encoded_input,
                                        input_ids,
                                        cur_len,
                                        max_length,
                                        min_length,
                                        do_sample,
                                        temperature,
                                        top_k,
                                        top_p,
                                        repetition_penalty,
                                        no_repeat_ngram_size,
                                        bad_words_ids,
                                        pad_token_id,
                                        eos_token_id,
                                        batch_size,
                                        attention_mask,
                                        use_cache,
                                        model_specific_kwargs,
                                          verbose=verbose,)

        return [tokenizer.decode(s[:], skip_special_tokens=True) for s in output] # 1: to delete the pad token

# hack of code from transformers/generation_utils.py
# to get our conditioning
def _generate_no_beam_search(
        model, tokenizer,
        conditioning_model,
        condition_lambda,
        precondition_topk,
        encoded_input,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
        verbose=True,
):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            # model_inputs = model.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs)
            model_inputs = model.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache)
            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # scores = model.postprocess_next_token_scores(
            scores = postprocess_next_token_scores(
                scores=next_token_logits,input_ids=input_ids,no_repeat_ngram_size=no_repeat_ngram_size,bad_words_ids=bad_words_ids,
                cur_len=cur_len,min_length=min_length,max_length=max_length,eos_token_id=eos_token_id,repetition_penalty=repetition_penalty,batch_size=batch_size,num_beams=1,)

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems
            tt = tokenizer
            top_logits, top_indices = scores.topk(precondition_topk, dim=1) # batch x topk
            input_ids_suffix = input_ids
            if condition_lambda>0:
                target_idx = (input_ids[0] == 16793).nonzero(as_tuple=True)[-1].tolist()
                if len(target_idx)>0 and input_ids[0][target_idx[0]+1].item()==60:
                    input_ids_suffix = input_ids[:,target_idx[0]+2:]
                # print(input_ids_suffix)
                # import pdb;pdb.set_trace()
            tplus1_candidates = torch.cat([input_ids_suffix.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2)[:, :, 1:] # batch x topk x seq+1, with pad dropped
            expanded_lengths = torch.LongTensor([[cur_len for _ in range(precondition_topk)] for _ in range(batch_size)]).to(scores.device)
            input_lengths = torch.LongTensor([tplus1_candidates.shape[2] for _ in range(precondition_topk)]).to(scores.device)

            if condition_lambda == 0:
                condition_logits = torch.zeros_like(top_logits).float()
            else:                
                condition_logits = conditioning_model(tplus1_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    # expanded_lengths.flatten(0, 1), # batch*topk
                                                    input_lengths,
                                                    None,
                                                    None,
                                                    None)
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1)[:, :, -1] # batch x topk of last formality pred
                # condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs
                # print(condition_logits)
            full_logits = top_logits + condition_lambda * condition_logits
            scores = F.softmax(full_logits, dim=-1)

            top_probs = F.softmax(top_logits, dim=-1)
            condition_logits_prob = F.softmax(condition_logits, dim=-1)
            # full_probs = top_probs + condition_lambda * condition_logits_prob
            # scores = F.normalize(full_probs,dim=-1, p=1)

            if do_sample:
                scores = scores / temperature
                # scores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                topchosen_token_indice = torch.multinomial(scores, num_samples=1).squeeze(1)
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), topchosen_token_indice]
            else:
                # Greedy decoding
                topchosen_token_indice = torch.argmax(scores, dim=-1)
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(scores, dim=-1)]

            if verbose and condition_lambda>0:
                print([(i, tokenizer.decode([z]), round(x,4),round(y,4), round(s,4)) for i, (x,y,s,z) in enumerate(zip(top_probs[0].tolist(), condition_logits_prob[0].tolist(), scores[0].tolist(), top_indices[0].tolist()))])
                # print([tokenizer.decode([x]) for x in top_indices[0]])
                print(tokenizer.decode(input_ids[0]))
                print(next_token[0], tokenizer.decode([next_token]), topchosen_token_indice[0].item())

            tokens_to_add = next_token
            
            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if next_token[0]==tokenizer.eos_token_id:
                break
            # extend attention_mask for new generated input if only decoder
            if model.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids

def postprocess_next_token_scores(
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        return scores

def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens

def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """
    Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
    """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty