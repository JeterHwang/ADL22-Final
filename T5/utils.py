import torch
import math
import spacy
import nltk
import numpy as np
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
                if noune_indexdict[noune] > verbe_index and noune_indexdict[noune] - verbe_index < 3:
                    # print(verbe, noune)
                    verb_phrases.append(verbe + ' ' + noune)

    features = [x for x in features if x not in ' '.join(verb_phrases)]
    features_verb = [x for x in features_verb if x not in ' '.join(verb_phrases)]
    features_verb = [x.replace('have', '').replace("'ve", '').strip() for x in features_verb]
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

def connect_entities(head_entity, tail_entity, generator, tokenizer, temperature=1, num_outs=1, top_k=0, top_p=1.0):
    gen_input = prepare_input(head_entity, tail_entity, tokenizer)
    gen_input = gen_input.to('cuda')
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

def get_min_path(paths, scores, parse_edges=True):
    min_score, max_scores = min(scores), max(scores)
    filt_paths, filt_scores = [], []
    for i, path in enumerate(paths):
        path = path.replace('_ ', '_')
        if scores[i]==min_score:
            filt_path=path
            filt_score=(scores[i])
    
    if parse_edges:
        filt_path = convert_edgesnames(filt_path)
    # print(paths, filt_path)
    return filt_path, filt_score

