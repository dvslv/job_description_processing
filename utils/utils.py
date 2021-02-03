import string
import pymorphy2
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams 
from tqdm.auto import tqdm
import numpy as np
from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import re
from Levenshtein import distance
from deeppavlov.models.preprocessors.str_lower import str_lower
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder

embedder = FasttextEmbedder(load_path="./ft_native_300_ru_wiki_lenta_lemmatize.bin")
tokenizer_simple = NLTKTokenizer()
morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer('\w+|[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')

def prepare_dataset(text_list):
    pad_token = '<unk>'
    documents = []
    documents_tags = []
    for text in tqdm(text_list):
        split_text = [token for token in tokenizer.tokenize(text)]
        lem_text = [morph.parse(token)[0].normal_form 
                    if morph.parse(token)[0].normal_form not in stopwords.words('russian') \
                    and token not in string.punctuation \
                    else pad_token \
                    for token in split_text]
        documents.append(lem_text)
        tag_text = [['_'.join([morph.parse(token)[0].normal_form,
                               str(morph.parse(token)[0].tag.POS)]) 
                      for token in split_text]
                    ]
        documents_tags.append(tag_text)

    documents_tags = [doc[0] for doc in documents_tags]
        
    return documents, documents_tags

def generate_ngrams(documents, max_len=None):
    ngramms = []
    for seq in tqdm(documents):
        seq_ngramms = []
        for i in range(1, max_len):
            seq_ngramms += [' '.join(trip) 
                            for trip in list(ngrams(seq, i)) 
                            if '<unk>' not in trip]

        ngramms += [seq_ngramms]
    return ngramms

def init_bert(bert_path):
    bert_config = read_json(configs.embedder.bert_embedder)
    bert_config['metadata']['variables']['BERT_PATH'] = bert_path

    bert_model = build_model(bert_config)

    return bert_model


def location_recognition(ner_model,text):
    location_tags = []
    for n,i in enumerate(ner_model([text])[1][0]):
        result = re.search(r'LOC', i)
        if result != None:
            #print(result.group(0),ner_model([text])[0][0][n])
            location_tags.append(ner_model([text])[0][0][n])    
    return location_tags


def get_nearest_terms(terms, terms_db, threshold=0.2):
    if len(terms_db) == 0:
        return term
    terms_dist = [(t,distance(term.lower(),t.lower())) for t in terms_db for term in terms]
    sort_terms = sorted(terms_dist, key=lambda x:x[1])
    top_token = sort_terms[0] 
    if type(threshold) is int and top_token[1] < threshold:
        return (top_token[0], top_token[1])
    elif type(threshold) is float and top_token[1]/len(top_token[0]) < threshold:
        return (top_token[0], top_token[1]/len(top_token[0]))
    else:
        return (terms, 1.)

def l2_norm(x):
    return np.sqrt(np.sum(x**2))

def div_norm(x):
    norm_value = l2_norm(x)
    if norm_value > 0:
        return x * ( 1.0 / norm_value)
    else:
        return x
    

def get_sent_fasttext_emb(text_string):
    tags = tokenizer_simple(str_lower([text_string]))
    tags_embs = embedder(tags)
    tags_embs_norm = [div_norm(e) for e in tags_embs[0]]
    arr = np.array(tags_embs_norm)
    sent_emb = arr.sum(axis=0)/len(tags[0])
    return sent_emb

def fuzzy_search(text_string, terms_db, treshold = 0.15 ):
    documents, _ = prepare_dataset([text_string])
    documents_ngamms = generate_ngrams(documents, 3)
    prof_list = []
    for i in documents_ngamms[0]:
        token, score = get_nearest_terms([i], terms_db , treshold)
        if score <= treshold:
            #prof_list.append(token)
            prof_list.append('#'+'_'.join(token.lower().split(' ')))
    return prof_list

def embs_sim_search_best_ngrams(text_string, terms_db, treshold = 0.82):
    documents, _ = prepare_dataset([text_string])
    documents_ngamms = generate_ngrams(documents, 3)
    df_p_embs = [[i, get_sent_fasttext_emb(i)] for i in terms_db]
    temp = {}
    for i in documents_ngamms[0]:
        for p_embs in df_p_embs:
            sim = cosine_similarity([get_sent_fasttext_emb(i)],[p_embs[1]])[0][0]
            if sim > treshold:
                if len(i.split(' ')) in temp:
                    if temp[len(i.split(' '))][1] < sim:
                        temp[len(i.split(' '))] = [i,sim]
                else:
                    temp[len(i.split(' '))] = [i,sim]            
    prof_list = []
    for i in temp.values():
        prof_list.append('#'+'_'.join(i[0].split(' ')))
       #prof_list.append(i[0])    
    return prof_list


def embs_sim_search(text_string, terms_db, treshold = 0.82):
    documents, _ = prepare_dataset([text_string])
    documents_ngamms = generate_ngrams(documents, 3)
    df_p_embs = [[i, get_sent_fasttext_emb(i)] for i in terms_db]
    temp = {}
    for i in documents_ngamms[0]:
        for p_embs in df_p_embs:
            sim = cosine_similarity([get_sent_fasttext_emb(i)],[p_embs[1]])[0][0]
            if sim > treshold:
                if i in temp:
                    if temp[i] < sim:
                        temp[i] = sim
                else:
                    temp[i] = sim 
    prof_list = []
    for i in temp.keys():
        prof_list.append('#'+'_'.join(i.split(' ')))   
    return prof_list
    