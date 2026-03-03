import sys
sys.path.append('..')
import os
import json
import pprint
import string
# import random
import nltk
import numpy as np
from typing import List, Literal, Union
from infinigram import api_client
# from infinigram import ngram_processor
from creativity_index.DJ_search_exact import find_exact_match, Document, HF_TOKEN, API_URL
from transformers import AutoTokenizer
from unidecode import unidecode
from tqdm import tqdm
# from statistics import geometric_mean
# from sacremoses import MosesDetokenizer
# md = MosesDetokenizer(lang='en')
# Create a client
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=HF_TOKEN,
                                            add_bos_token=False, add_eos_token=False)
tokenize_func = lambda x: tokenizer.tokenize(x)
detokenize = lambda x: tokenizer.decode(tokenizer.convert_tokens_to_ids(x))

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

def compute_ppl(text: str, client: api_client.InfinigramClient,
                 min_ngram_len: int = 0):
    # tokens = client.infgram_prob(text)['tokens']
    tokens = tokenize_func(unidecode(text))
    probs = []
    for i, t in enumerate(tokens):
        # print(tokens[:i+1])
        # skip empty token
        if i == 0 and t == '▁':
            continue
        detoks = detokenize(tokens[:i+1])
        # print(detoks)
        res = client.infgram_prob(detoks)
        # print(res)
        prob, longest = res['prob'], res['longest_suffix']
        if len(longest.split()) >= min_ngram_len:
            probs.append(prob)
        else:
            probs.append(0)

    # Compute perplexity
    # handle log(0) by adding a small value to probs
    # probs = [p if p > 0 else 1e-10 for p in probs]
    probs = np.array([p if p > 0 else 1.2e-10 for p in probs])
    geom_mean = probs.prod()**(1.0/len(probs))
    # ppl = 2**(-1 * np.mean(np.log2(probs)))
    # return ppl, probs, tokens
    ppl = 1/(geom_mean+1.2e-10)
    return ppl, probs, tokens

def compute_crindex(expr: str, client: api_client.InfinigramClient,
                    dj_cache_file: str = None,
                    min_ngrams=5, 
                    corpus = "v4_dolma-v1_7_llama",
                    debug=False):
    
    if len(expr) <= 1000:
        count, _ = client.get_occurrence_counts(expr)
        if count:
            return 1, 1, {expr: count}
        
    tokenized_text = tokenize_func(unidecode(expr))
    tgt_doc = Document(f'None', tokenized_text)
    output = find_exact_match(detokenize, tgt_doc, min_ngrams, 
                            cache_file=dj_cache_file, 
                            corpus=corpus,
                            verbose=debug)
    return output['matched_coverage'], 0, output['matched_spans']

def compute_agg_crindex(expr: str, client: api_client.InfinigramClient,
                    dj_cache_file: str = None,
                    min_ngrams: int = 1,
                    max_ngrams: int = 7, 
                    how_agg = "weighted",
                    alpha: float = 0.5,
                    debug=False):
    
    if len(expr) <= 1000:
        count, _ = client.get_occurrence_counts(expr)
        if count:
            return 1, 1, {expr: count}
        
    tokenized_text = tokenize_func(unidecode(expr))
    tgt_doc = Document(f'None', tokenized_text)
    u_scores = []
    for ngram_size in range(min_ngrams, max_ngrams+1):
        output = find_exact_match(detokenize, tgt_doc, ngram_size, 
                                cache_file=dj_cache_file, verbose=debug)
        coverage = output['matched_coverage']   
        u_scores.append(1 - coverage)

    if how_agg == "weighted":
        weighted_scores = [u * (alpha ** (2*(i+1))) for i, u in enumerate(u_scores)]
        u_score = sum(weighted_scores) / len(u_scores)
        print("weights: ", [round(alpha ** (i+1),2) for i in range(len(u_scores))])
        print("weighted_scores: ", [round(u, 4) for u in weighted_scores])
    elif how_agg == "average":
        u_score = sum(u_scores) / len(u_scores)
    # print("u_scores: ", u_scores)
    # print("u_score: ", u_score)
    return u_score, u_scores, 0, output['matched_spans']

def format_trailing(expr):
    while expr and (expr[0] in string.punctuation or expr[0].isspace()):
        expr = expr[1:]
    while expr and (expr[-1] in string.punctuation or expr[-1].isspace()):
        expr = expr[:-1]
    return expr 

def get_uniqueness_score(expr: str, client: api_client.InfinigramClient, 
                         n_choice: Literal["smallest, largest"] = "largest",
                         how: Literal["max", "min", 
                                        "mean", "median", 
                                        "sum", "logmean", "percent",
                                        "zero_ngram_words"] = "percent",
                         min_ngrams: int = 3, 
                         max_ngrams: int = 5,
                         smooth=0.01, 
                         debug=False, print_num_api_calls=False):
    """
    Get the uniqueness score of a given expression.

    Algorithm for percent_largest:
    1. Generate ngrams from min_ngram to num_words
    2. For each ngram, check if its subset has been assigned an occurrence count of 0
    (since a string containing a substring with 0 occurrence count will also have 0 occurrence count)
    3. If it has not, get the occurrence count from the API
    4. If the occurrence count is 0, add the ngram to zero_ngrams
    5. If the occurrence count is not 0, add the ngram to ngram_counts
    6. Now continue the loop with larger ngrams
    7. If no ngram has been found, get the largest substring of the last nonzero ngram found
    8. Check all ngrams of that size and return the largest occurrence count

    Algorithm for percent_smallest:
    1. Generate ngrams from min_ngram to num_words
    2. For each ngram, check if its subset has been assigned an occurrence count of 0
    (since a string containing a substring with 0 occurrence count will also have 0 occurrence count)
    3. If it has not, get the occurrence count from the API
    4. If the occurrence count is 0, break the loop
    5. If the occurrence count is not 0, continue looping to larger ngrams
    6. If an ngram has been found, get the smallest n for which there is an occurence
    7. Get the occurrence count for all ngrams of that size
    """

    # expr = format_trailing(unidecode(expr))
    # print("preprocessed expr: -", expr, "-")
    num_words = len(expr.split())
    num_api_calls = 0
    if len(expr) <= 1000: # input limit to infinigram
        count, _ = client.get_occurrence_counts(expr)
        num_api_calls += 1
        if count:
            if how == "percent" or how == "zero_ngram_words":
                return 0, num_api_calls, {expr: count}, []
            elif how == "logmean":
                return np.log(count+smooth), num_api_calls, {expr: count}, []
            return count, num_api_calls, {expr: count}, []
    
    ngram_counts = []
    zero_ngrams = set()
    # should not check the full expression since we know its count is 0
    max_ngrams = min(max_ngrams, num_words-1)
    # for n in range(min_ngrams, num_words+1):
    for n in range(min_ngrams, max_ngrams+1):

        if debug: print(n)
        occured = False
        ngrams = nltk.ngrams(expr.split(), n)
        ngrams = [format_trailing(" ".join(ng)) for ng in ngrams]

        for ngram_idx, ngram in enumerate(ngrams):
            if not any(z in ngram for z in zero_ngrams):
                # if debug: print(ngram)
                count, _ = client.get_occurrence_counts(ngram)
                if debug: print(ngram, '\t', count)
                num_api_calls += 1
                if count:
                    # if debug: pprint.pprint(ngram_counts)
                    occured = True
                    # now we can check larger n_grams
                    if n_choice == "largest":
                        ngram_counts.append((n, ngram_idx, ngram, count, ngrams))
                        break
                else:
                    if debug: print(ngram, '\t', count)
                    zero_ngrams.add(ngram)
                    if n_choice == "smallest":
                        # found a small unique ngram, break
                        ngram_counts.append((n, ngram_idx, ngram, count, ngrams))
                        occured = False
                        break
            # dont need to do this bc as soon as we add to zero ngrams we break for smallest:
            # and for largest we dont collect unqiue ngrams but rather the occuring ones
            # else:
            #     ngram_counts.append((n, ngram_idx, ngram, 0, ngrams))

        if not occured and ngram_counts:
            if debug: pprint.pprint(ngram_counts)
            n, ngram_idx, ngram, count, ngrams = ngram_counts[-1]
            if debug: print("Selected n: ", n, " ngram: ", ngram)
            substring_counts = {ngram: count}
            for ngram in ngrams:
                if not any(z in ngram for z in zero_ngrams):
                    count, _ = client.get_occurrence_counts(ngram)
                    num_api_calls += 1
                    substring_counts[ngram] = count
                else:
                    substring_counts[ngram] = 0
            # get largest value from substring_counts
            if how == "max":
                unqiueness_score = max(substring_counts.values())
            elif how == "sum":
                unqiueness_score = sum(substring_counts.values())
            elif how == "mean":
                unqiueness_score = np.mean(list(substring_counts.values()))
            elif how == "min":
                unqiueness_score = min(substring_counts.values())
            elif how == "median":
                unqiueness_score = np.median(list(substring_counts.values()))
            elif how == "logmean":
                # Add 1 to handle zeros, then take log, then take mean
                values = list(substring_counts.values())
                log_values = [np.log(v + smooth) for v in values]  # log(x + 0.01)
                # unqiueness_score = np.exp(np.mean(log_values)) - smooth  # subtract the smoothing factor
                unqiueness_score = np.mean(log_values)  
            elif how == "percent":
                # percent of unique ngrams
                total_ngrams = len(substring_counts)
                unique_ngrams = len([v for v in substring_counts.values() if v == 0])
                unqiueness_score = unique_ngrams / total_ngrams
            elif how == "zero_ngram_words":
                # percent of words that are in zero ngrams
                flags = [0] * len(expr)
                words_in_zero_ngrams = 0
                for ng, count in substring_counts.items():
                    if count == 0:
                        if debug: print(expr)
                        if debug: print(ng)
                        try:
                            start = expr.index(ng)
                        except Exception as e:
                            print(f"Error finding ngram '{ng}' in expr '{expr}': {e}")
                            exit(1)
                        end = start + len(ng)
                        if debug: print(start, end)
                        flags[start:end] = [1] * len(ng)
                        # nonzero_ngram_words = expr[start:end].split(" ")
                        # words_in_nonzero_ngrams += len(nonzero_ngram_words)
                if debug: print(flags)
                only_flagged_text = "".join([w for i, w in enumerate(expr) if flags[i] == 1 or w == " "]).strip()
                if debug: print(only_flagged_text)
                # words_in_zero_ngrams = len(only_flagged_text.split(" "))
                words_in_zero_ngrams = len(nltk.tokenize.casual.casual_tokenize(only_flagged_text))
                if debug: print(nltk.tokenize.casual.casual_tokenize(only_flagged_text))
                # print(only_flagged_text.split(" "))
                # words_in_nonzero_ngrams = len(nonzero_ngram_words)
                # total_words = len(expr.split())
                total_words = len(nltk.tokenize.casual.casual_tokenize(expr))
                if debug: print(nltk.tokenize.casual.casual_tokenize(expr))
                # if debug: print(only_flagged_text.split(" "))
                # words_in_nonzero_ngrams = len(nonzero_ngram_words)
                # total_words = len(expr.split())
                unqiueness_score = words_in_zero_ngrams / total_words
            else:
                raise ValueError(f"Invalid how: {how}. Must be one of: 'max', 'min', 'mean', 'median', 'sum', 'logmean', 'percent', 'zero_ngram_words'")
            if debug or print_num_api_calls: print("API/numwords: ", num_api_calls/len(expr.split()))
            return unqiueness_score, num_api_calls, substring_counts, ngram_counts
    else:
        if debug: print("No ngrams found")
        if how in ["max", "min", "mean", "median", "sum", "logmean"]:
            unqiueness_score = 0
        elif how in ["percent", "zero_ngram_words"]:
            if n_choice == "smallest":
                # if we don't find n s.t. at least 1 ngram is unique until we reach the full expression, 
                # mark the expression as not unique
                unqiueness_score = 0
            else:
                # if we don't find n s.t. at least one is not unique
                unqiueness_score = 1
        elif how == "logmean":
            unqiueness_score = np.log(smooth)
    if debug or print_num_api_calls: print("API/numwords: ", num_api_calls/len(expr.split()))    
    return unqiueness_score, num_api_calls, {}, ngram_counts

def split_by_punctuation(text):
    """
    Split a string by punctuation marks and newlines while preserving word groups.
    
    Args:
        text (str): Input string to split
        
    Returns:
        list: List of strings split by punctuation and newlines
    """
    import re
    
    # Define punctuation pattern excluding apostrophes, also include newlines
    # This will match common punctuation marks and newline characters
    pattern = r'[,.!?;:—()\n\r]+'
    
    # Split by punctuation and newlines, strip whitespace
    parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
    
    return parts

if __name__ == "__main__":

    from config import Config
    # print(Config.CACHE_FILE)
    # for batch_id in ["21_25", "26_30", "31_40", "41_50"]:
    for batch_id in ["21_25_olmo2", "26_30_olmo1"]:
        for hai in ["ai"]:
            infinigram_cache = f"/home/a.saakyan/projects/mcreat/infinigram/caches/{batch_id}_{hai}_cache.json"
            infinigram_cache_dclm = f"/home/a.saakyan/projects/mcreat/infinigram/caches/{batch_id}_{hai}_dclm_cache.json"

            if batch_id == "21_25" or "olmo1" in batch_id:
                client_dolma_corpus = 'v4_dolma-v1_7_llama'
            else:
                # we used olmo-32B for later batches
                client_dolma_corpus =  'v4_olmo-2-0325-32b-instruct_llama'

            client_dolma = api_client.InfinigramClient('https://api.infini-gram.io/', 
                                                client_dolma_corpus,
                                                infinigram_cache)
            client_dclm = api_client.InfinigramClient("https://api.infini-gram-mini.io/",
                                                    "v2_dclm_all",
                                                    infinigram_cache_dclm)
            
            dj_cache = f"/home/a.saakyan/projects/mcreat/creativity_index/caches/crindex_cache_{batch_id}_{hai}.json"
            # processor = ngram_processor.NgramProcessor()

            data_dir = "/home/a.saakyan/projects/mcreat/data"
            
            if hai == "human":
                source_path = f"{data_dir}/noveltyAnnot/hand_selected/{batch_id}/paragraphs"
            else:
                source_path = f"{data_dir}/noveltyAnnot/hand_selected/{batch_id}/ai_paragraphs"
            save_path = f"{data_dir}/noveltyAnnot/hand_selected/{batch_id}/highlighted_{hai}_{batch_id}.json"
            
            if ".json" in source_path:
                with open(source_path, "r") as f:
                    selected_paragraphs = json.load(f)
            else:
                selected_paragraphs = []
                for fname in os.listdir(source_path):
                    with open(os.path.join(source_path, fname), 'r', encoding='utf-8') as f:
                        para = f.read().strip()
                        selected_paragraphs.append({"para": para,
                                                    "id": fname.split("_story.txt")[0]})

            hihglihgted_paras = []

            for para_data in tqdm(selected_paragraphs):
                hihglihgted_para = {}
                para_txt = para_data['para']
                para_parts = split_by_punctuation(para_txt)
                hihglihgted_para = {"id": para_data['id'], 
                                    "para": para_txt, 
                                    "para_parts": para_parts}
                u_scores = {}
                for p in tqdm(para_parts):
                # for p in tqdm(para_parts[:5]):
                # for p in para_parts:
                    expr = format_trailing(unidecode(p))
                    if len(expr.split()) >= 2:
                        print(p)
                        print("preprocessed expr: -", expr, "-")
                        nwords = len(expr.split())
                        min_ng = min(nwords//3, 5)
                        print("min_ng: ", min_ng)
                        # if first letter is capital
                        exprs = [expr]
                        if expr[0].isupper() and expr[1].islower() and expr[1] != "I":
                            # get version of expr that first letter is lower
                            expr_lower = expr[0].lower() + expr[1:]
                            exprs.append(expr_lower)
                        print("EXPRS: ", exprs)
                        u_scores[p] = {}
                        # for how in ["logmean", "median", "percent", "zero_ngram_words",
                        #             "5gram_crindex", "3gram_crindex", "ppl"]:
                        # for how in ["logmean", "median", "percent", "zero_ngram_words",
                        #             "5gram_crindex", "3gram_crindex", "ppl"]:
                        for how in ["logmean", "median", "percent", "zero_ngram_words",
                                    "logmean_dclm", "median_dclm", "percent_dclm", "zero_ngram_words_dclm",
                                    "5gram_crindex", "3gram_crindex", "ppl"]:
                        # for how in ["logmean", "median", "percent", "zero_ngram_words",
                        #             "crindex", "ppl"]:
                            
                            # if how in ["5gram_crindex", "3gram_crindex", "crindex", "ppl"]:
                            if how in ["5gram_crindex", "3gram_crindex", "ppl"]:
                                u_score_list = []
                                for expr in exprs:
                                    if "5gram" in how:
                                        u_score, _, _ = compute_crindex(expr, client_dolma, 
                                                                        dj_cache,
                                                                        min_ngrams=5,
                                                                        corpus=client_dolma_corpus)
                                    elif "3gram" in how:
                                        u_score, _, _ = compute_crindex(expr, client_dolma, 
                                                                        dj_cache,
                                                                        min_ngrams=3,
                                                                        corpus=client_dolma_corpus)
                                    # elif "crindex" in how:
                                    #     u_score, _, _ = compute_crindex(expr, dj_cache,
                                    #                                             min_ngrams=min_ng)
                                    elif how == "ppl":
                                        u_score, _, _ = compute_ppl(expr, client_dolma)
                                    print(f"{how}: ", u_score)
                                    u_score_list.append(u_score)
                                # if higher is worse
                                if "crindex" in how:
                                    u_scores[p][how] = {"score_list": u_score_list, 
                                                        "worst": max(u_score_list)}
                                else:
                                    # raw score higher -> better (lower is worse)
                                    u_scores[p][how] = {"score_list": u_score_list,
                                                        "worst": min(u_score_list)}
                            else:
                                for n_choice in ["largest", "smallest"]:
                                    if "_dclm" in how:
                                        u_score_client = client_dclm
                                    else:
                                        u_score_client = client_dolma
                                    u_score_list = []
                                    for expr in exprs:
                                        if n_choice == "smallest":
                                            u_score, _, _, _ = get_uniqueness_score(expr, u_score_client, n_choice=n_choice,
                                                                                how=how.split("_dclm")[0], 
                                                                                min_ngrams=1, 
                                                                                max_ngrams=nwords, 
                                                                                debug=False)
                                        else:
                                            u_score, _, _, _ = get_uniqueness_score(expr, u_score_client, n_choice=n_choice,
                                                                                how=how.split("_dclm")[0], 
                                                                                min_ngrams=1, 
                                                                                max_ngrams = nwords,
                                                                                debug=False)
                                        print(f"{how}_{n_choice}: ", u_score)
                                        u_score_list.append(u_score)
                                    if how in ["logmean", "median"]:
                                        u_scores[p][f"{how}_{n_choice}"] = {"score_list": u_score_list, 
                                                            "worst": max(u_score_list)}
                                    else:
                                        # raw score higher -> better (lower is worse)
                                        u_scores[p][f"{how}_{n_choice}"] = {"score_list": u_score_list,
                                                            "worst": min(u_score_list)}
                    
                hihglihgted_para['u_scores'] = u_scores
                hihglihgted_paras.append(hihglihgted_para)
                print("Saving...")
                with open(save_path, "w") as f:
                    json.dump(hihglihgted_paras, f, indent=4)
                # break

