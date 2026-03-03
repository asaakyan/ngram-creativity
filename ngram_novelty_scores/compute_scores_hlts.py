from compute_scores import get_uniqueness_score, compute_crindex, compute_ppl, format_trailing
import json
import os
from infinigram import api_client
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

if __name__ == "__main__":

    infinigram_cache = f"/home/a.saakyan/projects/mcreat/infinigram/caches/front_hlts_cache.json"
    infinigram_cache_dclm = f"/home/a.saakyan/projects/mcreat/infinigram/caches/front_hlts_dclm_cache.json"
    dj_cache = f"/home/a.saakyan/projects/mcreat/creativity_index/caches/crindex_cache_front_hlts.json"

    hlts_df = pd.read_csv("../data/llm_performance/all_frontier_hlts.csv")
    OUTPUT_FILE = "../data/llm_performance/all_frontier_landen_hlts_w_scores.jsonl"

    data = []
    for i, row in tqdm(hlts_df.iterrows()):
        data_row = {}
        for c in hlts_df.columns:
            data_row[c] = row[c]
        if "front" in dj_cache:
            # always choose the larger dolma corpus for claude, gpt5
            model = "olmo2" 
        else:
            model = row['model']
        if model == "olmo1":
            client_dolma_corpus = 'v4_dolma-v1_7_llama'
        elif model == "olmo2":
            # we used olmo-32B for later batches
            client_dolma_corpus =  'v4_olmo-2-0325-32b-instruct_llama'
        else:
            raise ValueError(f"Unknown model: {model}")

        client_dolma = api_client.InfinigramClient('https://api.infini-gram.io/', 
                                            client_dolma_corpus,
                                            infinigram_cache)
        client_dclm = api_client.InfinigramClient("https://api.infini-gram-mini.io/",
                                                "v2_dclm_all",
                                                infinigram_cache_dclm)
        
        p = row['novel_expr']
        #standardize whitespace
        p = " ".join(p.split()).strip()
        expr = format_trailing(unidecode(p))
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
        u_scores = {}
        # for how in ["logmean", "median", "percent", "zero_ngram_words",
        #             "logmean_dclm", "median_dclm", "percent_dclm", "zero_ngram_words_dclm",
        #             "5gram_crindex", "3gram_crindex", "ppl"]:
        for how in ["logmean", "median", "percent", "zero_ngram_words",
                    "logmean_dclm", "median_dclm", "percent_dclm", "zero_ngram_words_dclm",
                    "ppl"]:

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
                    elif how == "ppl":
                        u_score, _, _ = compute_ppl(expr, client_dolma)
                    print(f"{how}: ", u_score)
                    u_score_list.append(u_score)
                # if higher is worse
                if "crindex" in how:
                    u_scores[how] = {"score_list": u_score_list, 
                                        "worst": max(u_score_list)}
                else:
                    # raw score higher -> better (lower is worse)
                    u_scores[how] = {"score_list": u_score_list,
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
                                                                debug=True)
                        else:
                            u_score, _, _, _ = get_uniqueness_score(expr, u_score_client, n_choice=n_choice,
                                                                how=how.split("_dclm")[0], 
                                                                min_ngrams=1, 
                                                                max_ngrams = nwords,
                                                                debug=True)
                        print(f"{how}_{n_choice}: ", u_score)
                        u_score_list.append(u_score)
                    if how in ["logmean", "median"]:
                        u_scores[f"{how}_{n_choice}"] = {"score_list": u_score_list, 
                                            "worst": max(u_score_list)}
                    else:
                        # raw score higher -> better (lower is worse)
                        u_scores[f"{how}_{n_choice}"] = {"score_list": u_score_list,
                                            "worst": min(u_score_list)}
        data_row['u_scores'] = u_scores
        data.append(data_row)
        print("U_scores: ", u_scores)
        print("-"*50)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(data, f, indent=4)