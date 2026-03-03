import nltk
import os
import json
import toolz
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from dataclasses import dataclass
from functools import partial
import multiprocessing as mp

from nltk import sent_tokenize
from unidecode import unidecode
from sacremoses import MosesDetokenizer


md = MosesDetokenizer(lang='en')
os.environ['TOKENIZERS_PARALLELISM'] = "false"


@dataclass
class RetrievedDoc:
    doc_text: str
    hit_ratio: float


def format_retrieved_doc(retrieved_doc: RetrievedDoc):
    return {
        'doc_text': retrieved_doc.doc_text,
        'hit_ratio': retrieved_doc.hit_ratio,
    }


def tokenize(x):
    return nltk.tokenize.casual.casual_tokenize(unidecode(x))


def detokenize(x):
    return md.detokenize(x)


def document_attribution(target_tokens: List[str], num_hits: int, span_length: int, ref_text: str):
    """
    :param target_tokens: a list of tokens (str) from the target document
    :param ref_text: text (str) from the reference document
    :param num_hits: minimal number of matched tokens for the matched span
    :param span_length: length of the fuzzy span
    :return: a RetrievedDoc object
    """
    ref_sentences = sent_tokenize(ref_text)
    ref_sent_tokens = [tokenize(x) for x in ref_sentences]
    ref_sent_idxes, last_idx = [], 0
    for sent in ref_sent_tokens:
        ref_sent_idxes.append((last_idx, last_idx + len(sent)))
        last_idx = last_idx + len(sent)
    ref_tokens = [y for x in ref_sent_tokens for y in x]
    ref_token_flags = [False for _ in ref_tokens]

    matrix = np.zeros((len(ref_tokens), len(target_tokens)))
    for i, ref_token in enumerate(ref_tokens):
        for j, target_token in enumerate(target_tokens):
            if ref_token == target_token:
                matrix[i, j] = 1

    # find sentences with matched span for document retrieval
    for i in range(len(ref_tokens) - span_length + 1):
        if not np.sum(matrix[i:i + span_length]):
            continue

        for j in range(len(target_tokens) - span_length + 1):
            next_span = [matrix[i + ki, j + kj] for ki in range(span_length) for kj in range(span_length)]
            if np.sum(next_span) >= num_hits:
                ref_token_flags[i] = True
                break

    ref_sent_flags = [any(ref_token_flags[max(0, s - span_length + 1): e]) for s, e in ref_sent_idxes]
    retrieved_doc = None
    if any(ref_sent_flags):
        ref_select_text = ' '.join([x for i, x in enumerate(ref_sentences) if ref_sent_flags[i]])
        hit_ratio = sum(ref_token_flags) / len(target_tokens)
        retrieved_doc = RetrievedDoc(doc_text=ref_select_text, hit_ratio=hit_ratio)

    return retrieved_doc


def run_DJ_attribute(retrieved_data_path, data_output_file, num_hits, span_length,
                     doc_pool_size, num_cpus, subset):
    retrieved_data = json.load(open(retrieved_data_path))[:subset]
    print(f"Data loaded from {retrieved_data_path}: {len(retrieved_data)} documents found")

    if os.path.isfile(data_output_file):
        saved_data = [json.loads(l) for l in open(data_output_file, 'r').readlines()]
        retrieved_data = retrieved_data[len(saved_data):]
        print(f'resume from previous output file with {len(saved_data)} items')

    for t_doc in tqdm(retrieved_data, desc='target docs'):
        target_doc_tokens = tokenize(t_doc['text'])

        if not t_doc['retrieval_details']:
            retrieved_output = {k: v for k, v in t_doc.items() if k != 'retrieval_details'}
            retrieved_output['retrieved_docs'] = {}
            with open(data_output_file, 'a') as f:
                f.write(json.dumps(retrieved_output))
                f.write('\n')
                f.flush()
            continue

        r_docs = [(q_idx, d['_source']['text']) for q_idx, query in enumerate(t_doc['retrieval_details']) for d in query['top_docs']]
        r_docs = list(toolz.unique(r_docs, key=lambda r_doc: r_doc[1]))
        ref_q_idx, ref_texts = [list(x) for x in zip(*r_docs)]

        document_attribution_func = partial(document_attribution, target_doc_tokens, num_hits, span_length)
        with mp.Pool(processes=num_cpus) as pool:
            doc_return = pool.map(document_attribution_func, ref_texts)

        # update the retrieved documents
        retrieved_docs = {k: [] for k in set(ref_q_idx)}
        for q_idx, ret_doc in zip(ref_q_idx, doc_return):
            if ret_doc is not None:
                retrieved_docs[q_idx].append(ret_doc)
        if sum([len(v) for k, v in retrieved_docs.items()]) > doc_pool_size:
            min_hit_ratio = sorted([d.hit_ratio for k, v in retrieved_docs.items() for d in v], reverse=True)[doc_pool_size]
            for q_idx in list(retrieved_docs.keys()):
                sorted_q_docs = sorted(retrieved_docs[q_idx], key=lambda x: x.hit_ratio, reverse=True)
                selected_q_docs = [d for d in sorted_q_docs if d.hit_ratio >= min_hit_ratio]
                selected_q_docs = selected_q_docs if selected_q_docs else [sorted_q_docs[0]]
                retrieved_docs[q_idx] = selected_q_docs
        retrieved_docs = {k: list(map(format_retrieved_doc, v)) for k, v in retrieved_docs.items()}

        retrieved_output = {k: v for k, v in t_doc.items() if k != 'retrieval_details'}
        retrieved_output['retrieved_docs'] = retrieved_docs

        with open(data_output_file, 'a') as f:
            f.write(json.dumps(retrieved_output))
            f.write('\n')
            f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="GPT3_book",
                        help="which type of corpus to analyze")
    parser.add_argument('--retrieved_data_path', type=str,
                        default=f"data/book/retrieved/GPT3_book_nbgens_100_nbdoc100_DOLMA.json")
    parser.add_argument('--data_output_dir', type=str,
                        default=f"data/new_book/filtered")

    parser.add_argument("--num_hits", type=int, default=3,
                        help="minimum number of matched tokens to be considered as a matched span")
    parser.add_argument("--span_length", type=int, default=5,
                        help="minimum covered span to be considered as a matched span")
    parser.add_argument("--doc_pool_size", type=int, default=1000,
                        help="maximum number of documents to keep")
    parser.add_argument("--num_cpus", type=int, default=96,
                        help="number of cpu to use for parallel")

    parser.add_argument("--subset", type=int, default=50,
                        help="size of example subset to run search on")

    args = parser.parse_args()
    os.makedirs(args.data_output_dir, exist_ok=True)
    args.data_output_file = os.path.join(args.data_output_dir, args.task + '_filtered.json')
    print(args)

    run_DJ_attribute(args.retrieved_data_path, args.data_output_file, args.num_hits, args.span_length,
                     args.doc_pool_size, args.num_cpus, args.subset)