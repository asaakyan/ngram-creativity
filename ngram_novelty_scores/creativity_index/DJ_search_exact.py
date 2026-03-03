import os
import nltk
import json
import time
import requests
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Callable
from dataclasses import dataclass
from unidecode import unidecode
from sacremoses import MosesDetokenizer
from transformers import AutoTokenizer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

md = MosesDetokenizer(lang='en')
HF_TOKEN = "YOUR_HF_TOKEN"
SAVE_INTERVAL = 1000

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=HF_TOKEN,
                                          add_bos_token=False, add_eos_token=False)


@dataclass
class Document:
    doc_id: str
    tokens: List[str]  # [num_tokens]


@dataclass
class Span:
    start_index: int
    end_index: int
    span_text: str
    occurrence: int


class Hypothesis:
    def __init__(self, target_doc: Document, min_ngram: int) -> None:
        self.target_doc = target_doc
        self.min_ngram = min_ngram
        self.spans = []
        self.umatched_spans = []
        self.finished = False

    def add_span(self, new_span: Span, matched=True) -> None:
        if matched:
            self.spans.append(new_span)
            if new_span.end_index >= len(self.target_doc.tokens):
                self.finished = True
        else:
            self.umatched_spans.append(new_span)

    def replace_span(self, new_span: Span, matched = True) -> None:
        if matched:
            self.spans = self.spans[:-1] + [new_span]
            if new_span.end_index >= len(self.target_doc.tokens):
                self.finished = True
        else:
            self.umatched_spans = self.umatched_spans[:-1] + [new_span]

    def get_score(self, matched=True) -> float:
        if not self.spans:
            return 0
        progress_len = self.spans[-1].end_index if not self.finished else len(self.target_doc.tokens)
        if matched:
            flags = [False for _ in range(progress_len)]
            for span in self.spans:
                span_length = span.end_index - span.start_index
                flags[span.start_index: span.end_index] = [True] * span_length
            coverage = len([fa for fa in flags if fa]) / len(flags)
        else:
            flags = [False for _ in range(progress_len)]
            for span in self.umatched_spans:
                span_length = span.end_index - span.start_index
                flags[span.start_index: span.end_index] = [True] * span_length
            coverage = len([fa for fa in flags if fa]) / len(flags)
        return coverage

    def format_span(self) -> str:
        return ' | '.join([s.span_text for s in self.spans])

    def __hash__(self) -> int:
        return hash(self.format_span())

    def __eq__(self, other) -> bool:
        if isinstance(other, Hypothesis):
            return self.format_span() == other.format_span()
        return NotImplemented

    def get_avg_span_len(self) -> float:
        if not self.spans:
            return 0
        span_len = [s.end_index - s.start_index for s in self.spans]
        return sum(span_len) / len(span_len)

    def export_json(self, matched=True) -> dict:
        matched_spans = [{'start_index': s.start_index,
                          'end_index': s.end_index,
                          'span_text': s.span_text,
                          'occurrence': s.occurrence} for s in self.spans]
        unmatched_spans = [{'start_index': s.start_index,
                            'end_index': s.end_index,
                            'span_text': s.span_text} for s in self.umatched_spans]
        return {
            'matched_spans': matched_spans,
            'matched_coverage': self.get_score(matched=True),
            'unmatched_coverage': self.get_score(matched=False),
            'avg_span_len': self.get_avg_span_len(),
            'unmatched_spans': unmatched_spans,
        }

@retry(wait=wait_random_exponential(min=1, max=60), 
           stop=stop_after_attempt(100))
def send_api_request(request_data, cache, cache_file):

    if "dclm" in request_data['corpus']:
        # send api request to infini-gram-mini if dclm corpus
        API_URL = 'https://api.infini-gram-mini.io/'
        request_data['index'] = request_data['corpus']
        del request_data['corpus']
        del request_data['engine']
    else:
        API_URL = 'https://api.infini-gram.io/'

    if not hasattr(send_api_request, "counter"):
        send_api_request.counter = 0  # Initialize the counter

    if cache_file:
        q = request_data['query']
        if q in cache:
            return cache[q], cache
        else:
            # print(f"NOT IN DJ CACHE: {q}")
            res = requests.post(API_URL, json=request_data).json()
            assert "count" in res, f"Error: {res}"
            cache[q] = res
            send_api_request.counter += 1
            if send_api_request.counter % SAVE_INTERVAL == 0:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=4)
                print(f"Saved cache to {cache_file} after {send_api_request.counter} requests")
            return res, cache
        
    res = requests.post(API_URL, json=request_data).json()
    assert "count" in res, f"Error: {res}"
    return res, cache

def find_exact_match(detokenize: Callable, doc: Document, min_ngram: int,
                     cache_file: str = None, 
                     corpus: str = 'v4_dolma-v1_7_llama',
                     verbose=False) -> dict:

    hypothesis = Hypothesis(doc, min_ngram)

    if cache_file and not os.path.exists(cache_file):
        with open(cache_file, 'w') as f:
            json.dump({}, f, indent=4)
        cache = {}
    elif cache_file:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = None

    first_pointer, second_pointer = 0, min_ngram
    while second_pointer <= len(doc.tokens):
        span_text = detokenize(doc.tokens[first_pointer: second_pointer])
        request_data = {
            # 'corpus': 'v4_rpj_llama_s4',
            'corpus': corpus,
            'engine': 'c++',
            'query_type': 'count',
            'query': span_text,
        }
        search_result, cache = send_api_request(request_data, cache, cache_file)
        occurrence = 0 if 'count' not in search_result else search_result['count']

        if occurrence:
            matched_span = Span(start_index=first_pointer,
                                end_index=second_pointer,
                                span_text=span_text,
                                occurrence=occurrence)
            if not hypothesis.spans:
                hypothesis.add_span(matched_span)
            else:
                last_span = hypothesis.spans[-1]
                if matched_span.start_index <= last_span.start_index and last_span.end_index <= matched_span.end_index:
                    hypothesis.replace_span(matched_span)
                else:
                    hypothesis.add_span(matched_span)
            second_pointer += 1
            if verbose:
                print("***************************************************************************************************")
                print(hypothesis.format_span())
                print(f'score: {hypothesis.get_score():.4f}  avg_span_length: {hypothesis.get_avg_span_len()}')
                print("***************************************************************************************************")

        else:
            unmatched_span = Span(start_index=first_pointer,
                                    end_index=second_pointer,
                                    span_text=span_text,
                                    occurrence=occurrence)
            if not hypothesis.spans:
                hypothesis.add_span(unmatched_span, matched=False)
            else:
                last_span = hypothesis.spans[-1]
                if unmatched_span.start_index <= last_span.start_index and last_span.end_index <= unmatched_span.end_index:
                    hypothesis.replace_span(unmatched_span, matched=False)
                else:
                    hypothesis.add_span(unmatched_span, matched=False)
            if second_pointer - first_pointer > min_ngram:
                first_pointer += 1
            elif second_pointer - first_pointer == min_ngram:
                first_pointer += 1
                second_pointer += 1
            else:
                raise ValueError

    hypothesis.finished = True
    return hypothesis.export_json()


def dj_search(data_path, output_file, min_ngram, subset=100, lm_tokenizer=False):
    data = json.load(open(data_path))[:subset]
    if not lm_tokenizer:
        tokenize_func = lambda x: nltk.tokenize.casual.casual_tokenize(x)
        detokenize = lambda x: md.detokenize(x)
    else:
        tokenize_func = lambda x: tokenizer.tokenize(x)
        detokenize = lambda x: tokenizer.decode(tokenizer.convert_tokens_to_ids(x))

    outputs = []
    if os.path.isfile(output_file):
        outputs = json.load(open(output_file, 'r'))
        data = data[len(outputs):]
        print(f'resume from previous output file with {len(outputs)} items')

    for t_idx, t_doc in tqdm(enumerate(data), desc='target docs', total=len(data)):
        tokenized_text = tokenize_func(unidecode(t_doc['text']))
        tgt_doc = Document(f'tgt_{t_idx}', tokenized_text)
        if len(tgt_doc.tokens) <= min_ngram:
            continue

        output = find_exact_match(detokenize, tgt_doc, min_ngram)
        t_doc.update(output)
        outputs.append(t_doc)

        avg_coverage = np.average([x['coverage'] for x in outputs])
        std = np.std([x['coverage'] for x in outputs])
        avg_len = np.average([x['avg_span_len'] for x in outputs])
        print(f'average {min_ngram}-ngram coverage: {avg_coverage:.3f}, std: {std:.3f}, average length: {avg_len}')

        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=4)
            f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='GPT3_book',
                        help="which type of corpus to analyze")
    parser.add_argument('--data', type=str,
                        default='data/book/GPT3_book.json')
    parser.add_argument('--output_dir', type=str,
                        default=f'outputs/exact/book')
    parser.add_argument("--min_ngram", type=int, default=5,
                        help="minimum n-gram size")
    parser.add_argument("--subset", type=int, default=100,
                        help="size of example subset to run search on")
    parser.add_argument('--lm_tokenizer', action='store_true',
                        help='whether to LM tokenizer instead of whitespace tokenizer')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.task + '.json')
    dj_search(args.data, args.output_file, args.min_ngram, args.subset, args.lm_tokenizer)


if __name__ == '__main__':
    main()
    