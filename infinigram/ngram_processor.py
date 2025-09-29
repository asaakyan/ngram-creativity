"""N-gram generation and processing utilities."""
import nltk
import random
import string
import re
from typing import List, Union
from unidecode import unidecode

class NgramProcessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('punkt_tab')

    def get_sents(self, paragraph: str,) -> List[str]:
        """Get words from a paragraph sentence-by-sentence."""
        sents = nltk.sent_tokenize(paragraph)
        out_sents = []
        for sent in sents:
            if sent [-1] in string.punctuation:
                sent = sent[:-1]
            sent = sent.split(" ")
            out_sents.append(sent)
        return out_sents
        
    def generate_ngrams(self, paragraph: str, n: int,
                        within: Union["sent", "punct", False] = "sent",
                       perc_sample: float = 0) -> List[str]:
        """Generate n-grams from a paragraph with optional sampling."""

        if within == "sent":
            sents = nltk.sent_tokenize(paragraph)
        elif within == "punct":
            # split by punctuation
            pattern = r'[,.!?;]+'
            # Split by punctuation and strip whitespace
            sents = [part.strip() for part in re.split(pattern, paragraph) if part.strip()]
        else:
            sents = [paragraph]
        ngrams = []
        
        for sent in sents:
            # words = sent.split(" ")
            words = nltk.tokenize.casual.casual_tokenize(unidecode(sent))
            # print(words)
            # remove beginning and trailing punctuation
            while words and words[0] in string.punctuation:
                words = words[1:]
            while words and words[-1] in string.punctuation:
                words = words[:-1]
            # if words[0][0] in string.punctuation:
            #     words[0] = words[0][1:]
            # if words[-1][-1] in string.punctuation:
            #     words[-1] = words[-1][:-1]
            n = min(n, len(words))
            # print(n)
            # ngrams.extend([" ".join(l) for l in list(nltk.ngrams(words, n))])
            # ngrams.extend(nltk.ngrams(words, n))
            # TODO: figure out how to get list of ngrams [["word1", "word2"], ["word3", "word4"]]
            ngrams.extend([list(l) for l in list(nltk.ngrams(words, n))])
            # ngrams.extend(list(nltk.ngrams(words, n)))
            
        if perc_sample:
            n_sample = max(1, int(len(ngrams) * perc_sample))
            return random.sample(ngrams, n_sample)
        return ngrams
    
