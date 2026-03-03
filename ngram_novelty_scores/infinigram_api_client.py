"""API client for infinigram service."""
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from typing import Dict, Tuple, List, Union
import json
import signal
import sys
import os

class InfinigramClient:

    def __init__(self, api_url: str, search_idx: str, 
                 cache_file: Union[None, str] = None,
                save_frequency: int = 10) -> None:
        self.api_url = api_url
        self.search_idx = search_idx
        self.request_count = 0
        self.save_frequency = save_frequency
        if cache_file is None:
            self.cache = {"count": {}, "infgram_prob": {}}
        else:
            signal.signal(signal.SIGINT, self._handle_exit)
            self.cache_file = cache_file
            # if cache file does not exist
            if not os.path.exists(cache_file):
                self.cache = {"count": {}, "infgram_prob": {}}
                with open(cache_file, "w") as f:
                    json.dump(self.cache, f, indent=2)
            with open(cache_file, "r") as f:
                self.cache = json.load(f)

    def save_cache(self):
        """Save the current cache to file."""
        if hasattr(self, 'cache_file'):
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)

    def _handle_exit(self, sig, frame):
        print('\nSaving cache before exit...')
        self.save_cache()
        print('Cache saved. Exiting gracefully.')
        sys.exit(0)
    
    def _maybe_save_cache(self):
        """Save cache if request count is a multiple of save_frequency."""
        self.request_count += 1
        if hasattr(self, 'cache_file') and self.request_count % self.save_frequency == 0:
            self.save_cache()
            print(f"Cache saved to {self.cache_file} after {self.request_count} requests.")

    @retry(wait=wait_random_exponential(min=1, max=60), 
           stop=stop_after_attempt(100))
    def get_occurrence_counts(self, ngram: str,
                            verbose: bool = False) -> Tuple[int, Dict]:
        """Get occurrence counts for a single n-gram."""
        # print(ngram)
        # print(ngram in self.cache['count'])
        if ngram in self.cache['count']:
            result = self.cache['count'][ngram]
            return result['count'], result
        # print("NOT IN CACHE: ", ngram)
        payload = {
            'index': self.search_idx,
            'query_type': 'count',
            'query': ngram,
        }
        result = requests.post(self.api_url, json=payload).json()
        assert "count" in result, f"Error: {result}"
        # update cache and update cach file
        self.cache['count'][ngram] = result
        self._maybe_save_cache()  # Instead of saving every time

        if verbose:
            print(self.search_idx)
            print(result)
                
        return result['count'], result
    
    @retry(wait=wait_random_exponential(min=1, max=60), 
           stop=stop_after_attempt(100))
    def infgram_prob(self, ngram: str):
        """Calculate the probability of the last word in ngram using Infinigram."""
        
        if ngram in self.cache['infgram_prob']:
            # print("IN INFPROB CACHE: ", ngram)
            result = self.cache['infgram_prob'][ngram]
            return result
        # print("NOT IN INFPROB CACHE: ", ngram)
        payload = {
            'index': self.search_idx,
            'query_type': 'infgram_prob',
            'query': ngram,
        }
        result = requests.post(self.api_url, json=payload).json()
        assert "prob" in result, f"Error: {result}"

        # update cache and update cach file
        self.cache['infgram_prob'][ngram] = result
        self._maybe_save_cache() 

        return result