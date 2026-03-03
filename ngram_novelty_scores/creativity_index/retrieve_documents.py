import argparse
import json
import logging
import time
from pathlib import Path

from elasticsearch import Elasticsearch
from tqdm import tqdm

# Changed to NLTK sent tokenize
from nltk import sent_tokenize
import unidecode

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname).1s %(asctime)s [ %(message)s ]",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("search_index")

API_KEY = "YOUR_API_KEY"

def clean_text(text):
    return unidecode.unidecode(text)

def _merge(ranked_results1, ranked_results2, topk: int = 0):
    """
    merge two ranked lists based on their retrieval scores
    """
    merged_results = []
    i, j = 0, 0
    while i < len(ranked_results1) or j < len(ranked_results2):
        doc_i = ranked_results1[i] if i < len(ranked_results1) else None
        doc_j = ranked_results2[j] if j < len(ranked_results2) else None

        if not doc_j or not doc_i:
            merged_results.append(doc_i or doc_j)
            if not doc_j:
                i += 1
            else:
                j += 1
        elif doc_i["_score"] > doc_j["_score"]:
            merged_results.append(doc_i)
            i += 1
        elif doc_i["_score"] < doc_j["_score"]:
            merged_results.append(doc_j)
            j += 1
        else:
            merged_results.append(doc_i)
            merged_results.append(doc_j)
            i += 1
            j += 1

        if 0 < topk <= len(merged_results):
            break

    return merged_results


def search_index(es, query, nb_documents, indices=None):
    # segment can be an 1-gram, 2-gram, I can play with that. Does it do exact match the API??
    # return documents for each query n-gram
    # maybe one sentence is constructed based on multiple documents.

    if not indices:
        indices = ["c4", "openwebtext", "re_pile"]

    if isinstance(indices, str):
        indices = [indices]

    c4_present = "c4" in indices
    if c4_present:
        indices = [c for c in indices if c != "c4"]

    if indices:
        results = es.search(
            index=indices,
            size=nb_documents,
            body={"query": {"bool": {"must": {"match": {"text": query}}}}},
        )["hits"]["hits"]
    else:
        results = []

    if c4_present:
        c4_results = es.search(
            index="c4",
            size=nb_documents,
            body={"query": {"bool": {"must": {"match": {"text": query}}, "filter": {"term": {"subset": "en"}}}}},
        )["hits"]["hits"]

        return _merge(results, c4_results, nb_documents)
    else:
        return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--input_file", type=str, default=None, help="input directory")
    parser.add_argument("--nb_documents", type=int, default=100, help="Number of retrieved documents")

    parser.add_argument(
        "--indices",
        type=str,
        nargs="*",
        default=("c4", "openwebtext", "re_pile"),
        # default=("docs_v1.5_2023-11-02"),
        choices=("c4", "openwebtext", "re_pile", "re_oscar", "s2orc-abstracts", "re_laion2b-en-1", "re_laion2b-en-2"),
        help="index names to search within",
    ) # Not used
    parser.add_argument("--data_type", type=str, default=None, help="data type (e.g., math, letter, etc)")
    parser.add_argument("--index", type=str, default="WIMD", help="DOLMA or WIMD")
    parser.add_argument("--subset", type=int, default=100, help="size of example subset to run search on")

    args = parser.parse_args()

    if args.index == "WIMD":
        es = Elasticsearch(
            cloud_id="lm-datasets:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDk1N2U5ODIwZDUxNTQ0YWViMjk0MmQwNzI1NjE0OTQ2JDhkN2M0OWMyZDEzMTRiNmM4NDNhNGEwN2U4NDE5NjRl",
            api_key=API_KEY,
            retry_on_timeout=True,
            http_compress=True,
            timeout=180,
            max_retries=10)
        indices = ("c4", "openwebtext", "re_pile")

    elif args.index == "DOLMA":
       es = Elasticsearch(
            cloud_id="dolma-v15:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ1MjQyM2ZiNjk0NGE0YzdkOGQ5N2Y3NDM2MmMzODY3ZSQxMDNiM2ZkYTUwYzk0MTNmYmUwODA1ZDMyNjQ5YTliNQ==",
            api_key=API_KEY,
            retry_on_timeout=True,
            http_compress=True,
            timeout=180,
            max_retries=10,
        )
       indices = ("docs_v1.5_2023-11-02")

    # indices = [name for name in es.indices.get(index="*").keys() if not name.startswith(".")]
    # breakpoint()
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    name = Path(args.input_file).stem
    # indices_str = "_".join(indices) # Not used

    doc_list = []

    with open(args.input_file, "r") as f:
        data = json.load(f)[:args.subset]

    total_runtime = 0
    num_sents = 0
    nbgens = len(data)
    output_path = out / f"{name}_nbgens_{nbgens}_nbdoc{args.nb_documents}_{args.index}.json"
    for i, cur_data in enumerate(tqdm(data, desc="iterating through data")):
        generation = cur_data["text"]
        
        # Clean the text
        generation = clean_text(generation)
        sentences = sent_tokenize(generation)
        doc_details = []
        # doc_sources = set()
        for segment in tqdm(sentences, desc="reading sentences", leave=False):
            begin = time.perf_counter()
            try:
                output = search_index(es, segment, args.nb_documents, indices)
            except:
                print('Elastic Search API failed')
                continue
            end = time.perf_counter()
            runtime = end - begin
            top_documents = output

            doc_details.append({"query": segment, "top_docs": top_documents, "retrieval_runtime": runtime})

        cur_data["retrieval_details"] = doc_details
        runtime_per_doc = sum(d["retrieval_runtime"] for d in doc_details)

        total_runtime += runtime_per_doc
        num_sents += len(sentences)
        runtime_per_doc += runtime_per_doc / num_sents
    print(f"Average runtime (top-{args.nb_documents}) per sentence: {total_runtime / num_sents:.3f} seconds")
    print(f"Average runtime (top-{args.nb_documents}) per doc: {runtime_per_doc / len(data):.3f} seconds")
    print(f"Total runtime (top-{args.nb_documents}): {total_runtime} seconds")

    with open(output_path, "w") as out:
        json.dump(data, out, indent=4)


if __name__ == "__main__":
    main()


