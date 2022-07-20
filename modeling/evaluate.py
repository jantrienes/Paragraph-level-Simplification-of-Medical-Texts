import json
import re
import string
from typing import Iterable

from easse.sari import corpus_sari
from easse.bleu import corpus_bleu
from utils import calculate_rouge
from nltk import ngrams, sent_tokenize, word_tokenize
import pandas as pd

from collections import defaultdict

SRC_PATH = 'data/data-1024/test.source'
TGT_PATH = 'data/data-1024/test.target'
RUN_PATH = 'trained_models/bart-no-ul-reproduction/gen_nucleus_test_1_0-none.json'

def load_data():
    with open(RUN_PATH) as fin:
        sys_sents = json.load(fin)
        sys_sents = [x['gen'] for x in sys_sents]
    with open(SRC_PATH) as fin:
        orig_sents = [l.strip() for l in fin.readlines()]
    with open(TGT_PATH) as fin:
        refs_sents = [l.strip() for l in fin.readlines()]

    print(f'len(sys_sents) = {len(sys_sents)}')
    print(f'len(orig_sents) = {len(orig_sents)}')
    print(f'len(refs_sents) = {len(refs_sents)}')

    return sys_sents, orig_sents, refs_sents


class Aggregator:
    def __init__(self):
        self.stats = defaultdict(list)

    def add(self, scores):
        for metric, value in scores.items():
            self.stats[metric].append(value)


def clean(s):
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s\s+", " ", s)
    s = s.lower()
    return s


def novel_ngrams(a: str, b: str, n=2):
    """Count number of n-grams in b but not in a."""
    a, b = word_tokenize(clean(a)), word_tokenize(clean(b))
    a = set(ngrams(a, n=n))
    b = set(ngrams(b, n=n))
    novel = len(b - a)
    total = len(b)
    return novel, total


def novelty(a, b, n):
    """
    Fraction of n-grams in b but not in a.
    If there are no n-grams in b, this is novelty=0.
    """
    novel, total = novel_ngrams(a, b, n=n)
    if total == 0:
        return 0
    return novel / total


def calculate_statistics(docs: Iterable[str], summaries: Iterable[str]):
    """Surface level statistics comparing documents with summaries.
    Sentence boundaries should be indicated with `<q>` and text is expected to be pre-tokenized.
    """

    agg = Aggregator()

    for doc, summary in zip(docs, summaries):
        doc_sents = sent_tokenize(doc)
        doc_tokens = word_tokenize(doc)
        
        sum_sents = sent_tokenize(summary)
        sum_tokens = word_tokenize(summary)

        n_words_doc = len(doc_tokens)
        n_words_summary = len(sum_tokens)
        cmp_w = 1 - n_words_summary / n_words_doc

        n_sents_doc = len(doc_sents)
        n_sents_summary = len(sum_sents)
        cmp_s = 1 - n_sents_summary / n_sents_doc

        novelty_uni = novelty(doc, summary, n=1)
        novelty_bi = novelty(doc, summary, n=2)

        agg.add(
            {
                "n_words_doc": n_words_doc,
                "n_sents_doc": n_sents_doc,
                "n_words_summary": n_words_summary,
                "n_sents_summary": n_sents_summary,
                "cmp_w": cmp_w,
                "cmp_s": cmp_s,
                "novelty_uni": novelty_uni,
                "novelty_bi": novelty_bi,
            }
        )

    return pd.DataFrame(agg.stats)


def main():
    sys_sents, orig_sents, refs_sents = load_data()

    rouge = calculate_rouge(sys_sents, refs_sents)
    print('R-1 = {:.2f}'.format(rouge['rouge1']))
    print('R-2 = {:.2f}'.format(rouge['rouge2']))
    print('R-L = {:.2f}'.format(rouge['rougeLsum']))

    bleu = corpus_bleu(
        sys_sents=sys_sents,
        refs_sents=[[t for t in refs_sents]],
        lowercase=False
    )
    print(f'BLEU = {bleu:.2f}')

    sari = corpus_sari(
        orig_sents=orig_sents,  
        sys_sents=sys_sents, 
        refs_sents=[[t for t in refs_sents]]
    )
    print(f'SARI = {sari:.2f}')

    df_gold_stats = calculate_statistics(orig_sents, refs_sents)
    df_sys_stats = calculate_statistics(orig_sents, sys_sents)

    print('Surface statistics (gold)')
    print(df_gold_stats.mean().round(2))
    print('Surface statistics (system)')
    print(df_sys_stats.mean().round(2))

if __name__ == "__main__":
    main()