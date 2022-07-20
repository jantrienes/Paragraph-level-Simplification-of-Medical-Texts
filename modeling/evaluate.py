import json

from easse.sari import corpus_sari
from easse.bleu import corpus_bleu
from utils import calculate_rouge

with open('trained_models/bart-no-ul/gen_nucleus_test_1_0-none.json') as fin:
    sys_sents = json.load(fin)
    sys_sents = [x['gen'] for x in sys_sents]
with open('data/data-1024/test.source') as fin:
    orig_sents = [l.strip() for l in fin.readlines()]
with open('data/data-1024/test.target') as fin:
    refs_sents = [l.strip() for l in fin.readlines()]

print(len(sys_sents))
print(len(orig_sents))
print(len(refs_sents))

scores = calculate_rouge(sys_sents, refs_sents)
print('R-1 = {:.2f}'.format(scores['rouge1']))
print('R-2 = {:.2f}'.format(scores['rouge2']))
print('R-L = {:.2f}'.format(scores['rougeLsum']))

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