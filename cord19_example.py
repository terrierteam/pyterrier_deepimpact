import pyterrier as pt
pt.init()

from pyterrier_deepimpact import DeepImpactIndexer

def text_iter(doc_iter):
    for doc in doc_iter:
        yield {"docno": doc['docno'], "text": '{title} {abstract}'.format(**doc)}


dataset = pt.get_dataset('irds:cord19/trec-covid')
indexer = DeepImpactIndexer('./di_index1', batch_size=1)
indexer.index(text_iter(dataset.get_corpus_iter()))
