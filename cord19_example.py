import pyterrier as pt
pt.init()

import os
from pyterrier_deepimpact import DeepImpactIndexer

def text_iter(doc_iter):
    encountered_docnos = set() # required to remove duplicates in cord19-like datasets :-(

    for doc in doc_iter:
        # Skipping over empty docs
        if len(doc['title'].strip()) == 0 or len(doc['abstract'].strip()) == 0:
            continue
        # Skipping over duplicate docs and merging fields
        if doc['docno'] not in encountered_docnos:
            yield {"docno": doc['docno'], "text": '{title} {abstract}'.format(**doc)}
            encountered_docnos.add(doc['docno'])

cord19 = pt.datasets.get_dataset('irds:cord19/trec-covid')

pt_index_path = './terrier_di_cord19'
if not os.path.exists(pt_index_path + "/data.properties"):
    parent = pt.index.IterDictIndexer(pt_index_path)
    parent.setProperty("termpipelines", "")
    indexer = DeepImpactIndexer(parent, batch_size=32)
    indexer.index(text_iter(cord19.get_corpus_iter()))

index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")
index_di = pt.IndexFactory.of(index_ref)

pt_index_path = './terrier_cord19'
if not os.path.exists(pt_index_path + "/data.properties"):
    indexer = pt.index.IterDictIndexer(pt_index_path)
    indexer.setProperty("termpipelines", "")
    index_ref = indexer.index(text_iter(cord19.get_corpus_iter()))

index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")
index = pt.IndexFactory.of(index_ref)


df = pt.Experiment([
        pt.BatchRetrieve(index, wmodel="BM25"),
        pt.BatchRetrieve(index_di, wmodel="Tf")
    ],
    cord19.get_topics(variant='description'),
    cord19.get_qrels(),
    names=['BM25 (unstemmed)', "Deep Impact"],
    eval_metrics=["map", "recip_rank", "ndcg_cut_10"]
)

print(df)