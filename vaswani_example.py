import pyterrier as pt
pt.init()

import os
from pyterrier_deepimpact import DeepImpactIndexer

vaswani = pt.datasets.get_dataset("vaswani")

pt_index_path = './terrier_di_vaswani'
if not os.path.exists(pt_index_path + "/data.properties"):
    indexer = DeepImpactIndexer(pt_index_path, batch_size=32)
    indexer.setProperty("termpipelines", "")
    indexer.index(vaswani.get_corpus_iter())

index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")
index_di = pt.IndexFactory.of(index_ref)


pt_index_path = './terrier_vaswani'
if not os.path.exists(pt_index_path + "/data.properties"):
    indexer = pt.index.IterDictIndexer(pt_index_path)
    indexer.setProperty("termpipelines", "")
    index_ref = indexer.index(vaswani.get_corpus_iter())

index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")
index = pt.IndexFactory.of(index_ref)

df = pt.Experiment([
        pt.BatchRetrieve(index, wmodel="BM25", properties={"termpipelines" : ""}),
        pt.BatchRetrieve(index_di, wmodel="Tf", properties={"termpipelines" : ""})
    ],
    vaswani.get_topics(),
    vaswani.get_qrels(),
    names=['bm25', "deep_impact"],
    eval_metrics=["map", "recip_rank", "ndcg_cut_10"]
)

print(df)