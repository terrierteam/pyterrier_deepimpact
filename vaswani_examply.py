import pyterrier as pt
pt.init()

from pyterrier_deepimpact import DeepImpactIndexer

indexer = DeepImpactIndexer('./di_index1', batch_size=32)
indexer.index(pt.get_dataset("vaswani").get_corpus_iter())

index_di = pt.IndexFactory.of('./di_index1')
index = pt.get_dataset("vaswani").get_index()

df = pt.Experiment([
        pt.BatchRetrieve(index, wmodel="BM25"),
        pt.BatchRetrieve(index_di, wmodel="Tf")
    ],
    pt.get_dataset("vaswani").get_topics(),
    pt.get_dataset("vaswani").get_qrels(),
    names=['bm25', "deep_impact"],
    eval_metrics=["map", "recip_rank", "ndcg_cut_10"]
)

print(df)