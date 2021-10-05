# PyTerrier - DeepImpact Plugin

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [DeepImpact](https://github.com/DI4IR/SIGIR2021) approach for sparse retrieval learning [Mallia20].

## Installation

This repository can be installed using `pip`:

```shell
pip install --upgrade git+https://github.com/tonellotto/pyterrier_di.git
```

You will also need a fine-tuned checkpoint (unzipped) for the DeepImpact model from [here](https://drive.google.com/file/d/17I2TWCB2hBSQ-E0Yt2sBEDH2z_rV0BN0/view?usp=sharing).

## What does it do?

A `DeepImpactIndexer`is a PyTerrier indexer. When created, such indexer requires the following specific parameters:

* `batch_size` (default 1): the number of documents to process in a single forward pass.
* `quantization_bits` (default 8): the number of bits to use for impact scores quantisation.
* `checkpoint` (default `'colbert-test-150000.dnn'`): the fine-tuned DeepImpact checkpoint to use.
* `base_model` (default `'bert-base-uncased'`): the base model to use with the fine-tuned checkpoint.

Once created, the indexer can be used by calling the `index(doc_iter)` method, passing a document iterator obect `doc_iter`.

```python
vaswani = pt.datasets.get_dataset("vaswani")

indexer = DeepImpactIndexer('./terrier_di_vaswani', batch_size=32)
indexer.setProperty("termpipelines", "")
indexer.index(vaswani.get_corpus_iter())
```

It is important to disable stemming and stopword removal from indexing and retrieval, since the DeepImpact model has been trained without stemming, and with stopwords in place.

At retrieval time, no special stuff is required. We can load the inverted index with basic PyTerrier transformers, but remember to:
* disable term pipelines, e.g., `properties={"termpipelines" : ""}` (since Terrier 5.6 you don't need to set the properties for `termpipelines` - this is loaded based on what was set at indexing time);
* use sum as weighting model, since we will sum up quantised impacts, e.g., `wmodel="Tf"`.

```python
index_ref = pt.IndexRef.of('./terrier_di_vaswani' + "/data.properties")
index_di = pt.IndexFactory.of(index_ref)

df = pt.Experiment(
    [pt.BatchRetrieve(index_di, wmodel="Tf")],
    vaswani.get_topics(),
    vaswani.get_qrels(),
    names=["deep_impact"],
    eval_metrics=["map", "recip_rank", "ndcg_cut_10"]
)
```

## Examples
Checkout our scripts:

* Vaswani [[Github]](https://github.com/tonellotto/pyterrier_di/blob/master/vaswani_example.py)
* TREC CORD 19 [[Github]](https://github.com/tonellotto/pyterrier_di/blob/master/cord19_example.py)

## References

[Mallia20] A. Mallia, O. Khattab, T. Suel, N. Tonellotto. *Learning Passage Impacts for Inverted Indexes*, ACM SIGIR 2021 [[link]](https://arxiv.org/abs/2104.12016)

## Credits

* Nicola Tonellotto, University of Pisa
* Sean Macavaney, University of Glasgow
