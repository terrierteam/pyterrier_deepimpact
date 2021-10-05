import os
import tempfile
import pickle
import itertools
import math
from more_itertools import chunked

import torch
import deepimpact
import deepimpact.model

from deepimpact.model import MultiBERT as DeepImpactModel
from deepimpact.utils2 import cleanD

import pyterrier as pt
from pyterrier.index import IterDictIndexer

def _load_model(checkpoint_gdrive_id, base_model):

    from deepimpact.utils import load_checkpoint

    checkpoint_path='https://drive.google.com/uc?id=' + checkpoint_gdrive_id
    print("Downloading checkpoint %s" % checkpoint_path)
    import tempfile, gdown
    targetFile = os.path.join(tempfile.mkdtemp(), 'checkpoint.dnn')
    gdown.download(checkpoint_path, targetFile, quiet=False)
    checkpoint_path = targetFile

    print("Loading checkpoint %s" % checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model = DeepImpactModel.from_pretrained(base_model)
    model.to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()      

    return model

from pyterrier.index import IterDictIndexerBase
class DeepImpactIndexer(IterDictIndexerBase):

    def __init__(self, 
                parent_indexer : IterDictIndexerBase,
                 *args,
                 batch_size=1,
                 quantization_bits=8,
                 checkpoint_gdrive_id='17I2TWCB2hBSQ-E0Yt2sBEDH2z_rV0BN0',
                 base_model='bert-base-uncased',
                 gpu=True,
                 **kwargs):
                 
        super().__init__(*args, **kwargs)
        self.parent = parent_indexer
        self.model = _load_model(checkpoint_gdrive_id, base_model)
        self.quantization_bits=quantization_bits
        self.batch_size=batch_size
        if not gpu:
            import deepimpact.parameters, torch
            deepimpact.model.DEVICE = deepimpact.parameters.DEVICE = torch.device("cpu")

    def index(self, doc_iter, *args, **kwargs):
        
        def _deepimpact_iter(doc_iter):

            def tok(d):
                from itertools import accumulate

                d = cleanD(d, join=False)
                content = ' '.join(d)
                tokenized_content = self.model.tokenizer.tokenize(content)

                terms = list(set([(t, d.index(t)) for t in d]))  # Quadratic!
                word_indexes = list(accumulate([-1] + tokenized_content, lambda a, b: a + int(not b.startswith('##'))))
                terms = [(t, word_indexes.index(idx)) for t, idx in terms]
                terms = [(t, idx) for (t, idx) in terms if idx < deepimpact.model.MAX_LENGTH]

                return tokenized_content, terms

            max_impact = 0.0
            with tempfile.NamedTemporaryFile() as tmp:
                from operator import itemgetter

                for batch in pt.tqdm(chunked(doc_iter, self.batch_size), desc='Computing the maximum score value and the impacts'):
                    batch = [(doc['docno'], tok(doc['text'])) for doc in batch]
                    batch = sorted(batch, key=lambda x: len(x[1][0]))
                    docnos, D = zip(*batch)
                    transformed_docs = self.model.index(D, 2 + len(D[-1][0]))
                    for docno, doc in zip(docnos, transformed_docs):
                        max_impact = max(max_impact, max(doc, key=itemgetter(1))[1])
                        pickle.dump({'docno': docno, 'text': doc}, tmp)

                # print('Max impact is', max_impact)
                scale = (1 << self.quantization_bits)/max_impact

                def quantize(transformed_doc):
                    transformed_doc = [[term] * int(math.ceil(value * scale)) for term, value in transformed_doc]
                    return ' '.join(itertools.chain.from_iterable(transformed_doc))

                encountered_docnos = set() # required to remove duplicates in cord19-like datasets :-(
                tmp.seek(0)
                while tmp.peek(1):
                    doc = pickle.load(tmp)
                    if doc['docno'] not in encountered_docnos:
                        q_text = quantize(doc['text'])
                        yield {'docno': doc['docno'], 'text': q_text}
                        encountered_docnos.add(doc['docno'])

        doc_iter = _deepimpact_iter(doc_iter)
        return self.parent.index(doc_iter, *args, **kwargs)