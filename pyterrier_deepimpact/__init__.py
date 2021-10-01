import tempfile
import pickle
import itertools
import math
from more_itertools import chunked

import torch
import deepimpact
import deepimpact.model

from deepimpact.model import MultiBERT as DeepImpactModel
from deepimpact.utils import load_checkpoint
from deepimpact.utils2 import cleanD

import pyterrier as pt
from pyterrier.index import IterDictIndexer

class DeepImpactIndexer(IterDictIndexer):

    def __init__(self, 
                 *args,
                 batch_size=1,
                 quantization_bits=8,
                 checkpoint='colbert-test-150000.dnn', 
                 base_model='bert-base-uncased',
                 **kwargs):
        super().__init__(*args, **kwargs)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        self.model = DeepImpactModel.from_pretrained(base_model)
        self.model.to(device)
        load_checkpoint(checkpoint, self.model)
        self.model.eval()      
        self.quantization_bits=quantization_bits
        self.batch_size=batch_size

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

                print('Max impact is', max_impact)
                scale = (1 << self.quantization_bits)/max_impact

                def quantize(transformed_doc):
                    transformed_doc = [[term] * int(math.ceil(value * scale)) for term, value in transformed_doc]
                    return ' '.join(itertools.chain.from_iterable(transformed_doc))

                tmp.seek(0)
                while tmp.peek(1):
                    doc = pickle.load(tmp)
                    q_text = quantize(doc['text'])
                    yield {'docno': doc['docno'], 'text': q_text}

        doc_iter = _deepimpact_iter(doc_iter)
        super().index(doc_iter, *args, **kwargs)