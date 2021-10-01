import tempfile
import pickle
import itertools
import math

import torch
import deepimpact
import deepimpact.model

from deepimpact.model import MultiBERT as DeepImpactModel
from deepimpact.utils import load_checkpoint
from deepimpact.utils2 import cleanD

from pyterrier.index import IterDictIndexer

class DeepImpactIndexer(IterDictIndexer):

    def __init__(self, 
                 *args,
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
                for doc in pt.tqdm(doc_iter, desc='Computing the maximum score value and the impacts'):
                    D = [tok(doc['text'])]
                    transformed_doc = self.model.index(D, len(D[-1][0])+2)[0]
                    max_impact = max(max_impact, max(transformed_doc, key=itemgetter(1))[1]) 
                    pickle.dump({'docno': doc['docno'], 'text': transformed_doc}, tmp)

                print('Max impact is', max_impact)
                scale = (1 << self.quantization_bits)/max_impact

                def quantize(transformed_doc, max_impact):
                    transformed_doc = [[term] * int(math.ceil(value * scale)) for term, value in transformed_doc]
                    return ' '.join(itertools.chain.from_iterable(transformed_doc))

                tmp.seek(0)
                while tmp.peek(1):
                    transformed_doc = pickle.load(tmp)
                    quantized_transformed_doc = quantize(transformed_doc['text'], max_impact)
                    yield {'docno': transformed_doc['docno'], 'text': quantized_transformed_doc}

        doc_iter = _deepimpact_iter(doc_iter)
        super().index(doc_iter, *args, **kwargs)