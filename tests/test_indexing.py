import unittest
import pandas as pd
import tempfile

class TestIndexing(unittest.TestCase):

    def test_indexing_1doc_torch(self):
        import pyterrier as pt
        from pyt_deepimpact import DeepImpactIndexer
        import os
        os.rmdir(self.test_dir)

        parent = pt.index._IterDictIndexer_nofifo(os.path.join(self.test_dir, "index"))
        parent.setProperty("termpipelines", "")
        indexer = DeepImpactIndexer(parent, batch_size=2, gpu=False)
        iter = pt.get_dataset("vaswani").get_corpus_iter()
        indexer.index([ next(iter) for i in range(200) ])

        index_ref = pt.IndexRef.of(indexer.index_dir + "/data.properties")
        index = pt.IndexFactory.of(index_ref)

        self.assertIsNotNone(index)
        self.assertEqual(200, index.getCollectionStatistics().getNumberOfDocuments())

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass
