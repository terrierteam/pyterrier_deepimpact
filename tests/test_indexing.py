import unittest
import pandas as pd
import tempfile

class TestIndexing(unittest.TestCase):

    def test_indexing_1doc_torch(self):
        import pyterrier as pt
        from pyt_deepimpact import DeepImpactIndexer
        import os
        os.rmdir(self.test_dir)

        indexer = DeepImpactIndexer(os.path.join(self.test_dir, "index"), batch_size=2)
        indexer.setProperty("termpipelines", "")
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