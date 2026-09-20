import unittest
from unittest.mock import Mock
from kb.retrieval import Retriever


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.encoder=Mock()
        self.encoder.encode.return_value=[[1,0]]
        self.nearest=Mock()
        self.best={'id':'faq-1','question':'Where?','answer':'Original answer',
                   'category':'portal','language':'en','source_row':'12','score':0.85}
        self.nearest.return_value=[self.best]
        self.retriever=Retriever(self.encoder,self.nearest,0.8)

    def test_match_preserves_answer_and_source(self):
        r=self.retriever.retrieve('  Справка?  ')
        self.assertEqual(r['status'],'matched')
        self.assertEqual(r['answer'],'Original answer')
        self.assertEqual(r['source']['id'],'faq-1')
        self.encoder.encode.assert_called_once_with(['Справка?'],query=True)
        self.nearest.assert_called_once_with([1,0],top_k=1)

    def test_below_threshold_hides_answer_and_source(self):
        self.best['score']=0.7999999
        r=self.retriever.retrieve('x')
        self.assertFalse(r['matched'])
        self.assertEqual(r['status'],'not_found')
        self.assertIsNone(r['answer']);self.assertIsNone(r['source'])

    def test_exact_threshold_is_inclusive(self):
        self.best['score']=0.8
        self.assertTrue(self.retriever.retrieve('x')['matched'])

    def test_empty_results(self):
        self.nearest.return_value=[]
        r=self.retriever.retrieve('x')
        self.assertFalse(r['matched']);self.assertIsNone(r['score'])

    def test_bad_thresholds(self):
        for t in [-2,2,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):Retriever(self.encoder,self.nearest,t)

    def test_bad_question_does_not_call_model(self):
        for q in ['', '  ',None,123,'a'*2001]:
            with self.assertRaises(ValueError):self.retriever.retrieve(q)
        self.encoder.encode.assert_not_called()

    def test_storage_error_is_not_not_found(self):
        self.nearest.side_effect=ConnectionError('DB offline')
        with self.assertRaises(ConnectionError):self.retriever.retrieve('x')

    def test_bad_score_and_empty_answer_are_errors(self):
        for value in [float('nan'),float('inf'),2]:
            self.best['score']=value
            with self.assertRaises(ValueError):self.retriever.retrieve('x')
        self.best['score']=1;self.best['answer']=' '
        with self.assertRaises(ValueError):self.retriever.retrieve('x')


if __name__=='__main__':unittest.main()
