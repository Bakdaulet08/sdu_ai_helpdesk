import unittest
from kb.ranking import select

class RankingTests(unittest.TestCase):
    def test_acronym_is_not_moodle(self):
        self.assertIsNone(select('MDE деген не?', [dict(question='What is a moodle?', score=.9)]))

    def test_definition_is_not_retake(self):
        self.assertIsNone(select('MDE деген не?', [dict(question='MDE 160 retake?', score=.9)]))

    def test_portal_definition_is_not_technical_failure(self):
        self.assertIsNone(select('Портал деген не?', [dict(question='Error in portal?', score=.9)]))

    def test_advisor_definition(self):
        right=dict(question='Куратор немесе эдвайзер кім және оларға қалай хабарласамын?',score=.85)
        wrong=dict(question='Эдвайзер кетіп қалса басқа эдвайзерді таңдай аламыз ба?',score=.95)
        self.assertEqual(select('Эдвайзер деген кім?', [wrong,right]),right)

    def test_certificate_over_generic_semantic_similarity(self):
        right=dict(question='How to get certificates',score=.84)
        wrong=dict(question='Сколько дней в неделю можно учиться?',score=.90)
        self.assertEqual(select('Как получить справку?', [wrong,right]),right)

    def test_definition_tolerance_is_bounded(self):
        from unittest.mock import Mock
        from kb.retrieval import Retriever
        from kb.ranking import select
        item=dict(id='advisor', question='Куратор немесе эдвайзер кім және оларға қалай хабарласамын?', answer='Definition', score=.793)
        encoder=Mock(); encoder.encode.return_value=[[1]]
        retriever=Retriever(encoder,lambda *a,**k:[item],selector=select)
        self.assertTrue(retriever.retrieve('Эдвайзер деген кім?')['matched'])
        item['score']=.77
        self.assertFalse(retriever.retrieve('Эдвайзер деген кім?')['matched'])
