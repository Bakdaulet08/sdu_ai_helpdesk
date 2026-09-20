from contextlib import closing
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
from kb.api import create_app
from kb.audit import RequestJournal
from kb.generation import GenerationError


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'requests.sqlite3'
        self.journal = RequestJournal(self.path)
        self.service = Mock()
        self.answer = dict(status='answered', forum_answer='Original answer',
            ai_explanation='Explanation', language='ru', basis='retrieved', reason=None,
            source={'id':'faq1'}, source_type='faq', retrieval_score=0.9)
        self.service.answer.return_value = self.answer
        self.client = self.enterContext(TestClient(create_app(self.service, self.journal)))

    def rows(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute('SELECT request_id, question, status, http_status, response_json FROM answer_requests').fetchall()

    def test_success_and_persistent_log(self):
        response = self.client.post('/generate-answer', json={'question':'  Вопрос?  '})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['forum_answer'], 'Original answer')
        self.service.answer.assert_called_once_with('Вопрос?', 'ru')
        row = self.rows()[0]
        self.assertEqual(row[:4], (body['request_id'], 'Вопрос?', 'answered', 200))
        self.assertEqual(json.loads(row[4]), body)
        RequestJournal(self.path)  # restart does not clear journal
        self.assertEqual(len(self.rows()), 1)

    def test_fallback_is_success(self):
        self.service.answer.return_value = dict(self.answer, status='fallback', basis='none',
            reason='insufficient_information', forum_answer=None, source=None, source_type=None)
        response = self.client.post('/generate-answer', json={'question':'Unknown?'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'fallback')
        self.assertEqual(self.rows()[0][2], 'fallback')

    def test_input_validation(self):
        for body in ({}, {'question':' '}, {'question':123}, {'question':'x'*2001},
                     {'question':'x','language':'de'}, {'question':'x','extra':True}):
            self.assertEqual(self.client.post('/generate-answer', json=body).status_code, 422)
        self.assertEqual(self.client.post('/generate-answer', content='{',
            headers={'Content-Type':'application/json'}).status_code, 422)
        self.service.answer.assert_not_called()
        self.assertEqual(self.rows(), [])

    def test_error_mapping_and_no_secret_leak(self):
        for error, status, code in [(GenerationError('SECRET'),502,'llm_unavailable'),
                                   (RuntimeError('postgres://SECRET'),503,'pipeline_unavailable')]:
            self.service.answer.side_effect = error
            response = self.client.post('/generate-answer', json={'question':'x'})
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.json()['error'], code)
            self.assertNotIn('SECRET', response.text)
        self.assertNotIn('SECRET', str(self.rows()))

    def test_journal_failure_before_generation(self):
        journal = Mock()
        journal.start.side_effect = OSError('disk full')
        with TestClient(create_app(self.service, journal)) as client:
            response = client.post('/generate-answer', json={'question':'x'})
        self.assertEqual(response.status_code, 503)
        self.service.answer.assert_not_called()

    def test_journal_failure_after_generation(self):
        journal = Mock()
        journal.finish.side_effect = OSError('disk full')
        with TestClient(create_app(self.service, journal)) as client:
            response = client.post('/generate-answer', json={'question':'x'})
        self.assertEqual(response.json()['error'], 'logging_unavailable')

    def test_languages(self):
        for lang in ('ru','kk','en'):
            self.service.answer.return_value = dict(self.answer, language=lang)
            response = self.client.post('/generate-answer', json={'question':'x','language':lang})
            self.assertEqual(response.json()['language'], lang)
            self.service.answer.assert_called_with('x', lang)

    def test_busy_request_and_lock_released(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        def slow(*args):
            entered.set()
            release.wait(5)
            return self.answer
        self.service.answer.side_effect = slow
        with ThreadPoolExecutor() as pool:
            first = pool.submit(self.client.post, '/generate-answer', json={'question':'first'})
            self.assertTrue(entered.wait(3))
            try:
                response = self.client.post('/generate-answer', json={'question':'second'})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()['error'], 'service_busy')
            finally:
                release.set()
            self.assertEqual(first.result().status_code, 200)
        self.service.answer.side_effect = None
        self.assertEqual(self.client.post('/generate-answer',json={'question':'third'}).status_code,200)

    def test_openapi_contract(self):
        schema = self.client.get('/openapi.json').json()
        self.assertIn('/generate-answer', schema['paths'])
        self.assertIn('forum_answer', schema['components']['schemas']['AnswerResponse']['properties'])


if __name__ == '__main__':
    unittest.main()
