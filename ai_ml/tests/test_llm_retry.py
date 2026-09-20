import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from kb.generation import ChatClient, GenerationError

class RetryTests(unittest.TestCase):
    def test_transient_then_success(self):
        c=ChatClient('gemini','model','key')
        with patch.object(c,'_complete_once',side_effect=[GenerationError('busy','llm_overloaded'),'{}']) as call, patch('kb.generation.time.sleep') as sleep:
            self.assertEqual(c.complete([]),'{}')
            self.assertEqual(call.call_count,2)
            sleep.assert_called_once_with(1)

    def test_overload_is_bounded(self):
        c=ChatClient('gemini','model','key')
        with patch('kb.generation.request.urlopen',side_effect=HTTPError('url',503,'busy',{},None)) as call, patch('kb.generation.time.sleep'):
            with self.assertRaises(GenerationError) as exc:c.complete([])
            self.assertEqual(exc.exception.code,'llm_overloaded')
            self.assertEqual(call.call_count,3)

    def test_auth_not_retried(self):
        c=ChatClient('gemini','model','key')
        with patch('kb.generation.request.urlopen',side_effect=HTTPError('url',401,'bad key',{},None)) as call:
            with self.assertRaises(GenerationError):c.complete([])
            self.assertEqual(call.call_count,1)

    def test_daily_quota_not_retried(self):
        import io, json
        payload=[{'error': {'details': [{'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}]}}]
        err=HTTPError('url',429,'quota',{},io.BytesIO(json.dumps(payload).encode()))
        c=ChatClient('gemini','model','key')
        with patch('kb.generation.request.urlopen',side_effect=err) as call:
            with self.assertRaises(GenerationError) as caught:c.complete([])
            self.assertEqual(caught.exception.code,'llm_daily_quota_exceeded')
            self.assertEqual(call.call_count,1)
