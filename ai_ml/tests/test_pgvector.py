"""Opt-in integration test. Set KB_TEST_DATABASE_URL to a dedicated test DB."""
import os
import unittest
import uuid
import numpy as np


@unittest.skipUnless(os.getenv('KB_TEST_DATABASE_URL'), 'KB_TEST_DATABASE_URL is not configured')
class PgvectorTests(unittest.TestCase):
    def test_atomic_refresh_and_dataset_isolation(self):
        from kb.store import init_database, replace_dataset, search, connect
        url=os.environ['KB_TEST_DATABASE_URL']
        first='test_'+uuid.uuid4().hex;other='test_'+uuid.uuid4().hex
        def record(id,answer='answer',language='en'):
            return dict(id=id,question='question',answer=answer,category='test',language=language)
        v=np.zeros((1,384),dtype=np.float32);v[0,0]=1
        init_database(url)
        try:
            replace_dataset(url,first,[record('old')],v,'a')
            replace_dataset(url,other,[record('keep')],v,'a')
            replace_dataset(url,first,[record('new','updated')],v,'b')
            replace_dataset(url,first,[record('new','updated')],v,'b')
            result=search(url,first,v[0])
            self.assertEqual([r['id'] for r in result],['new'])
            self.assertEqual(result[0]['answer'],'updated')
            self.assertAlmostEqual(result[0]['score'],1)
            self.assertEqual(search(url,other,v[0])[0]['id'],'keep')
            # The delete and manifest update must roll back if insert fails.
            with self.assertRaises(Exception):
                replace_dataset(url,first,[record('bad',language='invalid')],v,'c')
            self.assertEqual(search(url,first,v[0])[0]['id'],'new')
            with connect(url) as conn:
                meta=conn.execute('SELECT source_sha256 FROM ai_kb_datasets WHERE dataset=%s',(first,)).fetchone()
            self.assertEqual(meta['source_sha256'],'b')
        finally:
            with connect(url) as conn:
                conn.execute('DELETE FROM ai_kb_entries WHERE dataset = ANY(%s)',([first,other],))
                conn.execute('DELETE FROM ai_kb_datasets WHERE dataset = ANY(%s)',([first,other],))
