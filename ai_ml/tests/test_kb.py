import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from kb.dataset import prepare_records
from kb.embedding import E5, validate_vectors


class DatasetTests(unittest.TestCase):
    def make_csv(self, rows):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name)/'faq.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['id','question','answer','language','status','category_original'])
            writer.writerows(rows)
        return path

    def test_short_answers_preserved_and_ambiguous_excluded(self):
        path=self.make_csv([
            ['1','Можно?','Да','ru','needs_review','portal'],
            ['2','Бола ма?','Жоқ','kk','content_review','portal'],
            ['3','Where?','','en','missing_answer',''],
            ['4','Which?','Shared answer','en','alignment_review',''],
        ])
        rows,rejected,report=prepare_records(path)
        self.assertEqual([r['answer'] for r in rows],['Да','Жоқ'])
        self.assertEqual(rows[0]['category'],'portal')
        self.assertEqual(report['excluded'],2)
        self.assertEqual(rejected[1]['exclusion_reason'],'ambiguous_alignment')

    def test_different_answers_to_same_question_not_lost(self):
        rows,_,_=prepare_records(self.make_csv([
            ['1','Можно?','Да','ru','verified',''],
            ['2','Можно?','Нет','ru','verified','']]))
        self.assertEqual(len(rows),2)

    def test_duplicate_id_fails(self):
        path=self.make_csv([['1','Where?','Here','en','verified','']]*2)
        with self.assertRaises(ValueError):prepare_records(path)

    def test_empty_dataset_fails(self):
        path=self.make_csv([['1','Where?','','en','missing_answer','']])
        with self.assertRaises(ValueError):prepare_records(path)

    def test_actual_dataset_partition(self):
        rows,rejected,report=prepare_records(Path(__file__).resolve().parents[1]/'data/raw/faq.csv')
        self.assertEqual(len(rows)+len(rejected),report['total'])
        self.assertEqual(len({r['id'] for r in rows+rejected}),report['total'])
        self.assertTrue(all(r['question'] and r['answer'] for r in rows))


class EmbeddingTests(unittest.TestCase):
    def encoder(self):
        encoder=E5.__new__(E5)
        encoder.batch_size=2
        encoder.model=Mock()
        encoder.model.max_seq_length=512
        encoder.model.tokenizer.return_value={'input_ids':[[1,2]]}
        encoder.model.encode.return_value=np.ones((1,384),dtype=np.float32)
        return encoder

    def test_question_and_query_prefixes(self):
        e=self.encoder()
        for text in ['Справка?','Анықтама?','Certificate?']:
            e.encode([text]);self.assertEqual(e.model.encode.call_args.args[0],['passage: '+text])
            v=e.encode([text],query=True)
            self.assertEqual(e.model.encode.call_args.args[0],['query: '+text])
            self.assertAlmostEqual(float(np.linalg.norm(v[0])),1,places=5)

    def test_long_input_not_silently_truncated(self):
        e=self.encoder();e.model.tokenizer.return_value={'input_ids':[[1]*513]}
        with self.assertRaises(ValueError):e.encode(['long'])
        e.model.encode.assert_not_called()

    def test_invalid_vectors(self):
        for v in [np.zeros((1,384)),np.full((1,384),np.nan),np.ones((1,768))]:
            with self.assertRaises(ValueError):validate_vectors(v,1)


if __name__=='__main__':unittest.main()
