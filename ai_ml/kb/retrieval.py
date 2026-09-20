"""Top-1 retrieval with an explicit cosine-similarity threshold."""
import math


def validate_threshold(value):
    value = float(value)
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('Similarity threshold must be finite and between -1 and 1.')
    return value


class Retriever:
    """Construct once and reuse the encoder between student requests.

    nearest must accept (vector, top_k) and return cosine-ranked record dicts.
    Storage errors propagate: unavailable DB is not the same as no match.
    """
    def __init__(self, encoder, nearest, threshold=0.80, dataset='faq', selector=None, candidate_count=1):
        self.encoder = encoder
        self.nearest = nearest
        self.threshold = validate_threshold(threshold)
        self.dataset = dataset
        self.selector = selector
        self.candidate_count = candidate_count

    def retrieve(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question must be a nonempty string.')
        question = question.strip()
        if len(question) > 2000:
            raise ValueError('Question must not exceed 2000 characters.')
        vector = self.encoder.encode([question], query=True)[0]
        candidates = self.nearest(vector, top_k=self.candidate_count)
        result = {
            'status': 'not_found', 'matched': False, 'dataset': self.dataset,
            'threshold': self.threshold, 'score': None, 'answer': None,
            'source': None,
        }
        if not candidates:
            return result
        best = self.selector(question, candidates) if self.selector else candidates[0]
        if best is None:
            return result
        score = float(best['score'])
        if not math.isfinite(score) or score < -1.00001 or score > 1.00001:
            raise ValueError('Storage returned an invalid cosine similarity.')
        score = min(1.0, max(-1.0, score))
        result['score'] = score
        # Compare full precision, never a rounded display score.
        if score < self.threshold:
            from .ranking import confirmed_definition
            if not (self.selector and score >= self.threshold - 0.02 and confirmed_definition(question, best.get('question', ''))):
                return result
        if not isinstance(best.get('answer'), str) or not best['answer'].strip():
            raise ValueError('Matched index record has no answer. Rebuild the index.')
        result.update(
            status='matched', matched=True, answer=best['answer'],
            source={k: best.get(k) for k in ('id','question','category','language','source_row')},
        )
        return result


def postgres_retriever(url, encoder, threshold=0.80, dataset='faq'):
    from .store import search
    from .ranking import select
    return Retriever(encoder,
        lambda vector, top_k: search(url, dataset, vector, top_k),
        threshold=threshold, dataset=dataset, selector=select, candidate_count=1000)
