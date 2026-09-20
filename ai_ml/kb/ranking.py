"""Conservative lexical/intent checks supplement multilingual cosine similarity."""
import re

ALIASES = {
 'advisor': ('advisor', 'adviser', 'эдвайзер', 'куратор'),
 'certificate': ('certificate', 'справк', 'анықтама'),
 'portal': ('portal', 'портал'),
 'moodle': ('moodle', 'мудл'),
}
STOP = set('what is a an the how to can i do get in at of for my we you it деген не кім қалай где как что кто это можно сду sdu university университет об в на по получить'.split())

def terms(text):
    tokens = re.findall(r'[^\W_]+', text.casefold())
    result = set()
    for token in tokens:
        if token in STOP:
            continue
        canonical = next((key for key, variants in ALIASES.items() if any(token.startswith(v) for v in variants)), token)
        result.add(canonical)
    return result

def definition(text):
    return bool(re.search(r'деген\s+(не|кім)|\b(what|who)\s+(is|are)\b|что такое|кто такой|кім\s+және', text.casefold()))

def select(question, candidates):
    query_terms = terms(question)
    acronyms = set(re.findall(r'\b[A-Z][A-Z0-9]{1,7}\b', question)) - {'SDU', 'I'}
    ranked = []
    for candidate in candidates:
        text = candidate.get('question', '')
        candidate_terms = terms(text)
        if acronyms and not all(a.casefold() in candidate_terms for a in acronyms):
            continue
        overlap = len(query_terms & candidate_terms) / max(1, len(query_terms))
        # Broad definition requests must not select a troubleshooting/retake question.
        if definition(question) and not definition(text):
            continue
        if query_terms and not overlap:
            continue
        exact = question.strip().casefold().rstrip('?.!') == text.strip().casefold().rstrip('?.!')
        ranked.append((int(exact), overlap, float(candidate['score']), candidate))
    ranked.sort(key=lambda x: x[:3], reverse=True)
    return ranked[0][3] if ranked else None


def confirmed_definition(question, candidate):
    """Narrow tolerance for definitions sharing an explicitly mapped concept."""
    known = set(ALIASES)
    return (definition(question) and definition(candidate)
            and bool(terms(question) & terms(candidate) & known))
