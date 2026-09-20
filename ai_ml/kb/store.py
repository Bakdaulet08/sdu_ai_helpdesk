"""PostgreSQL storage. A refresh replaces only the requested dataset atomically."""
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from .embedding import MODEL, DIMENSION, STRATEGY, validate_vectors

DDL = """
CREATE TABLE IF NOT EXISTS ai_kb_datasets (
    dataset text PRIMARY KEY,
    embedding_model text NOT NULL,
    embedding_strategy text NOT NULL,
    dimension integer NOT NULL,
    source_sha256 text NOT NULL,
    record_count integer NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ai_kb_entries (
    dataset text NOT NULL REFERENCES ai_kb_datasets(dataset),
    id text NOT NULL,
    question text NOT NULL,
    answer text NOT NULL,
    category text NOT NULL,
    language text NOT NULL CHECK (language IN ('ru','kk','en')),
    source_row text NOT NULL DEFAULT '',
    embedding vector(384) NOT NULL,
    PRIMARY KEY (dataset, id)
);
"""


def connect(url):
    return psycopg.connect(url, connect_timeout=10, row_factory=dict_row)


def init_database(url):
    with connect(url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(DDL)


def replace_dataset(url, dataset, records, vectors, source_sha256):
    if not dataset.strip() or not records:
        raise ValueError("Refusing to replace a dataset with empty data.")
    if len({r['id'] for r in records}) != len(records):
        raise ValueError("Duplicate record IDs.")
    vectors = validate_vectors(vectors, len(records))
    with connect(url) as conn:
        register_vector(conn)
        # Serialize refreshes for the same dataset. Readers see old or new snapshot.
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", ("ai_kb:" + dataset,))
        conn.execute("""INSERT INTO ai_kb_datasets
            (dataset, embedding_model, embedding_strategy, dimension, source_sha256, record_count)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (dataset) DO UPDATE SET
              embedding_model=EXCLUDED.embedding_model, embedding_strategy=EXCLUDED.embedding_strategy,
              dimension=EXCLUDED.dimension, source_sha256=EXCLUDED.source_sha256,
              record_count=EXCLUDED.record_count, updated_at=now()""",
            (dataset, MODEL, STRATEGY, DIMENSION, source_sha256, len(records)))
        conn.execute("DELETE FROM ai_kb_entries WHERE dataset = %s", (dataset,))
        with conn.cursor() as cursor:
            cursor.executemany("""INSERT INTO ai_kb_entries
                (dataset,id,question,answer,category,language,source_row,embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""", [
                (dataset,r['id'],r['question'],r['answer'],r['category'],r['language'],
                 str(r.get('source_row') or ''), v) for r,v in zip(records,vectors)])


def search(url, dataset, vector, top_k=3):
    if not 1 <= top_k <= 2000:
        raise ValueError("top_k must be between 1 and 2000.")
    vector = validate_vectors(np.asarray(vector).reshape(1, -1), 1)[0]
    with connect(url) as conn:
        # Keep metadata and results consistent during a concurrent refresh.
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        register_vector(conn)
        meta = conn.execute("SELECT * FROM ai_kb_datasets WHERE dataset=%s", (dataset,)).fetchone()
        if not meta:
            raise ValueError("Dataset not indexed. Run the index command first.")
        if (meta['embedding_model'],meta['dimension'],meta['embedding_strategy']) != (MODEL,DIMENSION,STRATEGY):
            raise ValueError("Model or embedding strategy mismatch; reindex this dataset.")
        # Exact cosine search is sufficient for this small FAQ set. No ANN recall loss.
        return conn.execute("""SELECT id,question,answer,category,language,source_row,
            1 - (embedding <=> %s) AS score
            FROM ai_kb_entries WHERE dataset=%s
            ORDER BY embedding <=> %s, id LIMIT %s""",
            (vector,dataset,vector,top_k)).fetchall()
