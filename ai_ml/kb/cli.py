import argparse
import hashlib
import json
import os
from pathlib import Path

from .dataset import prepare_records

ROOT = Path(__file__).resolve().parents[1]


def load_source(path):
    # Detect a CSV changed while it was being read.
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    records, excluded, report = prepare_records(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError("CSV changed during reading. Retry with a stable file.")
    report['source_sha256'] = before
    return records, excluded, report


def write_prepared(records, excluded, report):
    for directory, filename, rows in [
        ('processed','qa_clean.jsonl',records), ('rejected','faq_excluded.jsonl',excluded)]:
        folder = ROOT / 'data' / directory
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (ROOT/'data/processed/index_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='SDU FAQ indexing: CSV -> E5 -> pgvector')
    parser.add_argument('command',choices=['prepare','init-db','index','search','retrieve','generate'])
    parser.add_argument('--input',type=Path,default=ROOT/'data/raw/faq.csv')
    parser.add_argument('--dataset',default='faq')
    parser.add_argument('--local-only',action='store_true',help='Build embeddings without PostgreSQL')
    parser.add_argument('--question')
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--threshold',type=float,help='retrieve/generate; default KB_MIN_SIMILARITY or 0.80')
    parser.add_argument('--language',choices=['ru','kk','en'],default='ru')
    args=parser.parse_args()
    if args.local_only and args.command!='index':parser.error('--local-only is for index only')
    if args.threshold is not None and args.command not in {'retrieve','generate'}:parser.error('--threshold is for retrieve/generate only')
    if args.command in {'prepare','index'}:
        records, excluded, report=load_source(args.input)
        write_prepared(records,excluded,report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if args.command=='prepare':return
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    url=os.getenv('KB_DATABASE_URL','')
    if args.command=='init-db':
        if not url:parser.error('Set KB_DATABASE_URL in ai_ml/.env')
        from .store import init_database
        init_database(url)
        print('Knowledge-base tables are ready.')
        return
    if not args.local_only and not url:parser.error('Set KB_DATABASE_URL in ai_ml/.env')
    if args.command in {'search','retrieve','generate'} and (not args.question or not args.question.strip()):parser.error('--question is required')
    if args.command in {'retrieve','generate'}:
        from .retrieval import validate_threshold
        try:
            threshold=validate_threshold(args.threshold if args.threshold is not None else os.getenv('KB_MIN_SIMILARITY','0.80'))
        except ValueError as exc:parser.error(str(exc))
    if args.command=='generate':
        from .generation import ChatClient
        try:client=ChatClient.from_env()
        except ValueError as exc:parser.error(str(exc))
        context_path=Path(os.getenv('KB_CONTEXT_FILE','config/university_context.txt'))
        if not context_path.is_absolute():context_path=ROOT/context_path
        context=context_path.read_text(encoding='utf-8')
    if not 1 <= args.top_k <= 20:parser.error('--top-k must be between 1 and 20')
    from .embedding import E5, MODEL, STRATEGY, DIMENSION
    model=E5(os.getenv('KB_DEVICE','cpu'),int(os.getenv('KB_BATCH_SIZE','16')))
    if args.command=='index':
        import numpy as np
        vectors=model.encode([r['question'] for r in records])
        folder=ROOT/'data/index';folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'embeddings.npy',vectors,allow_pickle=False)
        (folder/'records.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
        (folder/'metadata.json').write_text(json.dumps({**report,'model':MODEL,'strategy':STRATEGY,'dimension':DIMENSION},ensure_ascii=False,indent=2),encoding='utf-8')
        if not args.local_only:
            from .store import replace_dataset
            replace_dataset(url,args.dataset,records,vectors,report['source_sha256'])
        print(f'Created {len(records)} question embeddings of dimension {DIMENSION}. '+('Local files only.' if args.local_only else 'Committed to pgvector.'))
    elif args.command in {'retrieve','generate'}:
        from .retrieval import postgres_retriever
        retriever=postgres_retriever(url,model,threshold,args.dataset)
        if args.command=='generate':
            from .generation import AnswerService
            result=AnswerService(retriever,client,context).answer(args.question,args.language)
        else:result=retriever.retrieve(args.question)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        from .store import search
        result=search(url,args.dataset,model.encode([args.question],query=True)[0],args.top_k)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
