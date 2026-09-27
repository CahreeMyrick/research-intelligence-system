"""Run the connected Clarivue UI and research backend together."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer


DEFAULT_GRAPH_PATH = Path(__file__).resolve().parents[0] / 'knowledge-infusion' / 'demo_graph.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--backend-port', type=int, default=8012)
    parser.add_argument(
        '--knowledge-graph',
        type=str,
        default=str(DEFAULT_GRAPH_PATH) if DEFAULT_GRAPH_PATH.exists() else '',
        help='Explicit graph JSON (defaults to indexer/knowledge-infusion/demo_graph.json; pass "" or "none" to disable).',
    )
    parser.add_argument('--config', type=Path, help='Defaults to ingestion/clinivue_config.yaml (medical database and vectors).')
    parser.add_argument('--web-root', type=Path, default=Path(__file__).resolve().parents[2] / 'clarivue_qa')
    args = parser.parse_args()
    if args.config:
        os.environ['RIS_CONFIG'] = str(args.config.resolve())
    if args.knowledge_graph and args.knowledge_graph.lower() not in {'', 'none', 'false', '0'}:
        os.environ['RIS_KNOWLEDGE_GRAPH'] = str(Path(args.knowledge_graph).resolve())
    else:
        os.environ.pop('RIS_KNOWLEDGE_GRAPH', None)
    from indexer import http_api
    spec = importlib.util.spec_from_file_location('clarivue_web', args.web_root / 'api_server.py')
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    proxy.RESEARCH_RAG_URL = f'http://127.0.0.1:{args.backend_port}/api/qa'
    # Bind both before starting either, so port conflicts fail cleanly.
    with ThreadingHTTPServer(('127.0.0.1', args.backend_port), http_api.Handler) as backend:
        with ThreadingHTTPServer(('127.0.0.1', args.port), proxy.Handler) as web:
            worker = threading.Thread(target=backend.serve_forever, daemon=True)
            worker.start()
            print(f'Clarivue: http://127.0.0.1:{args.port}', flush=True)
            print(f'Research backend: {proxy.RESEARCH_RAG_URL}', flush=True)
            print(f'Configuration: {http_api.CONFIG_PATH}', flush=True)
            print(f'Graph: {os.environ.get("RIS_KNOWLEDGE_GRAPH", "not configured")}', flush=True)
            try:
                web.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                backend.shutdown()
                worker.join()


if __name__ == '__main__':
    main()
