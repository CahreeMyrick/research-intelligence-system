"""Export the existing illustrative NetworkX graph to the QA JSON format."""
import argparse
import json
from pathlib import Path

from .knowledge_graph import G


def export_demo_graph(output: Path) -> None:
    payload = {
        "metadata": {
            "name": "Chest X-ray demonstration graph",
            "status": "demo_unverified",
            "origin": "indexer/knowledge-infusion/knowledge_graph.py",
            "description": "Hand-authored illustrative associations and synthetic image observations; not a curated clinical knowledge base.",
        },
        "nodes": [{"id": node, **properties} for node, properties in G.nodes(data=True)],
        "edges": [
            {**properties, "source": source, "target": target, "quality": 0}
            for source, target, properties in G.edges(data=True)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("demo_graph.json"))
    args = parser.parse_args()
    export_demo_graph(args.output)
    print(f"Exported demo graph to {args.output}")


if __name__ == "__main__":
    main()
