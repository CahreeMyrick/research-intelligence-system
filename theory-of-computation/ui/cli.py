import argparse


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for the document intelligence system.

    Output:
        argparse.ArgumentParser:
            Configured CLI parser.
    """

    parser = argparse.ArgumentParser(
        prog="docintel",
        description="Document intelligence system",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ========================================================
    # ingest
    # ========================================================

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a PDF file or directory of PDFs",
    )

    ingest_parser.add_argument(
        "path",
        type=str,
        help="Path to a PDF file or directory containing PDFs",
    )

    # ========================================================
    # list
    # ========================================================

    subparsers.add_parser(
        "list",
        help="List stored documents",
    )

    # ========================================================
    # show
    # ========================================================

    show_parser = subparsers.add_parser(
        "show",
        help="Show information about one document",
    )

    show_parser.add_argument(
        "document_id",
        type=int,
        help="Document ID",
    )

    show_parser.add_argument(
        "--pages",
        action="store_true",
        help="Show document pages",
    )

    show_parser.add_argument(
        "--sections",
        action="store_true",
        help="Show document sections",
    )

    show_parser.add_argument(
        "--chunks",
        action="store_true",
        help="Show document chunks",
    )

    # ========================================================
    # status
    # ========================================================

    status_parser = subparsers.add_parser(
        "status",
        help="Show document processing status",
    )

    status_parser.add_argument(
        "document_id",
        type=int,
        help="Document ID",
    )

    # ========================================================
    # search
    # ========================================================

    search_parser = subparsers.add_parser(
        "search",
        help="Search document chunks",
    )

    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )

    search_parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Restrict search to one document",
    )

    # ========================================================
    # ask
    # ========================================================

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question over the document collection",
    )

    ask_parser.add_argument(
        "question",
        type=str,
        help="Question to ask",
    )

    ask_parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Restrict the question to one document",
    )

    # ========================================================
    # delete
    # ========================================================

    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a document",
    )

    delete_parser.add_argument(
        "document_id",
        type=int,
        help="Document ID",
    )

    return parser


def main() -> None:
    """
    Run the document intelligence command-line interface.

    Input:
        None

    Output:
        None
    """

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        print(f"Ingest path: {args.path}")

    elif args.command == "list":
        print("List documents")

    elif args.command == "show":
        print(f"Show document: {args.document_id}")

        if args.pages:
            print("Show pages")

        if args.sections:
            print("Show sections")

        if args.chunks:
            print("Show chunks")

    elif args.command == "status":
        print(f"Status for document: {args.document_id}")

    elif args.command == "search":
        print(f"Search query: {args.query}")
        print(f"Top K: {args.top_k}")

        if args.document_id is not None:
            print(f"Document filter: {args.document_id}")

    elif args.command == "ask":
        print(f"Question: {args.question}")

        if args.document_id is not None:
            print(f"Document filter: {args.document_id}")

    elif args.command == "delete":
        print(f"Delete document: {args.document_id}")


if __name__ == "__main__":
    main()
