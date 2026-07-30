"""Command line entry point for the summarizer.

Reads a text file, asks the OpenAI Chat API for a short summary and prints it.
With ``--embed`` it also prints the embedding vector length of the summary.
"""

import argparse
import os
import sys

import openai
from dotenv import load_dotenv

from summarizer.embeddings import embed_text

SYSTEM_PROMPT = "You are a concise assistant that summarizes documents in a few sentences."


def summarize(text: str, model: str = "gpt-3.5-turbo") -> str:
    """Return a short summary of ``text`` using the Chat Completion API."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    return response["choices"][0]["message"]["content"].strip()


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="summarizer", description=__doc__)
    parser.add_argument("path", help="Path to the text file to summarize")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Chat model to use")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also embed the summary and print the vector length",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]

    try:
        with open(args.path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        print(f"error: cannot read {args.path}: {exc}", file=sys.stderr)
        return 1

    summary = summarize(text, model=args.model)
    print(summary)

    if args.embed:
        vector = embed_text(summary)
        print(f"embedding dimensions: {len(vector)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
