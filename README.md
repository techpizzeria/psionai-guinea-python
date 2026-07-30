# summarizer

A small CLI that summarizes a text file with the OpenAI Chat API and can embed
the summary for downstream similarity search.

Usage: `python -m summarizer path/to/file.txt`

Requires the `OPENAI_API_KEY` environment variable (a `.env` file is also loaded).
