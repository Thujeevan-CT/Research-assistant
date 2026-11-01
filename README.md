# Research Assistant

A small research agent. You ask one question. It searches the web, can open a page, checks that each claim is backed by retrieved text, and answers with sources.

There is a web page for everyday use and a command-line mode for the same loop.

## What you need

- Python 3.11 or newer
- A [Gemini API key](https://aistudio.google.com/apikey)
- A [Tavily API key](https://tavily.com) (web search)

## Run the app

From the `app` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set:

```text
GEMINI_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key
```

Start the web UI:

```bash
python -m research_agent --web
```

Run that command from the `app` folder, or use the helper below.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), type a question, and click **Research**. Progress shows searches and page fetches. The answer lists only claims whose supporting text was found in a retrieved page or snippet. A transcript download appears under the answer.

On macOS or Linux you can also start the UI with:

```bash
./start.sh
```

The script creates the virtual environment, installs dependencies, and opens the server. It stops and tells you what to do if `.env` is missing.

## Ask from the terminal

```bash
source .venv/bin/activate
python -m research_agent "What are the best open-source vector databases in 2026?"
```

## Settings

All of these live in `.env`. Only the two API keys are required.

| Variable | Default | What it does |
| --- | --- | --- |
| `GEMINI_API_KEY` | | Gemini calls |
| `TAVILY_API_KEY` | | Web search |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model name |
| `MAX_ROUNDS` | `4` | How many tool rounds the agent may take |
| `SAVE_TRANSCRIPTS` | `1` | Write a JSON transcript of each run |
| `TRANSCRIPT_DIR` | `runs` | Folder for transcripts, next to this app |
| `HOST` | `127.0.0.1` | Web server address |
| `PORT` | `8000` | Web server port |

`.env` and `runs/` stay on your machine. They are not part of this repository.

## How a run works

1. The model decides whether to search or answer.
2. Search results come back as titles, URLs, and snippets. The model can also fetch a page.
3. Before the answer is shown, each claim must include a URL and a short quote that actually appears in that result.
4. The app builds the source list from those verified claims.

Design notes for each build step are in `../docs/`.
