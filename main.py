"""Search engine API: the inverted-index search engine served over HTTP.

The index lives in SQLite (search.db), built from the corpus by
build_db.py. Search semantics match the command-line search engine
exactly: a multi-term query returns only documents containing every term
(AND), a query in double quotes is a phrase search, and results rank by
total term frequency divided by the square root of the document length.

Endpoints:
    GET /health     -> {"status": "ok"}
    GET /search?q=  -> {"query": ..., "results": [...]}
    GET /documents -> [{"filename": ..., "title": ...}, ...]

The database path can be overridden with the SEARCH_DB environment
variable. The default is search.db next to this file.
"""

import json
import math
import os
import re
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException

DB_PATH = os.environ.get("SEARCH_DB", str(Path(__file__).parent / "search.db"))

app = FastAPI(title="Search Engine API")


def tokenize(text):
    """Split text into lowercase words.

    Anything that is not a letter or a digit breaks a word, so punctuation
    and case do not matter: "Alice," and "alice" become the same token.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _phrase_occurrences(position_sets):
    """Count occurrences of a phrase in one document.

    position_sets is a list of sets of token positions, one set per phrase
    word. A phrase occurs at position p when the first word is at p and
    each following word is at p + 1, p + 2, and so on.
    """
    count = 0
    for p in position_sets[0]:
        if all(p + i in position_sets[i] for i in range(1, len(position_sets))):
            count += 1
    return count


def find_snippet(text, terms, width=45):
    """Return a short excerpt of text around the first occurrence of any
    of the given terms. Returns "" when no term occurs."""
    lowered = text.lower()
    first = None
    for term in terms:
        at = lowered.find(term)
        if at != -1 and (first is None or at < first):
            first = at
    if first is None:
        return ""
    start = max(0, first - width)
    end = min(len(text), first + width)
    snippet = " ".join(text[start:end].split())
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def get_connection():
    """Open a read-only connection to the index database.

    Raises HTTPException 503 when the database has not been built yet
    (run `python build_db.py` first).
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            status_code=503,
            detail="Search index not built. Run 'python build_db.py' first.",
        )
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _load_postings(connection, terms):
    """Fetch postings for each term.

    Returns a list of {doc_id: [positions]} dicts, one per term, in the
    same order as terms. A missing term yields an empty dict.
    """
    postings = []
    for term in terms:
        rows = connection.execute(
            "SELECT doc_id, positions FROM postings WHERE term = ?", (term,)
        ).fetchall()
        postings.append({row["doc_id"]: json.loads(row["positions"])
                         for row in rows})
    return postings


def _and_search(connection, postings):
    """Documents containing every term, ranked by total term frequency
    divided by the square root of the document length.

    Returns a list of (doc_id, score) pairs, best score first.
    """
    if any(not posting for posting in postings):
        return []
    common = set(postings[0])
    for posting in postings[1:]:
        common &= set(posting)
    if not common:
        return []
    lengths = {
        row["id"]: row["length"]
        for row in connection.execute(
            "SELECT id, length FROM documents WHERE id IN (%s)"
            % ",".join("?" * len(common)),
            tuple(common),
        ).fetchall()
    }
    results = []
    for doc_id in common:
        total = sum(len(postings[i][doc_id]) for i in range(len(postings)))
        length = lengths[doc_id]
        score = total / math.sqrt(length) if length else 0.0
        results.append((doc_id, score))
    return sorted(results, key=lambda item: (-item[1], item[0]))


def _phrase_search(connection, postings):
    """Documents containing the exact phrase, ranked by phrase frequency
    divided by the square root of the document length.

    Returns a list of (doc_id, score) pairs, best score first.
    """
    if any(not posting for posting in postings):
        return []
    common = set(postings[0])
    for posting in postings[1:]:
        common &= set(posting)
    if not common:
        return []
    lengths = {
        row["id"]: row["length"]
        for row in connection.execute(
            "SELECT id, length FROM documents WHERE id IN (%s)"
            % ",".join("?" * len(common)),
            tuple(common),
        ).fetchall()
    }
    results = []
    for doc_id in common:
        sets = [set(postings[i][doc_id]) for i in range(len(postings))]
        count = _phrase_occurrences(sets)
        if count:
            length = lengths[doc_id]
            score = count / math.sqrt(length) if length else 0.0
            results.append((doc_id, score))
    return sorted(results, key=lambda item: (-item[1], item[0]))


def search(connection, query):
    """Search the index. Returns a list of result dicts, best score first.

    A query in double quotes is a phrase search. Otherwise every query
    word must appear in a document (AND). An empty query, or a word that
    appears nowhere, returns no results.
    """
    query = query.strip()
    if not query:
        return []
    if query.startswith('"') and query.endswith('"') and len(query) >= 2:
        terms = tokenize(query[1:-1])
        snippet_terms = [query[1:-1]]
        search_fn = _phrase_search
    else:
        terms = tokenize(query)
        snippet_terms = terms
        search_fn = _and_search
    if not terms:
        return []
    postings = _load_postings(connection, terms)
    ranked = search_fn(connection, postings)
    if not ranked:
        return []
    docs = {
        row["id"]: row
        for row in connection.execute(
            "SELECT id, filename, title, text FROM documents WHERE id IN (%s)"
            % ",".join("?" * len(ranked)),
            tuple(doc_id for doc_id, _ in ranked),
        ).fetchall()
    }
    results = []
    for doc_id, score in ranked:
        doc = docs[doc_id]
        results.append({
            "filename": doc["filename"],
            "title": doc["title"],
            "score": round(score, 4),
            "snippet": find_snippet(doc["text"], snippet_terms),
        })
    return results


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search_endpoint(q: str = ""):
    """Ranked search results for q.

    An empty query returns an empty result list, the same graceful
    behaviour as the command-line search engine.
    """
    with get_connection() as connection:
        return {"query": q, "results": search(connection, q)}


@app.get("/documents")
def list_documents():
    """Every indexed document, with file name and title."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT filename, title FROM documents ORDER BY id"
        ).fetchall()
        return [{"filename": row["filename"], "title": row["title"]}
                for row in rows]
