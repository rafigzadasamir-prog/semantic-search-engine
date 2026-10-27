import json

from semantic_search_engine.documents import Document, load_corpus


def test_load_corpus_returns_bundled_documents_with_unique_ids():
    documents = load_corpus()
    assert len(documents) == 120
    assert all(isinstance(doc, Document) for doc in documents)
    ids = [doc.id for doc in documents]
    assert len(ids) == len(set(ids))


def test_document_content_combines_title_and_text():
    doc = Document(id="x", topic="t", title="Title", text="Body text.")
    assert doc.content == "Title. Body text."


def test_load_corpus_from_explicit_path(tmp_path):
    custom = [{"id": "z1", "topic": "demo", "title": "Demo", "text": "Demo text."}]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(custom), encoding="utf-8")

    documents = load_corpus(path)

    assert len(documents) == 1
    assert documents[0].id == "z1"
    assert documents[0].title == "Demo"
