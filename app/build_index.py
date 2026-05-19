"""
build_index.py
--------------
Reads catalog.json and builds a FAISS vector index.

Run from project root:
    python app/build_index.py
"""

import json
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FIX: was defining BASE_DIR/CATALOG_PATH/INDEX_DIR with pathlib at top,
# then immediately OVERWRITING them with plain strings below — causing
# load_catalog() to open CATALOG_PATH (pathlib) while receiving CATALOG_FILE
# (string) as its argument. Now everything uses pathlib consistently.

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "catalog.json"
INDEX_DIR = BASE_DIR / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_catalog() -> list[dict]:
    """Load the scraped catalog JSON."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"{CATALOG_PATH} not found. Run scraper.py first."
        )
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} assessments from {CATALOG_PATH}")
    return data


def build_document(item: dict) -> Document:
    """Convert one catalog item into a LangChain Document for embedding."""
    test_types_str = ", ".join(item.get("test_types", [])) or "Unknown"
    remote = "Yes" if item.get("remote_testing") else "No"
    adaptive = "Yes" if item.get("adaptive_irt") else "No"
    description = item.get("description", "") or ""
    job_levels = item.get("job_levels", "") or ""
    languages = item.get("languages", "") or ""

    page_content = f"""
Assessment Name: {item['name']}
Description: {description}
Test Types: {test_types_str}
Remote Testing Supported: {remote}
Adaptive/IRT: {adaptive}
Job Levels: {job_levels}
Languages: {languages}
URL: {item['url']}
""".strip()

    metadata = {
        "name": item["name"],
        "url": item["url"],
        "test_types": test_types_str,
        "remote_testing": item.get("remote_testing", False),
        "adaptive_irt": item.get("adaptive_irt", False),
        "description": description[:500],
        "job_levels": job_levels,
    }

    return Document(page_content=page_content, metadata=metadata)


def main():
    logger.info("=== Building FAISS Index ===")

    catalog = load_catalog()
    docs = [build_document(item) for item in catalog]
    logger.info(f"Created {len(docs)} documents")

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(str(INDEX_DIR))
    logger.info(f"✅ Index saved to {INDEX_DIR}")
    logger.info(f"   Total vectors: {vectorstore.index.ntotal}")

    logger.info("\nSmoke test — searching for 'Java developer':")
    results = vectorstore.similarity_search("Java developer skills test", k=3)
    for r in results:
        logger.info(f"  - {r.metadata['name']} | {r.metadata['url']}")


if __name__ == "__main__":
    main()