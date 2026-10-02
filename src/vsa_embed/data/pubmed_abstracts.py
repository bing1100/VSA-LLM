"""PubMed baseline XML (gzipped) → abstracts with their chemical lists, for the T4 chemistry corpus.

A small read-only streaming parser over NLM baseline files (WP-T1 has its own extractor for T1-open);
files are verified against NCBI's `.md5` companions before use.
"""

from __future__ import annotations

import gzip
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator


def iter_abstracts(path: Path) -> Iterator[dict[str, Any]]:
    """Yield `{pmid, title, abstract, chemicals, language}` for citations that have an abstract."""
    with gzip.open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag != "PubmedArticle":
                continue
            citation = element.find("MedlineCitation")
            if citation is not None:
                parts = ["".join(node.itertext()).strip() for node in citation.findall("Article/Abstract/AbstractText")]
                abstract = " ".join(p for p in parts if p)
                if abstract:
                    yield {
                        "pmid": citation.findtext("PMID"),
                        "title": "".join(citation.find("Article/ArticleTitle").itertext()).strip()
                        if citation.find("Article/ArticleTitle") is not None else "",
                        "abstract": abstract,
                        "chemicals": [c.findtext("NameOfSubstance") for c in citation.findall("ChemicalList/Chemical")],
                        "language": citation.findtext("Article/Language"),
                    }
            element.clear()


def verify_md5(path: Path) -> bool:
    """Check a baseline file against its NCBI `.md5` companion (`MD5(name)= hex`); False if absent."""
    companion = Path(str(path) + ".md5")
    if not companion.exists():
        return False
    expected = companion.read_text().strip().split("=")[-1].strip()
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected
