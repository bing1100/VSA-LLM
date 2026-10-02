"""PubMed annual baseline: download (md5-verified), parse title + abstract, extract to parquet (T1-open).

NLM terms (baseline README): NLM must be acknowledged as the source; some abstracts may be protected
by copyright; NLM endorses nothing built from the data. Extracted text stays under the data root and
is never committed.

One parquet file per baseline file, columns `pmid` (int64), `year` (int16, 0 if unknown), `text`
(title, a blank line, then the abstract sections as `Label: text` lines, all-caps labels in sentence case), `mesh` (descriptor UIs
indexed for the citation; empty for records not yet indexed). Records without an English abstract
of at least `min_abstract_words` words are skipped and counted.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
_MD5 = re.compile(r"=\s*([0-9a-f]{32})")
_BLOCK_TAGS = {"b"}   # inline <b>IMPORTANCE</b> headings inside AbstractText start a new line
EXTRACTOR_VERSION = 2   # bump when the text format changes; finished files of another version are redone


def baseline_names(prefix: str, first: int, last: int) -> list[str]:
    """File names `prefix` + 4-digit number for `first..last` inclusive, in the given direction
    (`first > last` lists newest first)."""
    step = 1 if last >= first else -1
    return [f"{prefix}{n:04d}.xml.gz" for n in range(first, last + step, step)]


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_md5_sidecar(text: str) -> str:
    """`MD5(pubmed26n0001.xml.gz)= <hex>` → hex."""
    match = _MD5.search(text)
    if not match:
        raise ValueError(f"unrecognised md5 sidecar: {text!r}")
    return match.group(1)


def download_baseline(names: list[str], dest: Path, *, base_url: str = BASELINE_URL, retries: int = 3,
                      timeout: int = 600) -> list[dict[str, Any]]:
    """Download each file and its `.md5` sidecar; keep a file only if its md5 matches. Files already
    present with a matching md5 are not fetched again. Returns one record per file."""
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        target, sidecar = dest / name, dest / f"{name}.md5"
        expected = None
        for attempt in range(retries):
            try:
                if not sidecar.exists():
                    response = requests.get(base_url + name + ".md5", timeout=60)
                    response.raise_for_status()
                    sidecar.write_text(response.text)
                expected = parse_md5_sidecar(sidecar.read_text())
                if target.exists() and file_digest(target, "md5") == expected:
                    break
                partial = target.with_suffix(target.suffix + ".part")
                with requests.get(base_url + name, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    with open(partial, "wb") as handle:
                        for chunk in response.iter_content(1 << 20):
                            handle.write(chunk)
                if file_digest(partial, "md5") != expected:
                    partial.unlink(missing_ok=True)
                    raise ValueError(f"{name}: md5 mismatch")
                partial.replace(target)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))
        records.append({"name": name, "bytes": target.stat().st_size, "md5": expected,
                        "sha256": file_digest(target)})
    return records


def section_label(label: str) -> str:
    """All-caps section labels in sentence case ("MATERIALS AND METHODS" → "Materials and methods"):
    the linker is case-insensitive and GPT-2 splits capitals into several subtokens, so "METHODS"
    would otherwise be a ≥ 2-subtoken link to the descriptor Methods in every structured abstract."""
    label = " ".join(label.split())
    return label.capitalize() if label.isupper() else label


def _text_of(element: ET.Element) -> str:
    """Element text with inline markup flattened (<i>, <sup>, <sub> joined; <b> starts a line)."""
    parts: list[str] = [element.text or ""]
    for child in element:
        inner = _text_of(child)
        if child.tag in _BLOCK_TAGS:
            parts.append(f"\n{section_label(inner.strip())}: ")
        else:
            parts.append(inner)
        parts.append(child.tail or "")
    return "".join(parts)


def _clean(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _year(article: ET.Element) -> int:
    for path in ("Journal/JournalIssue/PubDate/Year", "ArticleDate/Year"):
        value = article.findtext(path)
        if value and value.strip().isdigit():
            return int(value)
    medline = article.findtext("Journal/JournalIssue/PubDate/MedlineDate") or ""
    match = re.search(r"(1[89]|20)\d\d", medline)
    return int(match.group(0)) if match else 0


def parse_citation(element: ET.Element) -> dict[str, Any] | None:
    """One `PubmedArticle` → record, or None if it has no PMID/title/abstract."""
    citation = element.find("MedlineCitation")
    if citation is None:
        return None
    article = citation.find("Article")
    pmid = citation.findtext("PMID")
    if article is None or not pmid:
        return None
    title_element = article.find("ArticleTitle")
    title = _clean(_text_of(title_element)) if title_element is not None else ""
    sections = []
    for section in article.findall("Abstract/AbstractText"):
        body = _clean(_text_of(section))
        if not body:
            continue
        label = section_label(section.get("Label") or "")
        sections.append(f"{label}: {body}" if label and label.upper() != "UNLABELLED" else body)
    abstract = "\n".join(sections)
    languages = [lang.text for lang in article.findall("Language") if lang.text]
    mesh = [d.get("UI") for d in citation.findall("MeshHeadingList/MeshHeading/DescriptorName") if d.get("UI")]
    return {"pmid": int(pmid), "year": _year(article), "title": title, "abstract": abstract,
            "languages": languages, "mesh": mesh}


def iter_citations(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag == "PubmedArticle":
                record = parse_citation(element)
                element.clear()
                if record is not None:
                    yield record
            elif element.tag == "PubmedBookArticle":
                element.clear()


def citation_text(record: dict[str, Any]) -> str:
    title = record["title"].strip()
    if title.startswith("[") and title.rstrip(".").endswith("]"):
        title = title.rstrip(".")[1:-1].strip() + "."   # translated titles are bracketed
    return f"{title}\n\n{record['abstract']}" if title else record["abstract"]


def extract_file(path: Path, out_path: Path, *, min_abstract_words: int = 30,
                 english_only: bool = True) -> dict[str, Any]:
    """Write the usable citations of one baseline file to parquet (zstd); return counts."""
    counts = {"citations": 0, "no_abstract": 0, "short_abstract": 0, "not_english": 0, "kept": 0,
              "extractor_version": EXTRACTOR_VERSION}
    columns: dict[str, list] = {"pmid": [], "year": [], "text": [], "mesh": []}
    for record in iter_citations(path):
        counts["citations"] += 1
        if not record["abstract"]:
            counts["no_abstract"] += 1; continue
        if english_only and "eng" not in record["languages"]:
            counts["not_english"] += 1; continue
        if len(record["abstract"].split()) < min_abstract_words:
            counts["short_abstract"] += 1; continue
        columns["pmid"].append(record["pmid"]); columns["year"].append(record["year"])
        columns["text"].append(citation_text(record)); columns["mesh"].append(record["mesh"])
        counts["kept"] += 1
    table = pa.table({"pmid": pa.array(columns["pmid"], pa.int64()), "year": pa.array(columns["year"], pa.int16()),
                      "text": pa.array(columns["text"], pa.string()),
                      "mesh": pa.array(columns["mesh"], pa.list_(pa.string()))})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".part")
    pq.write_table(table, partial, compression="zstd", row_group_size=4096)
    partial.replace(out_path)
    counts["characters"] = int(sum(len(t) for t in columns["text"]))
    return counts


def _extract_job(args: tuple[str, str, int, bool]) -> tuple[str, dict[str, Any]]:
    source, target, min_words, english = args
    counts = extract_file(Path(source), Path(target), min_abstract_words=min_words, english_only=english)
    Path(target + ".json").write_text(json.dumps(counts) + "\n")
    return Path(source).name, counts


def extract_baseline(names: list[str], raw_dir: Path, text_dir: Path, *, workers: int = 4,
                     min_abstract_words: int = 30, english_only: bool = True) -> dict[str, dict[str, Any]]:
    """Extract every listed file (resumable: a finished file has `<name>.parquet.json`)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    results: dict[str, dict[str, Any]] = {}
    jobs = []
    for name in names:
        target = text_dir / name.replace(".xml.gz", ".parquet")
        done = Path(str(target) + ".json")
        finished = json.loads(done.read_text()) if target.exists() and done.exists() else {}
        if finished.get("extractor_version") == EXTRACTOR_VERSION:
            results[name] = finished
        else:
            jobs.append((str(raw_dir / name), str(target), min_abstract_words, english_only))
    if jobs:
        with ProcessPoolExecutor(max(1, min(workers, len(jobs))), mp_context=get_context("spawn")) as pool:
            for name, counts in pool.map(_extract_job, jobs):
                results[name] = counts
    return {name: results[name] for name in names}


def text_paths(names: list[str], text_dir: Path) -> list[Path]:
    return [text_dir / name.replace(".xml.gz", ".parquet") for name in names]


def iter_pubmed(paths: list[Path], *, columns: tuple[str, ...] = ("pmid", "text")) -> Iterator[dict[str, Any]]:
    """Records of the extracted parquet files, in file order."""
    for path in paths:
        parquet = pq.ParquetFile(path)   # keep referenced while iterating (pyarrow segfault otherwise)
        for batch in parquet.iter_batches(columns=list(columns), batch_size=2048):
            data = batch.to_pydict()
            for i in range(batch.num_rows):
                yield {k: data[k][i] for k in columns}


def pmid_bucket(pmid: int, buckets: int = 10_000) -> int:
    """Stable hash bucket of a PMID (eval/train split independent of file order)."""
    return int.from_bytes(hashlib.sha256(str(int(pmid)).encode()).digest()[:8], "big") % buckets
