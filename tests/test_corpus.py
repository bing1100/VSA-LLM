from pathlib import Path

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import TokenCorpus, build_corpus, eval_windows, sample_batch
from vsa_embed.span_channel import AliasTable


def _tokenizer_available() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _tokenizer_available(), reason="gpt2 tokenizer not cached")
def test_build_and_sample_corpus(tmp_path: Path) -> None:
    table = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("cat", 2)])
    texts = ["The cat went to New York. Hydroxychloroquine is a drug."] * 20
    manifest = build_corpus(texts, tmp_path, tokenizer_name="gpt2", table=table, eos_id=50256,
                            max_tokens=10_000, batch_texts=4, workers=2)
    corpus = TokenCorpus.open(tmp_path)
    assert len(corpus) == manifest["tokens"] and manifest["documents"] == 20
    ids, spans = corpus.window(0, 40)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    for start, end in zip(spans["start"], spans["end"]):
        text = tokenizer.decode(ids[start:end + 1]).strip().lower().rstrip(".")
        assert text in table.alias_to_entry
    first = sample_batch(corpus, seed=1, step=5, micro_step=0, batch=3, length=16)
    again = sample_batch(corpus, seed=1, step=5, micro_step=0, batch=3, length=16)
    assert np.array_equal(first[0].numpy(), again[0].numpy())
    long_only = sample_batch(corpus, seed=1, step=5, micro_step=0, batch=3, length=16, min_subtokens=2)
    assert bool((long_only[1]["length"] >= 2).all())
    starts = eval_windows(corpus, count=5, length=16)
    assert len(starts) == 5 and starts == sorted(starts)


def test_window_search_matches_for_int32_and_int64_span_index() -> None:
    import numpy as np
    from vsa_embed.data.corpus import TokenCorpus
    rng = np.random.default_rng(0)
    inject = np.sort(rng.integers(0, 5000, size=400))
    spans = {"start": inject - 1, "end": inject, "inject": inject, "entry": rng.integers(0, 50, size=400),
             "length": np.full(400, 2), "confidence": np.ones(400, dtype=np.float16)}
    tokens = np.arange(5100, dtype=np.uint16)
    wide = TokenCorpus.__new__(TokenCorpus); wide.tokens = tokens; wide.spans = {k: v.astype(np.int64) if k != "confidence" else v for k, v in spans.items()}
    narrow = TokenCorpus.__new__(TokenCorpus); narrow.tokens = tokens; narrow.spans = {k: v.astype(np.int32) if k != "confidence" else v for k, v in spans.items()}
    for start in (0, 17, 999, 4000):
        a, b = wide.window(start, 256), narrow.window(start, 256)
        assert (a[0] == b[0]).all() and all((a[1][k] == b[1][k]).all() for k in a[1])
