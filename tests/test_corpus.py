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
