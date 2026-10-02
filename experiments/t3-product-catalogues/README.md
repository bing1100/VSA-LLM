# T3 product catalogues (E8, task C7)

Google Product Taxonomy categories as concepts, linked into Amazon product texts; Shopify's open
taxonomy contributes product attributes and finer category names.

## Sources and licences (details and hashes in `~/data/vsa-llm/DATA_SOURCES.md`, section "C7 application tracks T3–T6")

- **Google Product Taxonomy** 2021-09-21 (`taxonomy-with-ids.en-US.txt`, 5,595 categories). Google
  publishes it for merchants' product feeds without an explicit licence; only category names appear in
  committed items.
- **Shopify Standard Product Taxonomy** (MIT), commit `3ef0220a…`: category attributes and the official
  Shopify → Google 2021-09-21 mapping. Used as the open substitute for **GS1 GPC**, which is reachable
  through the GPC Browser but only under GS1's website terms (redistribution of derived items unclear).
- **Amazon Shopping Queries (ESCI)** (Apache-2.0): product catalogue (1,215,854 US products) and the
  query–product judgements. The WDC product corpus was not needed (ESCI alone exceeds 100M tokens).

## Recipe

1. **Ontology** (`ontologies/google_product.py`): concept = category; frames (≤ 16 edges) = parent,
   top and second-level category, depth, name tokens, ancestors' path tokens (the taxonomy's atomics,
   naively singularised) and up to 6 Shopify attributes; 8,192 atomics. Aliases: the category name,
   below level 2 its "&"-parts (a one-word modifier takes the head noun: "Liquid & Frozen Eggs" →
   "Liquid Eggs"), multi-word Shopify names mapped onto it, singulars of multi-word aliases; generic
   single words and aliases shared by > 8 categories are dropped (28,894 aliases, 6,151 entries).
2. **Corpus**: US ESCI products as "title / brand / colour / bullets / description" (HTML stripped), in
   a hashed order; 2% of products for evaluation (3.0M tokens used). The training stream is 100M tokens
   of product text (387,946 products) — no general text is mixed in.
3. **Holdout** (C3 procedure, 5M-token presample): 10% of the 1,662 eligible entries → 182 categories,
   225 entries (`items/holdout.json`).
4. **Synthetic product types and SKUs** (E5.4): 300 invented names under existing internal categories,
   100 each `type_invented` (no surface cue), `type_headed` (invented stem + a head noun of the parent's
   children) and `sku` (invented brand + model code + head noun).
5. **Feasibility**: see `runs/v1/report.md`. **Infeasible at every ℓ_min under the fixed criterion**:
   the seen-rare stratum is empty — 100M tokens of product text see almost every linkable category ≥ 10
   times, so only ≈ 38 rare entries (64 spans) occur in 1M evaluation tokens at ℓ_min = 2. The held-out
   stratum and locality alone are powered at ℓ_min = 2 with 1,024 evaluation windows (148 held-out
   entries, 1,401 spans; linked text covers 3% of tokens). Whether T3 runs on the held-out part of the
   gate only is the author's call (see the open decisions in the C7 report).
6. **Items**: `category_probe` (product title → level-1 / level-2 category of the one category its
   title links; distant supervision through the linker, since ESCI has no category labels),
   `relevance_probe` (ESCI US small-version E/S/C/I judgements; 8,000 from ESCI train, 4,000 from ESCI
   test; flagged by whether the title links a held-out category), `zeroshot_property` and
   `zeroshot_entailment` (parent / department / product group / attribute facts of synthetic and
   held-out categories).

## Build

```bash
PYTHONPATH=src python -m vsa_embed.experiments.track_corpus \
  --config experiments/t3-product-catalogues/t3.yaml --output experiments/t3-product-catalogues/runs/v1
```

≈ 4 min with 3 tokenizer workers (≈ 6 GB peak RSS while the product table is shuffled). Outputs in
`~/data/vsa-llm/tracks/t3-product/v1/`.
