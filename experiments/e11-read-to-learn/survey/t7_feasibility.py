import collections
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from vsa_embed.data.corpus import TokenCorpus
from vsa_embed.ontologies.mesh_novel import parse_supplementary

BOILER = re.compile(r"^(rn given|structure|mf given|in first source|for .* see|see also|no structure|mixture of|"
                    r"also see|isomer|.*\bfirst source\b|.*\bgiven in\b.*source)", re.I)


def informative(note: str) -> str:
    parts = [p.strip() for p in note.split(";")]
    kept = [p for p in parts if p and not BOILER.match(p) and len(p.split()) >= 2]
    return "; ".join(kept)


records = {r["ui"]: r for r in parse_supplementary(Path("/home/bhux/data/vsa-llm/mesh/supp-2026/supp2026.gz"))}
o = torch.load("/home/bhux/data/vsa-llm/tracks/t7-newvocab/v1/ontology.pt", weights_only=False)
names, ec = o["concept_names"], o["entry_concepts"]
held = [int(e) for e in o["heldout_entries"]]
introduced = dict(zip(names, o.get("introduced") or [None] * len(names))) if "introduced" in o else {}
ev = TokenCorpus.open(Path("/home/bhux/data/vsa-llm/tracks/t7-newvocab/v1/eval-pubmed"))
keep = ev.spans["length"] >= 2
occ = collections.Counter(int(e) for e in ev.spans["entry"][keep])
rows = []
for e in held:
    ui = names[ec[e][0]]
    r = records.get(ui, {})
    text = informative(r.get("note") or "")
    rows.append((e, ui, r.get("name"), r.get("introduced"), text, occ.get(e, 0)))
good = [r for r in rows if len(r[4].split()) >= 3]
print("held-out", len(rows), "informative notes (>=3 words)", len(good), "with eval occurrences", sum(1 for r in good if r[5]),
      "occurrences", sum(r[5] for r in good))
years = collections.Counter((r[3] or 0) for r in good)
print("introduced (informative):", dict(sorted(years.items())[-8:]), " >=2023:", sum(1 for r in good if (r[3] or 0) >= 2023))
print("all held-out introduced >= 2023:", sum(1 for r in rows if (r[3] or 0) >= 2023), "with occurrences",
      sum(1 for r in rows if (r[3] or 0) >= 2023 and r[5]))
print("median informative words", np.median([len(r[4].split()) for r in good]))
for r in good[:10]:
    print(" ", r[2], r[3], r[5], "|", r[4][:150])
# windows with >= 2 distinct held-out read terms (multi-term passages), 1,024-token tiles
heldset = {r[0] for r in good}
inject = ev.spans["inject"][keep]
entry = ev.spans["entry"][keep]
tiles = collections.defaultdict(set)
for pos, ent in zip(inject.tolist(), entry.tolist()):
    if ent in heldset:
        tiles[pos // 1024].add(ent)
print("1,024-token tiles with >=1 / >=2 / >=3 distinct informative held-out terms:",
      sum(1 for s in tiles.values() if s), sum(1 for s in tiles.values() if len(s) >= 2), sum(1 for s in tiles.values() if len(s) >= 3))
