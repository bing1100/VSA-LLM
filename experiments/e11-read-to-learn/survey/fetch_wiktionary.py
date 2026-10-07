"""Fetch Wiktionary 'Category:English neologisms' members and their wikitext (MediaWiki API, serial, 1 req/s)."""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1]).expanduser()
OUT.mkdir(parents=True, exist_ok=True)
UA = "VSA-LLM-research/0.1 (bingxuhu@gmail.com) read-to-learn survey"
API = "https://en.wiktionary.org/w/api.php"


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read())
    time.sleep(1.0)
    return data


members, cont = [], {}
while True:
    data = get({"action": "query", "list": "categorymembers", "cmtitle": "Category:English neologisms", "cmlimit": "500",
                "cmnamespace": "0", **cont})
    members += [m["title"] for m in data["query"]["categorymembers"]]
    if "continue" not in data:
        break
    cont = {"cmcontinue": data["continue"]["cmcontinue"]}
print("members", len(members), flush=True)
pages = {}
for start in range(0, len(members), 50):
    titles = members[start:start + 50]
    data = get({"action": "query", "prop": "revisions", "rvprop": "content|ids|timestamp", "rvslots": "main",
                "titles": "|".join(titles)})
    for page in data["query"]["pages"]:
        revision = (page.get("revisions") or [{}])[0]
        pages[page["title"]] = {"revid": revision.get("revid"), "timestamp": revision.get("timestamp"),
                                "wikitext": revision.get("slots", {}).get("main", {}).get("content", "")}
(OUT / "members.json").write_text(json.dumps(members, indent=0))
with (OUT / "pages.jsonl").open("w") as handle:
    for title in members:
        handle.write(json.dumps({"title": title, **pages.get(title, {})}) + "\n")
(OUT / "source.json").write_text(json.dumps({
    "api": API, "category": "Category:English neologisms", "retrieved": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "members": len(members), "pages": len(pages),
    "licence": "CC BY-SA 4.0 (and GFDL) — Wiktionary text; attribution: Wiktionary contributors, revision ids per page"},
    indent=2))
print("pages", len(pages))
