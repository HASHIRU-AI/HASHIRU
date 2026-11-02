# Reference check for the HASHIRU v2 whitepaper

Checked 2026-11-02, before the whitepaper went public. Only the 37 keys the paper cites appear in the PDF, so these are
the ones checked. Every one matched its source; none was missing, fabricated or wrong.

## Checked by script (30)

Title and every author's surname compared with the source, or the URL or repository confirmed to exist.

| Source | Keys |
|---|---|
| arXiv API (title and full author list) | bansal2025magentic, hamri2026zebra, hu2024routerbench, jiang2026specialize, lin2026bagen, liu2026markets, pai2025hashiru, pecerskis2026mixtureofmodels, sabbatella2025malbo, tomasev2025virtual, wang2026allocbench, wang2026r3bench, wu2026ecoagent, zhao2026atom, zhao2026scmas |
| Crossref DOI metadata | li2025papillon, liu2026costbench, yang2026bamas, yue2025masrouter |
| GitHub API (repository exists) | berriai2026litellm, conway2026automaton, ferrox2026waylandcore, googleagenticcommerce2025a2a, langchain2026langgraph, ollama2026ollama, presidio2026presidio |
| URL returns 200 | a2aproject2026agent2agent, google2026agent, modelcontextprotocol2026specification, narayan2025costefficient (PMLR) |

## Checked by hand (7)

| Key | Why by hand | Confirmed by |
|---|---|---|
| fourney2024magenticone | arXiv's record splits "Erkang (Eric) Zhu" into two names | the other 19 authors match in order; the bib's "Zhu, Erkang" is right |
| chen2024frugalgpt | OpenReview's API refuses anonymous requests | arXiv 2305.05176 (authors match); TMLR 2024 per mlanthology.org/tmlr/2024/chen2024tmlr-frugalgpt |
| ding2024hybrid | same | ICLR 2024 proceedings page; 8 authors match |
| hong2024metagpt | same | ICLR 2024 proceedings page; forum id VtmBAGCN7o in MetaGPT's own BibTeX |
| liu2026budgetaware | same | arXiv 2511.17006, comment "Accepted to COLM 2026"; 15 authors match |
| ong2025routellm | same | ICLR 2025 proceedings page; 8 authors match |
| wu2024autogen | same | Microsoft Research lists COLM 2024; arXiv 2308.08155 has the same 14 authors |

Script: arXiv API with a 3 s pause between calls, Crossref works API, `gh api repos/...`, and plain GET for URLs.
