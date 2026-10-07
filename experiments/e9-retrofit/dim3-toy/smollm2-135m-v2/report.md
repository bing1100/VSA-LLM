# Weight-editing toy check — HuggingFaceTB/SmolLM2-135M

Margins are log p(new) − log p(true) (summed over target tokens); efficacy/paraphrase should turn positive after the edit, the neighbour should stay negative. ROME edits one fact at a time; MEMIT/AlphaEdit edit all four at once.

| method | subject | true → new | efficacy before → after | paraphrase before → after | neighbour before → after |
|---|---|---|---|---|---|
| rome | The Eiffel Tower | Paris → Rome | -8.04 → +12.49 | -6.00 → +11.02 | -6.49 → -6.11 |
| rome | The Great Wall | China → Brazil | -8.98 → +15.93 | -6.31 → +10.14 | -11.79 → +0.01 |
| rome | Danielle Darrieux | French → English | -1.45 → +7.13 | -1.53 → +2.47 | -0.58 → -0.63 |
| rome | The iPhone | Apple → Nokia | -3.33 → +6.91 | -2.98 → +0.48 | -3.29 → +3.71 |
| memit | The Eiffel Tower | Paris → Rome | -8.04 → +2.82 | -6.00 → +2.20 | -6.49 → -5.82 |
| memit | The Great Wall | China → Brazil | -8.98 → +11.05 | -6.31 → +6.75 | -11.79 → -7.86 |
| memit | Danielle Darrieux | French → English | -1.45 → +3.28 | -1.53 → +1.10 | -0.58 → -0.59 |
| memit | The iPhone | Apple → Nokia | -3.33 → +2.57 | -2.98 → -2.62 | -3.29 → -0.59 |
| alphaedit | The Eiffel Tower | Paris → Rome | -8.04 → +11.85 | -6.00 → +9.21 | -6.49 → -5.55 |
| alphaedit | The Great Wall | China → Brazil | -8.98 → +14.67 | -6.31 → +9.57 | -11.79 → -6.71 |
| alphaedit | Danielle Darrieux | French → English | -1.45 → +7.24 | -1.53 → +3.08 | -0.58 → -0.59 |
| alphaedit | The iPhone | Apple → Nokia | -3.33 → +6.62 | -2.98 → -1.81 | -3.29 → +1.91 |

| method | knew the true facts | efficacy | paraphrase | neighbour kept |
|---|---|---:|---:|---:|
| rome | True | 1.00 | 1.00 | 0.50 |
| memit | True | 1.00 | 0.75 | 1.00 |
| alphaedit | True | 1.00 | 0.75 | 0.75 |
