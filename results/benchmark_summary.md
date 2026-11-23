| Agent | Success | Mean clicks (wins) | Extra clicks vs optimal | Optimal paths | Time / game | LLM calls / game | Pages loaded / game |
|---|---|---|---|---|---|---|---|
| `random` | 4/22 (18%) | 8.0 | +6.5 | 50% | 0.0s | 0.0 | 17.8 |
| `greedy` | 20/22 (91%) | 4.7 | +2.5 | 20% | 0.9s | 0.0 | 6.0 |
| `lookahead` | 20/22 (91%) | 3.1 | +1.0 | 40% | 0.5s | 0.0 | 20.5 |
| `llm` | 20/22 (91%) | 4.1 | +1.9 | 35% | 7.9s | 4.6 | 5.5 |
| `llm-cot` | 21/22 (95%) | 4.9 | +2.8 | 29% | 35.4s | 4.6 | 5.6 |
