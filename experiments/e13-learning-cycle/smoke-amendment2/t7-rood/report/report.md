# SMOKE — E13 learning cycle — t7-rood

Pre-registration: `experiments/e13-learning-cycle/preregistration.md` (amendment 2: L1's co-primaries, the definition comparators, L5's cost criterion).

## Primary endpoints (primary host; Holm across L1 (read − noread), L1 (read − random), L2, L3, L4, L5)

| endpoint | estimate | 95% CI | p | p (Holm) | met |
|---|---:|---|---:|---:|---|
| L1 read − noread | 0.001093 | [0, 0.00328] | 0.7264 | 1 | False |
| L1 read − random (gate) | 0.01106 | [0, 0.02038] | 0.1592 | 0.9552 | False |
| L2 | — | [—, —] | 1 | 1 | False |
| L3 | 0.001731 | [0, 0.005192] | 0.7264 | 1 | False |
| L4 precision (null rate) | — (0) | None | 1 | 1 | False |
| L5 | 0.08333 | [-0.08333, 0.1667] | 0.5473 | 1 | False |

**L1 (both co-primaries met): False** (None: not evaluable without the `random` arm).

## Definition comparators (secondary; amendment 2)

| comparison | estimate | 95% CI | added prompt tokens per item / window (accuracy gained over none per 100 tokens) |
|---|---:|---|---|
| L1 read − definition (step-0 loss) | 0.1051 | [0, 0.1651] | read 0 / definition 12.67 per window (read: 104 forward tokens once per term) |
| L5 recall:own − definition (accuracy; δ 0.05: non-inferior True) | 0.25 | [0.25, 0.25] | recall:own 60.33 (0.1381 per 100) / definition 14 (0 per 100) / symbolic 116 (0.1796 per 100) |

L5 cost criterion (store advantage over the definition): False.

## Per host

```json
{
 "SmolLM2-135M": {
  "L1": {
   "mean": 0.0010932025810082753,
   "ci_low": 0.0,
   "ci_high": 0.003279607743024826,
   "p_value": 0.7263681592039801,
   "clusters": 3,
   "seeds": 1,
   "resamples": 200
  },
  "L1_random": {
   "mean": 0.011064768768846989,
   "ci_low": 0.0,
   "ci_high": 0.020379366376437287,
   "p_value": 0.15920398009950248,
   "clusters": 3,
   "seeds": 1,
   "resamples": 200
  },
  "L1_definition": {
   "read_minus_definition": {
    "mean": 0.10506284376606345,
    "ci_low": 0.0,
    "ci_high": 0.16514191910391682,
    "p_value": 0.15920398009950248,
    "clusters": 3,
    "seeds": 1,
    "resamples": 200
   },
   "read_minus_definition_defined_windows": {
    "mean": 0.15759426564909518,
    "ci_low": 0.13602954149246216,
    "ci_high": 0.1791589898057282,
    "p_value": 0.009950248756218905,
    "clusters": 2,
    "seeds": 1,
    "resamples": 200
   },
   "definition_minus_none": {
    "mean": -0.10396964118505518,
    "ci_low": -0.16407604658743366,
    "ci_high": 0.0,
    "p_value": 0.15920398009950248,
    "clusters": 3,
    "seeds": 1,
    "resamples": 200
   },
   "windows": 3,
   "windows_with_definitions": 2,
   "tokens": {
    "definition_added_per_window": 12.666666666666666,
    "definition_added_per_defined_window": 19.0,
    "read_added_per_window": 0.0,
    "read_one_off_forward_tokens_per_term": 104.0
   }
  },
  "L2": {
   "ratio": NaN,
   "ci_low": null,
   "ci_high": null,
   "p_value": 1.0,
   "aulc_difference": 0.0006764108315113759,
   "aulc_ci": [
    0.0,
    0.0033820541575551033
   ],
   "offset_tokens": 69.33333333333333,
   "seeds": 1,
   "windows": 3,
   "resamples": 200
  },
  "L2_without_reading_cost": {
   "ratio": NaN,
   "ci_low": null,
   "ci_high": null,
   "p_value": 1.0,
   "aulc_difference": 0.0006764108315113759,
   "aulc_ci": [
    0.0,
    0.0033820541575551033
   ],
   "offset_tokens": 0.0,
   "seeds": 1,
   "windows": 3,
   "resamples": 200
  },
  "secondary_vs_noread": {
   "fvt": {
    "step0": {
     "mean": 0.009029986899501333,
     "ci_low": -0.002119656652212143,
     "ci_high": 0.019986819264886494,
     "p_value": 0.7164179104477612,
     "clusters": 3,
     "seeds": 1,
     "resamples": 200
    },
    "end": {
     "mean": 0.00896867381137175,
     "ci_low": -0.0021624152723234147,
     "ci_high": 0.019954472285462602,
     "p_value": 0.7164179104477612,
     "clusters": 3,
     "seeds": 1,
     "resamples": 200
    },
    "efficiency": {
     "ratio": NaN,
     "ci_low": null,
     "ci_high": null,
     "p_value": 1.0,
     "aulc_difference": 0.011080905029666965,
     "aulc_ci": [
      -0.0021434711015899666,
      0.030235361788072623
     ],
     "offset_tokens": 0.0,
     "seeds": 1,
     "windows": 3,
     "resamples": 200
    }
   },
   "random": {
    "step0": {
     "mean": -0.009971566187838713,
     "ci_low": -0.019313493859954175,
     "ci_high": 0.0,
     "p_value": 0.15920398009950248,
     "clusters": 3,
     "seeds": 1,
     "resamples": 200
    },
    "end": {
     "mean": -0.010175716481171548,
     "ci_low": -0.019762591141625305,
     "ci_high": 0.0,
     "p_value": 0.15920398009950248,
     "clusters": 3,
     "seeds": 1,
     "resamples": 200
    },
    "efficiency": {
     "ratio": NaN,
     "ci_low": null,
     "ci_high": null,
     "p_value": 1.0,
     "aulc_difference": -0.011590535973664107,
     "aulc_ci": [
      -0.027750643523177132,
      0.0
     ],
     "offset_tokens": 0.0,
     "seeds": 1,
     "windows": 3,
     "resamples": 200
    }
   }
  },
  "forgetting": {
   "read": 0.0050197783857584,
   "noread": 0.005021007731556892,
   "fvt": 0.005271347239613533,
   "random": 0.0050201211124658585
  },
  "locality": {
   "read": 0.002248638957327742,
   "noread": 0.002248638957327742,
   "fvt": 0.002263057190412976,
   "random": 0.002248638957327742
  },
  "eval_round1_change": {
   "read": 0.0028826595635038643,
   "noread": 0.0028826595635038643,
   "fvt": 0.0029228597759929364,
   "random": 0.0028826595635038643
  },
  "L3": {
   "rtn": {
    "read_vs_noread_end": {
     "mean": 0.0017307748397191365,
     "ci_low": 0.0,
     "ci_high": 0.00519232451915741,
     "p_value": 0.7263681592039801,
     "clusters": 3,
     "seeds": 1,
     "resamples": 200
    },
    "read_vs_noread_step0": {
     "mean": 0.001701674113670985,
     "ci_low": 0.0,
     "ci_high": 0.005105022341012955,
     "p_value": 0.7263681592039801,
     "clusters": 3,
     "seeds": 1,
     "resamples": 200
    },
    "vs_qlora_end": {
     "mean": 0.017898464556007337,
     "ci_low": -0.030150339007377625,
     "ci_high": 0.052679762405750924,
     "p_value": 0.5671641791044776,
     "clusters": 3,
     "seeds": 1,
     "resamples": 200,
     "margin": 0.03308721793885343,
     "noninferior": false
    },
    "bytes": {
     "q4-read": {
      "host_bytes": 116413056,
      "bits_per_weight": 4.5,
      "adapter_parameters": 0,
      "trainable_parameters": 998139,
      "trainable_bytes_fp32": 3992556
     },
     "q4-noread": {
      "host_bytes": 116413056,
      "bits_per_weight": 4.5,
      "adapter_parameters": 0,
      "trainable_parameters": 998139,
      "trainable_bytes_fp32": 3992556
     },
     "qlora": {
      "host_bytes": 116413056,
      "bits_per_weight": 4.5,
      "adapter_parameters": 19537920,
      "trainable_parameters": 20536059,
      "trainable_bytes_fp32": 82144236
     }
    }
   }
  },
  "L4": {
   "precision": NaN,
   "precision_ci": null,
   "accepted": 0,
   "erased_gold": 2,
   "recall": 0.0,
   "null_rate": 0.0,
   "null_ci": [
    0.0,
    0.3903430336530645
   ],
   "p_value": 1.0,
   "null_rate_pooled": 0.0,
   "rule": "holm+decoy"
  },
  "write_items": {
   "accuracy": [
    {
     "none": 0.3333333333333333,
     "store:linker": 0.3333333333333333,
     "store:oracle": 0.3333333333333333,
     "store:random": 0.3333333333333333,
     "definition-in-context": 0.4166666666666667
    }
   ],
   "contrasts_vs_none": [
    [
     {
      "a": "store:linker",
      "b": "none",
      "norm": "sum",
      "mean": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "n": 12,
      "p_value": 1.0
     },
     {
      "a": "store:oracle",
      "b": "none",
      "norm": "sum",
      "mean": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "n": 12,
      "p_value": 1.0
     },
     {
      "a": "store:random",
      "b": "none",
      "norm": "sum",
      "mean": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "n": 12,
      "p_value": 1.0
     },
     {
      "a": "definition-in-context",
      "b": "none",
      "norm": "sum",
      "mean": 0.08333333333333333,
      "ci_low": 0.0,
      "ci_high": 0.25,
      "n": 12,
      "p_value": 0.736318407960199
     }
    ]
   ]
  },
  "L5": {
   "mean": 0.08333333333333333,
   "ci_low": -0.08333333333333334,
   "ci_high": 0.16666666666666666,
   "p_value": 0.5472636815920398,
   "clusters": 4,
   "seeds": 1,
   "resamples": 200,
   "by_family": {
    "recall:own": [
     {
      "negation/all": 0.5,
      "negation/heldout": 0.5,
      "reverse/all": 0.25,
      "reverse/heldout": 0.25
     }
    ],
    "none": [
     {
      "negation/all": 0.375,
      "negation/heldout": 0.375,
      "reverse/all": 0.25,
      "reverse/heldout": 0.25
     }
    ],
    "symbolic": [
     {
      "negation/all": 0.5,
      "negation/heldout": 0.5,
      "reverse/all": 0.625,
      "reverse/heldout": 0.625
     }
    ],
    "definition": [
     {
      "negation/all": 0.25,
      "negation/heldout": 0.25
     }
    ]
   },
   "tokens": {
    "recall:own": {
     "items": 12.0,
     "accuracy": 0.4166666666666667,
     "accuracy_none": 0.3333333333333333,
     "added_tokens": 60.333333333333336,
     "gain_per_100_tokens": 0.13812154696132603,
     "per_seed": [
      {
       "items": 12,
       "accuracy": 0.4166666666666667,
       "accuracy_none": 0.3333333333333333,
       "added_tokens": 60.333333333333336,
       "gain_per_100_tokens": 0.13812154696132603
      }
     ]
    },
    "definition": {
     "items": 2.0,
     "accuracy": 0.25,
     "accuracy_none": 0.25,
     "added_tokens": 14.0,
     "gain_per_100_tokens": 0.0,
     "per_seed": [
      {
       "items": 2,
       "accuracy": 0.25,
       "accuracy_none": 0.25,
       "added_tokens": 14.0,
       "gain_per_100_tokens": 0.0
      }
     ]
    },
    "symbolic": {
     "items": 12.0,
     "accuracy": 0.5416666666666666,
     "accuracy_none": 0.3333333333333333,
     "added_tokens": 116.0,
     "gain_per_100_tokens": 0.17959770114942528,
     "per_seed": [
      {
       "items": 12,
       "accuracy": 0.5416666666666666,
       "accuracy_none": 0.3333333333333333,
       "added_tokens": 116.0,
       "gain_per_100_tokens": 0.17959770114942528
      }
     ]
    }
   },
   "vs_definition": {
    "mean": 0.25,
    "ci_low": 0.25,
    "ci_high": 0.25,
    "p_value": 0.009950248756218905,
    "clusters": 1,
    "seeds": 1,
    "resamples": 200,
    "margin": 0.05,
    "noninferior": true,
    "superior": true,
    "inferior": false,
    "tokens_recall_minus_definition": {
     "mean": 2.0,
     "ci_low": 2.0,
     "ci_high": 2.0,
     "p_value": 0.009950248756218905,
     "clusters": 1,
     "seeds": 1,
     "resamples": 200,
     "fewer": false
    }
   },
   "anchors_with_definition": {
    "mean": 0.16666666666666666,
    "ci_low": 0.16666666666666666,
    "ci_high": 0.16666666666666666,
    "p_value": 0.009950248756218905,
    "clusters": 1,
    "seeds": 1,
    "resamples": 200
   }
  }
 }
}
```
