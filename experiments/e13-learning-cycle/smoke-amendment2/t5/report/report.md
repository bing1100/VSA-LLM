# SMOKE — E13 learning cycle — t5

Pre-registration: `experiments/e13-learning-cycle/preregistration.md` (amendment 2: L1's co-primaries, the definition comparators, L5's cost criterion).

## Primary endpoints (primary host; Holm across L1 (read − noread), L1 (read − random), L2, L3, L4, L5)

| endpoint | estimate | 95% CI | p | p (Holm) | met |
|---|---:|---|---:|---:|---|
| L1 read − noread | -0.009471 | [-0.02093, 0.002315] | 0.1194 | 0.597 | False |
| L1 read − random (gate) | -0.005057 | [-0.02164, 0.01153] | 0.597 | 1 | False |
| L2 | 7.626 | [7.378, —] | 0.00995 | 0.0597 | False |
| L3 | -0.005447 | [-0.01401, 0.004189] | 0.3483 | 1 | False |
| L4 precision (null rate) | — (0) | None | 1 | 1 | False |
| L5 | 0.005245 | [-0.05793, 0.08328] | 0.9055 | 1 | False |

**L1 (both co-primaries met): False** (None: not evaluable without the `random` arm).

## Definition comparators (secondary; amendment 2)

| comparison | estimate | 95% CI | added prompt tokens per item / window (accuracy gained over none per 100 tokens) |
|---|---:|---|---|
| L1 read − definition (step-0 loss) | 2.152 | [1.311, 3.667] | read 0 / definition 99 per window (read: 2682 forward tokens once per term) |
| L5 recall:own − definition (accuracy; δ 0.05: non-inferior False) | -0.1914 | [-0.2273, -0.1402] | recall:own 44.86 (0 per 100) / definition 140 (0.1394 per 100) / symbolic 158.5 (0.103 per 100) |

L5 cost criterion (store advantage over the definition): False.

## Per host

```json
{
 "SmolLM2-135M": {
  "L1": {
   "mean": -0.009471437111642445,
   "ci_low": -0.020934893705998547,
   "ci_high": 0.0023152902906076775,
   "p_value": 0.11940298507462686,
   "clusters": 4,
   "seeds": 1,
   "resamples": 200
  },
  "L1_random": {
   "mean": -0.005056794170741341,
   "ci_low": -0.021644854379701428,
   "ci_high": 0.011531266038218746,
   "p_value": 0.5970149253731343,
   "clusters": 4,
   "seeds": 1,
   "resamples": 200
  },
  "L1_definition": {
   "read_minus_definition": {
    "mean": 2.1522869522323163,
    "ci_low": 1.310839826004667,
    "ci_high": 3.6669653677119642,
    "p_value": 0.009950248756218905,
    "clusters": 4,
    "seeds": 1,
    "resamples": 200
   },
   "read_minus_definition_defined_windows": {
    "mean": 2.1522869522323163,
    "ci_low": 1.310839826004667,
    "ci_high": 3.6669653677119642,
    "p_value": 0.009950248756218905,
    "clusters": 4,
    "seeds": 1,
    "resamples": 200
   },
   "definition_minus_none": {
    "mean": -2.1617583893439587,
    "ci_low": -3.6646500774213564,
    "ci_high": -1.3317747197106655,
    "p_value": 0.009950248756218905,
    "clusters": 4,
    "seeds": 1,
    "resamples": 200
   },
   "windows": 4,
   "windows_with_definitions": 4,
   "tokens": {
    "definition_added_per_window": 99.0,
    "definition_added_per_defined_window": 99.0,
    "read_added_per_window": 0.0,
    "read_one_off_forward_tokens_per_term": 2682.5
   }
  },
  "L2": {
   "ratio": 7.625562744627253,
   "ci_low": 7.377892043426806,
   "ci_high": Infinity,
   "p_value": 0.009950248756218905,
   "aulc_difference": -0.016734437952849568,
   "aulc_ci": [
    -0.02349426532873622,
    0.0047659257834311575
   ],
   "offset_tokens": 3576.6666666666665,
   "seeds": 1,
   "windows": 4,
   "resamples": 200
  },
  "L2_without_reading_cost": {
   "ratio": 0.6398856612939201,
   "ci_low": 0.3922149600934731,
   "ci_high": Infinity,
   "p_value": 0.07960199004975124,
   "aulc_difference": -0.016734437952849568,
   "aulc_ci": [
    -0.02349426532873622,
    0.0047659257834311575
   ],
   "offset_tokens": 0.0,
   "seeds": 1,
   "windows": 4,
   "resamples": 200
  },
  "secondary_vs_noread": {
   "fvt": {
    "step0": {
     "mean": -0.016922278393394663,
     "ci_low": -0.062320342020029784,
     "ci_high": 0.016661510897392873,
     "p_value": 0.5373134328358209,
     "clusters": 4,
     "seeds": 1,
     "resamples": 200
    },
    "end": {
     "mean": -0.015559257921268,
     "ci_low": -0.06159267020484549,
     "ci_high": 0.016527187362953555,
     "p_value": 0.5373134328358209,
     "clusters": 4,
     "seeds": 1,
     "resamples": 200
    },
    "efficiency": {
     "ratio": 0.4941533629916455,
     "ci_low": 0.0,
     "ci_high": Infinity,
     "p_value": 0.5870646766169154,
     "aulc_difference": -0.024006988134218865,
     "aulc_ci": [
      -0.07319826346501923,
      0.016985741308114033
     ],
     "offset_tokens": 0.0,
     "seeds": 1,
     "windows": 4,
     "resamples": 200
    }
   },
   "random": {
    "step0": {
     "mean": -0.004414642940901103,
     "ci_low": -0.017762818823030102,
     "ci_high": 0.005689381025149487,
     "p_value": 0.4577114427860697,
     "clusters": 4,
     "seeds": 1,
     "resamples": 200
    },
    "end": {
     "mean": -0.00412131917073566,
     "ci_low": -0.013729391863307683,
     "ci_high": 0.005486753521836363,
     "p_value": 0.417910447761194,
     "clusters": 4,
     "seeds": 1,
     "resamples": 200
    },
    "efficiency": {
     "ratio": 0.9410069125476879,
     "ci_low": 0.5305050357770689,
     "ci_high": Infinity,
     "p_value": 0.6268656716417911,
     "aulc_difference": -0.0037669084915910034,
     "aulc_ci": [
      -0.01809666231274587,
      0.009397460508625777
     ],
     "offset_tokens": 0.0,
     "seeds": 1,
     "windows": 4,
     "resamples": 200
    }
   }
  },
  "forgetting": {
   "read": -0.047644564620573426,
   "noread": -0.04682392723504769,
   "fvt": -0.04677379616084654,
   "random": -0.04627036672153961
  },
  "locality": {
   "read": 0.0006835323689147188,
   "noread": 0.0006144334814690566,
   "fvt": 0.0006088006509497035,
   "random": 0.0006076724682397838
  },
  "L3": {
   "rtn": {
    "read_vs_noread_end": {
     "mean": -0.005447330226161284,
     "ci_low": -0.014008102552907076,
     "ci_high": 0.004188736871128647,
     "p_value": 0.3482587064676617,
     "clusters": 4,
     "seeds": 1,
     "resamples": 200
    },
    "read_vs_noread_step0": {
     "mean": -0.005765340247307904,
     "ci_low": -0.014438872662140056,
     "ci_high": 0.004593117785043439,
     "p_value": 0.3482587064676617,
     "clusters": 4,
     "seeds": 1,
     "resamples": 200
    },
    "vs_qlora_end": {
     "mean": 0.06310698771267198,
     "ci_low": 0.03336054179817438,
     "ci_high": 0.11962597261299379,
     "p_value": 0.009950248756218905,
     "clusters": 4,
     "seeds": 1,
     "resamples": 200,
     "margin": 0.04511670108286622,
     "noninferior": false
    },
    "bytes": {
     "q4-read": {
      "host_bytes": 116413056,
      "bits_per_weight": 4.5,
      "adapter_parameters": 0,
      "trainable_parameters": 1195403,
      "trainable_bytes_fp32": 4781612
     },
     "q4-noread": {
      "host_bytes": 116413056,
      "bits_per_weight": 4.5,
      "adapter_parameters": 0,
      "trainable_parameters": 1195403,
      "trainable_bytes_fp32": 4781612
     },
     "qlora": {
      "host_bytes": 116413056,
      "bits_per_weight": 4.5,
      "adapter_parameters": 19537920,
      "trainable_parameters": 20733323,
      "trainable_bytes_fp32": 82933292
     }
    }
   }
  },
  "L4": {
   "precision": NaN,
   "precision_ci": null,
   "accepted": 0,
   "erased_gold": 6024,
   "recall": 0.0,
   "null_rate": 0.0,
   "null_ci": [
    0.0,
    0.13798057582897535
   ],
   "p_value": 1.0
  },
  "write_items": {
   "accuracy": [
    {
     "none": 0.42592592592592593,
     "store:linker": 0.42592592592592593,
     "store:oracle": 0.42592592592592593,
     "store:random": 0.42592592592592593,
     "definition-in-context": 0.7222222222222222
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
      "n": 54,
      "p_value": 1.0
     },
     {
      "a": "store:oracle",
      "b": "none",
      "norm": "sum",
      "mean": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "n": 54,
      "p_value": 1.0
     },
     {
      "a": "store:random",
      "b": "none",
      "norm": "sum",
      "mean": 0.0,
      "ci_low": 0.0,
      "ci_high": 0.0,
      "n": 54,
      "p_value": 1.0
     },
     {
      "a": "definition-in-context",
      "b": "none",
      "norm": "sum",
      "mean": 0.2962962962962963,
      "ci_low": 0.14814814814814814,
      "ci_high": 0.42592592592592593,
      "n": 54,
      "p_value": 0.009950248756218905
     }
    ]
   ]
  },
  "L5": {
   "mean": 0.005244755244755241,
   "ci_low": -0.0579326923076923,
   "ci_high": 0.0832823426573427,
   "p_value": 0.9054726368159204,
   "clusters": 4,
   "seeds": 1,
   "resamples": 200,
   "by_family": {
    "recall:own": [
     {
      "negation/all": 0.546875,
      "negation/heldout": 0.546875,
      "reverse/all": 0.5625,
      "reverse/heldout": 0.5625,
      "two_hop/all": 0.05555555555555555,
      "two_hop/heldout": 0.05555555555555555
     }
    ],
    "none": [
     {
      "negation/all": 0.53125,
      "negation/heldout": 0.53125,
      "reverse/all": 0.5625,
      "reverse/heldout": 0.5625,
      "two_hop/all": 0.1111111111111111,
      "two_hop/heldout": 0.1111111111111111
     }
    ],
    "symbolic": [
     {
      "negation/all": 0.515625,
      "negation/heldout": 0.515625,
      "reverse/all": 0.625,
      "reverse/heldout": 0.625,
      "two_hop/all": 1.0,
      "two_hop/heldout": 1.0
     }
    ],
    "definition": [
     {
      "negation/all": 0.53125,
      "negation/heldout": 0.53125,
      "two_hop/all": 1.0,
      "two_hop/heldout": 1.0
     }
    ]
   },
   "tokens": {
    "recall:own": {
     "items": 49.0,
     "accuracy": 0.45918367346938777,
     "accuracy_none": 0.45918367346938777,
     "added_tokens": 44.857142857142854,
     "gain_per_100_tokens": 0.0,
     "per_seed": [
      {
       "items": 49,
       "accuracy": 0.45918367346938777,
       "accuracy_none": 0.45918367346938777,
       "added_tokens": 44.857142857142854,
       "gain_per_100_tokens": 0.0
      }
     ]
    },
    "definition": {
     "items": 41.0,
     "accuracy": 0.6341463414634146,
     "accuracy_none": 0.43902439024390244,
     "added_tokens": 139.97560975609755,
     "gain_per_100_tokens": 0.13939710751001916,
     "per_seed": [
      {
       "items": 41,
       "accuracy": 0.6341463414634146,
       "accuracy_none": 0.43902439024390244,
       "added_tokens": 139.97560975609755,
       "gain_per_100_tokens": 0.13939710751001916
      }
     ]
    },
    "symbolic": {
     "items": 49.0,
     "accuracy": 0.6224489795918368,
     "accuracy_none": 0.45918367346938777,
     "added_tokens": 158.48979591836735,
     "gain_per_100_tokens": 0.10301313417460727,
     "per_seed": [
      {
       "items": 49,
       "accuracy": 0.6224489795918368,
       "accuracy_none": 0.45918367346938777,
       "added_tokens": 158.48979591836735,
       "gain_per_100_tokens": 0.10301313417460727
      }
     ]
    }
   },
   "vs_definition": {
    "mean": -0.19141414141414143,
    "ci_low": -0.22727272727272727,
    "ci_high": -0.14015151515151514,
    "p_value": 0.009950248756218905,
    "clusters": 4,
    "seeds": 1,
    "resamples": 200,
    "margin": 0.05,
    "noninferior": false,
    "superior": false,
    "inferior": true,
    "tokens_recall_minus_definition": {
     "mean": -108.84419191919191,
     "ci_low": -150.59166666666667,
     "ci_high": -74.46111111111111,
     "p_value": 0.009950248756218905,
     "clusters": 4,
     "seeds": 1,
     "resamples": 200,
     "fewer": true
    }
   }
  }
 }
}
```
