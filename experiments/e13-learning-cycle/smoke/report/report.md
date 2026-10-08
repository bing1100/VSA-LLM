# SMOKE — E13 learning cycle — t5

Pre-registration: `experiments/e13-learning-cycle/preregistration.md`.

## Primary endpoints (primary host; Holm across L1–L5)

| endpoint | estimate | 95% CI | p | p (Holm) | met |
|---|---:|---|---:|---:|---|
| L1 | -0.009471 | [-0.02093, 0.002315] | 0.1194 | 0.4776 | False |
| L2 | 7.626 | [7.378, —] | 0.00995 | 0.04975 | False |
| L3 | -0.005447 | [-0.01401, 0.004189] | 0.3483 | 1 | False |
| L4 precision (null rate) | — (0) | None | 1 | 1 | False |
| L5 | 0.01761 | [-0.1528, 0.1865] | 0.796 | 1 | False |

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
   }
  },
  "forgetting": {
   "read": -0.047644564620573426,
   "noread": -0.04682392723504769,
   "fvt": -0.04677379616084654
  },
  "locality": {
   "read": 0.0006835323689147188,
   "noread": 0.0006144334814690566,
   "fvt": 0.0006088006509497035
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
   "mean": 0.017609126984126977,
   "ci_low": -0.1527777777777778,
   "ci_high": 0.1865079365079365,
   "p_value": 0.7960199004975125,
   "clusters": 4,
   "seeds": 1,
   "resamples": 200,
   "by_family": {
    "recall:own": [
     {
      "negation/all": 0.40625,
      "negation/heldout": 0.40625,
      "reverse/all": 0.5625,
      "reverse/heldout": 0.5625,
      "two_hop/all": 0.05555555555555555,
      "two_hop/heldout": 0.05555555555555555
     }
    ],
    "none": [
     {
      "negation/all": 0.375,
      "negation/heldout": 0.375,
      "reverse/all": 0.5625,
      "reverse/heldout": 0.5625,
      "two_hop/all": 0.1111111111111111,
      "two_hop/heldout": 0.1111111111111111
     }
    ],
    "symbolic": [
     {
      "negation/all": 0.90625,
      "negation/heldout": 0.90625,
      "reverse/all": 0.625,
      "reverse/heldout": 0.625,
      "two_hop/all": 1.0,
      "two_hop/heldout": 1.0
     }
    ],
    "definition": [
     {
      "negation/all": 0.84375,
      "negation/heldout": 0.84375,
      "two_hop/all": 1.0,
      "two_hop/heldout": 1.0
     }
    ]
   }
  }
 }
}
```
