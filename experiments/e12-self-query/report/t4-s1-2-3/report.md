# E12 T4 — self-query

Pre-registration: `experiments/e12-self-query/preregistration.md`. Contrasts: units × seeds crossed model (Satterthwaite t, 95% CI, two-sided p; one seed: one-sample t), Holm within Q1 and within F1.

## SmolLM2-135M

Runs: {'P0': [1], 'C0p': [1], 'C2': [1], 'C5': [1]}

### Natural role items (choice)

| host model | condition | contrast | item | decode | seeds |
|---|---|---:|---:|---:|---|
| C0p | none | 0.552 | 0.511 | — | [1] |
| C0p | recall:C5 | 0.901 | 0.687 | 0.909 | [1] |
| C0p | symbolic | 0.931 | 0.700 | 1.000 | [1] |
| C0p | definition | 0.874 | 0.633 | — | [1] |
| C0p | roleless:C5 | 0.464 | 0.505 | 0.910 | [1] |
| C2 | none | 0.550 | 0.514 | — | [1] |
| C2 | recall:C5 | 0.901 | 0.686 | 0.909 | [1] |
| C2 | symbolic | 0.927 | 0.706 | 1.000 | [1] |
| C2 | definition | 0.878 | 0.635 | — | [1] |
| C2 | roleless:C5 | 0.482 | 0.504 | 0.910 | [1] |
| C5 | none | 0.565 | 0.517 | — | [1] |
| C5 | recall:own | 0.894 | 0.683 | 0.909 | [1] |
| C5 | symbolic | 0.924 | 0.697 | 1.000 | [1] |
| C5 | definition | 0.875 | 0.629 | — | [1] |
| C5 | roleless:own | 0.481 | 0.510 | 0.910 | [1] |
| P0 | none | 0.573 | 0.529 | — | [1] |
| P0 | recall:C5@1 | 0.865 | 0.636 | 0.909 | [1] |
| P0 | symbolic | 0.897 | 0.642 | 1.000 | [1] |
| P0 | definition | 0.833 | 0.599 | — | [1] |
| P0 | roleless:C5@1 | 0.515 | 0.506 | 0.910 | [1] |

### Natural role items (cloze)

| host model | condition | contrast | item | decode | seeds |
|---|---|---:|---:|---:|---|
| C0p | none | 0.610 | 0.549 | — | [1] |
| C0p | recall:C5 | 0.874 | 0.676 | 0.909 | [1] |
| C0p | symbolic | 0.897 | 0.690 | 1.000 | [1] |
| C0p | definition | 0.898 | 0.671 | — | [1] |
| C0p | roleless:C5 | 0.492 | 0.502 | 0.910 | [1] |
| C2 | none | 0.610 | 0.546 | — | [1] |
| C2 | recall:C5 | 0.871 | 0.666 | 0.909 | [1] |
| C2 | symbolic | 0.894 | 0.676 | 1.000 | [1] |
| C2 | definition | 0.895 | 0.663 | — | [1] |
| C2 | roleless:C5 | 0.502 | 0.508 | 0.910 | [1] |
| C5 | none | 0.603 | 0.549 | — | [1] |
| C5 | recall:own | 0.863 | 0.660 | 0.909 | [1] |
| C5 | symbolic | 0.891 | 0.682 | 1.000 | [1] |
| C5 | definition | 0.887 | 0.657 | — | [1] |
| C5 | roleless:own | 0.509 | 0.503 | 0.910 | [1] |
| P0 | none | 0.604 | 0.557 | — | [1] |
| P0 | recall:C5@1 | 0.842 | 0.653 | 0.909 | [1] |
| P0 | symbolic | 0.873 | 0.666 | 1.000 | [1] |
| P0 | definition | 0.897 | 0.659 | — | [1] |
| P0 | roleless:C5@1 | 0.555 | 0.530 | 0.910 | [1] |

### WP-UB two-hop and reverse items

| host model | condition | family / subset | accuracy | − chance | hop 1 | bridge | hop 2 | pair | seeds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0p | none | reverse/all | 0.454 | -0.046 | — | — | — | — | 1 |
| C0p | none | reverse/heldout | 0.322 | -0.178 | — | — | — | — | 1 |
| C0p | none | reverse/new | 0.428 | -0.072 | — | — | — | — | 1 |
| C0p | none | reverse/rare | 0.545 | +0.045 | — | — | — | — | 1 |
| C0p | none | reverse/seen | 0.532 | +0.032 | — | — | — | — | 1 |
| C0p | none | two_hop/all | 0.123 | -0.077 | — | — | — | — | 1 |
| C0p | none | two_hop/heldout | 0.179 | -0.021 | — | — | — | — | 1 |
| C0p | none | two_hop/new | 0.091 | -0.109 | — | — | — | — | 1 |
| C0p | none | two_hop/rare | 0.095 | -0.105 | — | — | — | — | 1 |
| C0p | none | two_hop/seen | 0.113 | -0.087 | — | — | — | — | 1 |
| C0p | recall:C5 | reverse/all | 0.484 | -0.016 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/heldout | 0.480 | -0.020 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/new | 0.524 | +0.024 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/rare | 0.455 | -0.045 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/seen | 0.478 | -0.022 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | two_hop/all | 0.637 | +0.437 | 0.804 | 0.784 | 0.549 | — | 1 |
| C0p | recall:C5 | two_hop/heldout | 0.554 | +0.354 | 0.679 | 0.643 | 0.500 | — | 1 |
| C0p | recall:C5 | two_hop/new | 0.591 | +0.391 | 0.864 | 0.818 | 0.591 | — | 1 |
| C0p | recall:C5 | two_hop/rare | 0.714 | +0.514 | 0.905 | 0.905 | 0.571 | — | 1 |
| C0p | recall:C5 | two_hop/seen | 0.694 | +0.494 | 0.806 | 0.806 | 0.548 | — | 1 |
| C0p | symbolic | reverse/all | 0.652 | +0.152 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/heldout | 0.680 | +0.180 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/new | 0.637 | +0.137 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/rare | 0.625 | +0.125 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/seen | 0.660 | +0.160 | — | — | — | 1.000 | 1 |
| C0p | symbolic | two_hop/all | 0.922 | +0.722 | 1.000 | 0.951 | 1.000 | — | 1 |
| C0p | symbolic | two_hop/heldout | 0.821 | +0.621 | 1.000 | 0.857 | 1.000 | — | 1 |
| C0p | symbolic | two_hop/new | 0.886 | +0.686 | 1.000 | 0.955 | 1.000 | — | 1 |
| C0p | symbolic | two_hop/rare | 1.000 | +0.800 | 1.000 | 1.000 | 1.000 | — | 1 |
| C0p | symbolic | two_hop/seen | 0.984 | +0.784 | 1.000 | 1.000 | 1.000 | — | 1 |
| C5 | none | reverse/all | 0.451 | -0.049 | — | — | — | — | 1 |
| C5 | none | reverse/heldout | 0.324 | -0.176 | — | — | — | — | 1 |
| C5 | none | reverse/new | 0.428 | -0.072 | — | — | — | — | 1 |
| C5 | none | reverse/rare | 0.541 | +0.041 | — | — | — | — | 1 |
| C5 | none | reverse/seen | 0.520 | +0.020 | — | — | — | — | 1 |
| C5 | none | two_hop/all | 0.098 | -0.102 | — | — | — | — | 1 |
| C5 | none | two_hop/heldout | 0.161 | -0.039 | — | — | — | — | 1 |
| C5 | none | two_hop/new | 0.091 | -0.109 | — | — | — | — | 1 |
| C5 | none | two_hop/rare | 0.048 | -0.152 | — | — | — | — | 1 |
| C5 | none | two_hop/seen | 0.081 | -0.119 | — | — | — | — | 1 |
| C5 | recall:own | reverse/all | 0.476 | -0.024 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/heldout | 0.474 | -0.026 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/new | 0.522 | +0.022 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/rare | 0.448 | -0.052 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/seen | 0.462 | -0.038 | — | — | — | 1.000 | 1 |
| C5 | recall:own | two_hop/all | 0.632 | +0.432 | 0.804 | 0.784 | 0.549 | — | 1 |
| C5 | recall:own | two_hop/heldout | 0.536 | +0.336 | 0.679 | 0.643 | 0.500 | — | 1 |
| C5 | recall:own | two_hop/new | 0.591 | +0.391 | 0.864 | 0.818 | 0.591 | — | 1 |
| C5 | recall:own | two_hop/rare | 0.738 | +0.538 | 0.905 | 0.905 | 0.571 | — | 1 |
| C5 | recall:own | two_hop/seen | 0.677 | +0.477 | 0.806 | 0.806 | 0.548 | — | 1 |
| C5 | symbolic | reverse/all | 0.651 | +0.151 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/heldout | 0.664 | +0.164 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/new | 0.648 | +0.148 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/rare | 0.643 | +0.143 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/seen | 0.646 | +0.146 | — | — | — | 1.000 | 1 |
| C5 | symbolic | two_hop/all | 0.926 | +0.726 | 1.000 | 0.951 | 1.000 | — | 1 |
| C5 | symbolic | two_hop/heldout | 0.821 | +0.621 | 1.000 | 0.857 | 1.000 | — | 1 |
| C5 | symbolic | two_hop/new | 0.909 | +0.709 | 1.000 | 0.955 | 1.000 | — | 1 |
| C5 | symbolic | two_hop/rare | 1.000 | +0.800 | 1.000 | 1.000 | 1.000 | — | 1 |
| C5 | symbolic | two_hop/seen | 0.984 | +0.784 | 1.000 | 1.000 | 1.000 | — | 1 |
| P0 | none | reverse/all | 0.474 | -0.026 | — | — | — | — | 1 |
| P0 | none | reverse/heldout | 0.383 | -0.117 | — | — | — | — | 1 |
| P0 | none | reverse/new | 0.463 | -0.037 | — | — | — | — | 1 |
| P0 | none | reverse/rare | 0.493 | -0.007 | — | — | — | — | 1 |
| P0 | none | reverse/seen | 0.558 | +0.058 | — | — | — | — | 1 |
| P0 | none | two_hop/all | 0.118 | -0.082 | — | — | — | — | 1 |
| P0 | none | two_hop/heldout | 0.143 | -0.057 | — | — | — | — | 1 |
| P0 | none | two_hop/new | 0.045 | -0.155 | — | — | — | — | 1 |
| P0 | none | two_hop/rare | 0.119 | -0.081 | — | — | — | — | 1 |
| P0 | none | two_hop/seen | 0.145 | -0.055 | — | — | — | — | 1 |
| P0 | recall:C5@1 | reverse/all | 0.486 | -0.014 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | reverse/heldout | 0.498 | -0.002 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | reverse/new | 0.543 | +0.043 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | reverse/rare | 0.450 | -0.050 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | reverse/seen | 0.452 | -0.048 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | two_hop/all | 0.554 | +0.354 | 0.804 | 0.784 | 0.549 | — | 1 |
| P0 | recall:C5@1 | two_hop/heldout | 0.482 | +0.282 | 0.679 | 0.643 | 0.500 | — | 1 |
| P0 | recall:C5@1 | two_hop/new | 0.523 | +0.323 | 0.864 | 0.818 | 0.591 | — | 1 |
| P0 | recall:C5@1 | two_hop/rare | 0.619 | +0.419 | 0.905 | 0.905 | 0.571 | — | 1 |
| P0 | recall:C5@1 | two_hop/seen | 0.597 | +0.397 | 0.806 | 0.806 | 0.548 | — | 1 |
| P0 | symbolic | reverse/all | 0.673 | +0.173 | — | — | — | 1.000 | 1 |
| P0 | symbolic | reverse/heldout | 0.686 | +0.186 | — | — | — | 1.000 | 1 |
| P0 | symbolic | reverse/new | 0.687 | +0.187 | — | — | — | 1.000 | 1 |
| P0 | symbolic | reverse/rare | 0.673 | +0.173 | — | — | — | 1.000 | 1 |
| P0 | symbolic | reverse/seen | 0.646 | +0.146 | — | — | — | 1.000 | 1 |
| P0 | symbolic | two_hop/all | 0.848 | +0.648 | 1.000 | 0.951 | 1.000 | — | 1 |
| P0 | symbolic | two_hop/heldout | 0.768 | +0.568 | 1.000 | 0.857 | 1.000 | — | 1 |
| P0 | symbolic | two_hop/new | 0.773 | +0.573 | 1.000 | 0.955 | 1.000 | — | 1 |
| P0 | symbolic | two_hop/rare | 0.952 | +0.752 | 1.000 | 1.000 | 1.000 | — | 1 |
| P0 | symbolic | two_hop/seen | 0.903 | +0.703 | 1.000 | 1.000 | 1.000 | — | 1 |

## SmolLM2-360M

Runs: {'P0': [1], 'C0p': [1, 2, 3], 'C2': [1, 2, 3], 'C5': [1, 2, 3], 'C5rf': [1, 2, 3], 'C5sh': [1, 2, 3], 'C5tr': [1, 2, 3], 'C5ut': [1, 2, 3], 'C6d': [1, 2, 3], 'C6g': [1, 2, 3], 'C6m': [1, 2, 3]}

### Natural role items (choice)

| host model | condition | contrast | item | decode | seeds |
|---|---|---:|---:|---:|---|
| C0p | none | 0.546 | 0.513 | — | [1, 2, 3] |
| C0p | recall:C5 | 0.877 | 0.655 | 0.908 | [1, 2, 3] |
| C0p | symbolic | 0.894 | 0.681 | 1.000 | [1, 2, 3] |
| C0p | definition | 0.913 | 0.682 | — | [1] |
| C0p | roleless:C5 | 0.471 | 0.486 | 0.909 | [1, 2, 3] |
| C2 | none | 0.547 | 0.516 | — | [1, 2, 3] |
| C2 | recall:C5 | 0.879 | 0.650 | 0.908 | [1, 2, 3] |
| C2 | symbolic | 0.891 | 0.676 | 1.000 | [1, 2, 3] |
| C2 | definition | 0.917 | 0.681 | — | [1] |
| C2 | roleless:C5 | 0.472 | 0.487 | 0.909 | [1, 2, 3] |
| C5 | none | 0.554 | 0.520 | — | [1, 2, 3] |
| C5 | recall:own | 0.882 | 0.645 | 0.908 | [1, 2, 3] |
| C5 | symbolic | 0.897 | 0.671 | 1.000 | [1, 2, 3] |
| C5 | definition | 0.915 | 0.682 | — | [1, 2, 3] |
| C5 | roleless:own | 0.479 | 0.489 | 0.909 | [1, 2, 3] |
| C5 | recall:C5ut | 0.500 | 0.491 | 0.882 | [1, 2, 3] |
| C5 | recall:C5tr | 0.511 | 0.500 | 0.625 | [1, 2, 3] |
| C5rf | none | 0.552 | 0.517 | — | [1, 2, 3] |
| C5rf | recall:own | 0.881 | 0.650 | 0.960 | [1, 2, 3] |
| C5rf | symbolic | 0.901 | 0.674 | 1.000 | [1, 2, 3] |
| C5rf | definition | 0.917 | 0.679 | — | [1] |
| C5rf | roleless:own | 0.485 | 0.493 | 0.960 | [1, 2, 3] |
| C5sh | none | 0.543 | 0.514 | — | [1, 2, 3] |
| C5sh | recall:own | 0.559 | 0.525 | 0.439 | [1, 2, 3] |
| C5sh | symbolic | 0.898 | 0.677 | 1.000 | [1, 2, 3] |
| C5sh | roleless:own | 0.552 | 0.519 | 0.461 | [1, 2, 3] |
| C5tr | none | 0.551 | 0.522 | — | [1, 2, 3] |
| C5tr | recall:own | 0.502 | 0.498 | 0.625 | [1, 2, 3] |
| C5tr | symbolic | 0.896 | 0.670 | 1.000 | [1, 2, 3] |
| C5tr | roleless:own | 0.538 | 0.506 | 0.670 | [1, 2, 3] |
| C5ut | none | 0.547 | 0.517 | — | [1, 2, 3] |
| C5ut | recall:own | 0.497 | 0.493 | 0.882 | [1, 2, 3] |
| C5ut | symbolic | 0.901 | 0.671 | 1.000 | [1, 2, 3] |
| C5ut | definition | 0.914 | 0.682 | — | [1, 2, 3] |
| C5ut | roleless:own | 0.478 | 0.492 | 0.882 | [1, 2, 3] |
| C5ut | recall:C5 | 0.882 | 0.646 | 0.908 | [1, 2, 3] |
| C6d | none | 0.558 | 0.521 | — | [1, 2, 3] |
| C6d | recall:C5 | 0.879 | 0.652 | 0.908 | [1, 2, 3] |
| C6d | symbolic | 0.896 | 0.680 | 1.000 | [1, 2, 3] |
| C6d | roleless:C5 | 0.486 | 0.487 | 0.909 | [1, 2, 3] |
| C6g | none | 0.546 | 0.512 | — | [1, 2, 3] |
| C6g | recall:C5 | 0.879 | 0.644 | 0.908 | [1, 2, 3] |
| C6g | symbolic | 0.897 | 0.669 | 1.000 | [1, 2, 3] |
| C6g | roleless:C5 | 0.481 | 0.490 | 0.909 | [1, 2, 3] |
| C6m | none | 0.550 | 0.518 | — | [1, 2, 3] |
| C6m | recall:C5 | 0.877 | 0.645 | 0.908 | [1, 2, 3] |
| C6m | symbolic | 0.896 | 0.673 | 1.000 | [1, 2, 3] |
| C6m | roleless:C5 | 0.475 | 0.480 | 0.909 | [1, 2, 3] |
| P0 | none | 0.524 | 0.514 | — | [1] |
| P0 | recall:C5@1 | 0.857 | 0.605 | 0.921 | [1] |
| P0 | symbolic | 0.887 | 0.624 | 1.000 | [1] |
| P0 | definition | 0.822 | 0.604 | — | [1] |
| P0 | roleless:C5@1 | 0.430 | 0.483 | 0.922 | [1] |

### Natural role items (cloze)

| host model | condition | contrast | item | decode | seeds |
|---|---|---:|---:|---:|---|
| C0p | none | 0.541 | 0.502 | — | [1, 2, 3] |
| C0p | recall:C5 | 0.811 | 0.644 | 0.908 | [1, 2, 3] |
| C0p | symbolic | 0.848 | 0.656 | 1.000 | [1, 2, 3] |
| C0p | definition | 0.867 | 0.659 | — | [1] |
| C0p | roleless:C5 | 0.525 | 0.505 | 0.909 | [1, 2, 3] |
| C2 | none | 0.542 | 0.505 | — | [1, 2, 3] |
| C2 | recall:C5 | 0.800 | 0.643 | 0.908 | [1, 2, 3] |
| C2 | symbolic | 0.830 | 0.655 | 1.000 | [1, 2, 3] |
| C2 | definition | 0.862 | 0.659 | — | [1] |
| C2 | roleless:C5 | 0.524 | 0.504 | 0.909 | [1, 2, 3] |
| C5 | none | 0.540 | 0.507 | — | [1, 2, 3] |
| C5 | recall:own | 0.794 | 0.634 | 0.908 | [1, 2, 3] |
| C5 | symbolic | 0.822 | 0.648 | 1.000 | [1, 2, 3] |
| C5 | definition | 0.856 | 0.651 | — | [1, 2, 3] |
| C5 | roleless:own | 0.526 | 0.507 | 0.909 | [1, 2, 3] |
| C5 | recall:C5ut | 0.518 | 0.500 | 0.882 | [1, 2, 3] |
| C5 | recall:C5tr | 0.502 | 0.504 | 0.625 | [1, 2, 3] |
| C5rf | none | 0.535 | 0.505 | — | [1, 2, 3] |
| C5rf | recall:own | 0.796 | 0.637 | 0.960 | [1, 2, 3] |
| C5rf | symbolic | 0.821 | 0.653 | 1.000 | [1, 2, 3] |
| C5rf | definition | 0.863 | 0.658 | — | [1] |
| C5rf | roleless:own | 0.533 | 0.514 | 0.960 | [1, 2, 3] |
| C5sh | none | 0.545 | 0.501 | — | [1, 2, 3] |
| C5sh | recall:own | 0.552 | 0.516 | 0.439 | [1, 2, 3] |
| C5sh | symbolic | 0.838 | 0.658 | 1.000 | [1, 2, 3] |
| C5sh | roleless:own | 0.572 | 0.520 | 0.461 | [1, 2, 3] |
| C5tr | none | 0.546 | 0.505 | — | [1, 2, 3] |
| C5tr | recall:own | 0.505 | 0.505 | 0.625 | [1, 2, 3] |
| C5tr | symbolic | 0.820 | 0.652 | 1.000 | [1, 2, 3] |
| C5tr | roleless:own | 0.566 | 0.521 | 0.670 | [1, 2, 3] |
| C5ut | none | 0.535 | 0.505 | — | [1, 2, 3] |
| C5ut | recall:own | 0.515 | 0.501 | 0.882 | [1, 2, 3] |
| C5ut | symbolic | 0.820 | 0.650 | 1.000 | [1, 2, 3] |
| C5ut | definition | 0.860 | 0.654 | — | [1, 2, 3] |
| C5ut | roleless:own | 0.522 | 0.509 | 0.882 | [1, 2, 3] |
| C5ut | recall:C5 | 0.793 | 0.632 | 0.908 | [1, 2, 3] |
| C6d | none | 0.546 | 0.512 | — | [1, 2, 3] |
| C6d | recall:C5 | 0.791 | 0.635 | 0.908 | [1, 2, 3] |
| C6d | symbolic | 0.822 | 0.654 | 1.000 | [1, 2, 3] |
| C6d | roleless:C5 | 0.522 | 0.506 | 0.909 | [1, 2, 3] |
| C6g | none | 0.532 | 0.506 | — | [1, 2, 3] |
| C6g | recall:C5 | 0.798 | 0.638 | 0.908 | [1, 2, 3] |
| C6g | symbolic | 0.832 | 0.658 | 1.000 | [1, 2, 3] |
| C6g | roleless:C5 | 0.529 | 0.508 | 0.909 | [1, 2, 3] |
| C6m | none | 0.536 | 0.505 | — | [1, 2, 3] |
| C6m | recall:C5 | 0.793 | 0.638 | 0.908 | [1, 2, 3] |
| C6m | symbolic | 0.832 | 0.656 | 1.000 | [1, 2, 3] |
| C6m | roleless:C5 | 0.522 | 0.509 | 0.909 | [1, 2, 3] |
| P0 | none | 0.576 | 0.526 | — | [1] |
| P0 | recall:C5@1 | 0.754 | 0.601 | 0.921 | [1] |
| P0 | symbolic | 0.775 | 0.611 | 1.000 | [1] |
| P0 | definition | 0.820 | 0.631 | — | [1] |
| P0 | roleless:C5@1 | 0.491 | 0.511 | 0.922 | [1] |

### WP-UB two-hop and reverse items

| host model | condition | family / subset | accuracy | − chance | hop 1 | bridge | hop 2 | pair | seeds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0p | none | reverse/all | 0.556 | +0.056 | — | — | — | — | 1 |
| C0p | none | reverse/heldout | 0.530 | +0.030 | — | — | — | — | 1 |
| C0p | none | reverse/new | 0.504 | +0.004 | — | — | — | — | 1 |
| C0p | none | reverse/rare | 0.625 | +0.125 | — | — | — | — | 1 |
| C0p | none | reverse/seen | 0.568 | +0.068 | — | — | — | — | 1 |
| C0p | none | two_hop/all | 0.196 | -0.004 | — | — | — | — | 1 |
| C0p | none | two_hop/heldout | 0.232 | +0.032 | — | — | — | — | 1 |
| C0p | none | two_hop/new | 0.136 | -0.064 | — | — | — | — | 1 |
| C0p | none | two_hop/rare | 0.214 | +0.014 | — | — | — | — | 1 |
| C0p | none | two_hop/seen | 0.194 | -0.006 | — | — | — | — | 1 |
| C0p | recall:C5 | reverse/all | 0.483 | -0.017 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/heldout | 0.480 | -0.020 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/new | 0.457 | -0.043 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/rare | 0.470 | -0.030 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/seen | 0.522 | +0.022 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | two_hop/all | 0.623 | +0.423 | 0.853 | 0.833 | 0.588 | — | 1 |
| C0p | recall:C5 | two_hop/heldout | 0.607 | +0.407 | 0.786 | 0.750 | 0.536 | — | 1 |
| C0p | recall:C5 | two_hop/new | 0.591 | +0.391 | 0.818 | 0.773 | 0.591 | — | 1 |
| C0p | recall:C5 | two_hop/rare | 0.595 | +0.395 | 0.905 | 0.905 | 0.524 | — | 1 |
| C0p | recall:C5 | two_hop/seen | 0.677 | +0.477 | 0.903 | 0.903 | 0.677 | — | 1 |
| C0p | symbolic | reverse/all | 0.772 | +0.272 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/heldout | 0.759 | +0.259 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/new | 0.765 | +0.265 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/rare | 0.755 | +0.255 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/seen | 0.806 | +0.306 | — | — | — | 1.000 | 1 |
| C0p | symbolic | two_hop/all | 0.853 | +0.653 | 1.000 | 0.951 | 1.000 | — | 1 |
| C0p | symbolic | two_hop/heldout | 0.821 | +0.621 | 1.000 | 0.857 | 1.000 | — | 1 |
| C0p | symbolic | two_hop/new | 0.841 | +0.641 | 1.000 | 0.955 | 1.000 | — | 1 |
| C0p | symbolic | two_hop/rare | 0.881 | +0.681 | 1.000 | 1.000 | 1.000 | — | 1 |
| C0p | symbolic | two_hop/seen | 0.871 | +0.671 | 1.000 | 1.000 | 1.000 | — | 1 |
| C5 | none | reverse/all | 0.554 | +0.054 | — | — | — | — | 1 |
| C5 | none | reverse/heldout | 0.530 | +0.030 | — | — | — | — | 1 |
| C5 | none | reverse/new | 0.509 | +0.009 | — | — | — | — | 1 |
| C5 | none | reverse/rare | 0.609 | +0.109 | — | — | — | — | 1 |
| C5 | none | reverse/seen | 0.572 | +0.072 | — | — | — | — | 1 |
| C5 | none | two_hop/all | 0.230 | +0.030 | — | — | — | — | 1 |
| C5 | none | two_hop/heldout | 0.268 | +0.068 | — | — | — | — | 1 |
| C5 | none | two_hop/new | 0.136 | -0.064 | — | — | — | — | 1 |
| C5 | none | two_hop/rare | 0.286 | +0.086 | — | — | — | — | 1 |
| C5 | none | two_hop/seen | 0.226 | +0.026 | — | — | — | — | 1 |
| C5 | recall:C5tr | reverse/all | 0.429 | -0.071 | — | — | — | 0.900 | 1 |
| C5 | recall:C5tr | reverse/heldout | 0.415 | -0.085 | — | — | — | 0.941 | 1 |
| C5 | recall:C5tr | reverse/new | 0.400 | -0.100 | — | — | — | 0.900 | 1 |
| C5 | recall:C5tr | reverse/rare | 0.436 | -0.064 | — | — | — | 0.923 | 1 |
| C5 | recall:C5tr | reverse/seen | 0.462 | -0.038 | — | — | — | 0.840 | 1 |
| C5 | recall:C5tr | two_hop/all | 0.324 | +0.124 | 0.402 | 0.373 | 0.196 | — | 1 |
| C5 | recall:C5tr | two_hop/heldout | 0.411 | +0.211 | 0.321 | 0.250 | 0.179 | — | 1 |
| C5 | recall:C5tr | two_hop/new | 0.432 | +0.232 | 0.682 | 0.636 | 0.364 | — | 1 |
| C5 | recall:C5tr | two_hop/rare | 0.214 | +0.014 | 0.429 | 0.429 | 0.143 | — | 1 |
| C5 | recall:C5tr | two_hop/seen | 0.242 | +0.042 | 0.258 | 0.258 | 0.129 | — | 1 |
| C5 | recall:C5ut | reverse/all | 0.471 | -0.029 | — | — | — | 1.000 | 1 |
| C5 | recall:C5ut | reverse/heldout | 0.472 | -0.028 | — | — | — | 1.000 | 1 |
| C5 | recall:C5ut | reverse/new | 0.426 | -0.074 | — | — | — | 1.000 | 1 |
| C5 | recall:C5ut | reverse/rare | 0.486 | -0.014 | — | — | — | 1.000 | 1 |
| C5 | recall:C5ut | reverse/seen | 0.498 | -0.002 | — | — | — | 1.000 | 1 |
| C5 | recall:C5ut | two_hop/all | 0.358 | +0.158 | 0.431 | 0.402 | 0.265 | — | 1 |
| C5 | recall:C5ut | two_hop/heldout | 0.375 | +0.175 | 0.357 | 0.286 | 0.179 | — | 1 |
| C5 | recall:C5ut | two_hop/new | 0.386 | +0.186 | 0.682 | 0.636 | 0.409 | — | 1 |
| C5 | recall:C5ut | two_hop/rare | 0.405 | +0.205 | 0.524 | 0.524 | 0.333 | — | 1 |
| C5 | recall:C5ut | two_hop/seen | 0.290 | +0.090 | 0.258 | 0.258 | 0.194 | — | 1 |
| C5 | recall:own | reverse/all | 0.477 | -0.023 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/heldout | 0.478 | -0.022 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/new | 0.457 | -0.043 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/rare | 0.470 | -0.030 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/seen | 0.500 | +0.000 | — | — | — | 1.000 | 1 |
| C5 | recall:own | two_hop/all | 0.623 | +0.423 | 0.853 | 0.833 | 0.588 | — | 1 |
| C5 | recall:own | two_hop/heldout | 0.607 | +0.407 | 0.786 | 0.750 | 0.536 | — | 1 |
| C5 | recall:own | two_hop/new | 0.568 | +0.368 | 0.818 | 0.773 | 0.591 | — | 1 |
| C5 | recall:own | two_hop/rare | 0.595 | +0.395 | 0.905 | 0.905 | 0.524 | — | 1 |
| C5 | recall:own | two_hop/seen | 0.694 | +0.494 | 0.903 | 0.903 | 0.677 | — | 1 |
| C5 | symbolic | reverse/all | 0.766 | +0.266 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/heldout | 0.755 | +0.255 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/new | 0.761 | +0.261 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/rare | 0.739 | +0.239 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/seen | 0.806 | +0.306 | — | — | — | 1.000 | 1 |
| C5 | symbolic | two_hop/all | 0.843 | +0.643 | 1.000 | 0.951 | 1.000 | — | 1 |
| C5 | symbolic | two_hop/heldout | 0.821 | +0.621 | 1.000 | 0.857 | 1.000 | — | 1 |
| C5 | symbolic | two_hop/new | 0.864 | +0.664 | 1.000 | 0.955 | 1.000 | — | 1 |
| C5 | symbolic | two_hop/rare | 0.833 | +0.633 | 1.000 | 1.000 | 1.000 | — | 1 |
| C5 | symbolic | two_hop/seen | 0.855 | +0.655 | 1.000 | 1.000 | 1.000 | — | 1 |
| C5ut | none | reverse/all | 0.553 | +0.053 | — | — | — | — | 1 |
| C5ut | none | reverse/heldout | 0.534 | +0.034 | — | — | — | — | 1 |
| C5ut | none | reverse/new | 0.504 | +0.004 | — | — | — | — | 1 |
| C5ut | none | reverse/rare | 0.627 | +0.127 | — | — | — | — | 1 |
| C5ut | none | reverse/seen | 0.552 | +0.052 | — | — | — | — | 1 |
| C5ut | none | two_hop/all | 0.186 | -0.014 | — | — | — | — | 1 |
| C5ut | none | two_hop/heldout | 0.214 | +0.014 | — | — | — | — | 1 |
| C5ut | none | two_hop/new | 0.136 | -0.064 | — | — | — | — | 1 |
| C5ut | none | two_hop/rare | 0.190 | -0.010 | — | — | — | — | 1 |
| C5ut | none | two_hop/seen | 0.194 | -0.006 | — | — | — | — | 1 |
| C5ut | recall:C5 | reverse/all | 0.476 | -0.024 | — | — | — | 1.000 | 1 |
| C5ut | recall:C5 | reverse/heldout | 0.476 | -0.024 | — | — | — | 1.000 | 1 |
| C5ut | recall:C5 | reverse/new | 0.450 | -0.050 | — | — | — | 1.000 | 1 |
| C5ut | recall:C5 | reverse/rare | 0.468 | -0.032 | — | — | — | 1.000 | 1 |
| C5ut | recall:C5 | reverse/seen | 0.506 | +0.006 | — | — | — | 1.000 | 1 |
| C5ut | recall:C5 | two_hop/all | 0.613 | +0.413 | 0.853 | 0.833 | 0.588 | — | 1 |
| C5ut | recall:C5 | two_hop/heldout | 0.607 | +0.407 | 0.786 | 0.750 | 0.536 | — | 1 |
| C5ut | recall:C5 | two_hop/new | 0.500 | +0.300 | 0.818 | 0.773 | 0.591 | — | 1 |
| C5ut | recall:C5 | two_hop/rare | 0.595 | +0.395 | 0.905 | 0.905 | 0.524 | — | 1 |
| C5ut | recall:C5 | two_hop/seen | 0.710 | +0.510 | 0.903 | 0.903 | 0.677 | — | 1 |
| C5ut | recall:own | reverse/all | 0.466 | -0.034 | — | — | — | 1.000 | 1 |
| C5ut | recall:own | reverse/heldout | 0.486 | -0.014 | — | — | — | 1.000 | 1 |
| C5ut | recall:own | reverse/new | 0.409 | -0.091 | — | — | — | 1.000 | 1 |
| C5ut | recall:own | reverse/rare | 0.464 | -0.036 | — | — | — | 1.000 | 1 |
| C5ut | recall:own | reverse/seen | 0.502 | +0.002 | — | — | — | 1.000 | 1 |
| C5ut | recall:own | two_hop/all | 0.358 | +0.158 | 0.431 | 0.402 | 0.265 | — | 1 |
| C5ut | recall:own | two_hop/heldout | 0.375 | +0.175 | 0.357 | 0.286 | 0.179 | — | 1 |
| C5ut | recall:own | two_hop/new | 0.341 | +0.141 | 0.682 | 0.636 | 0.409 | — | 1 |
| C5ut | recall:own | two_hop/rare | 0.381 | +0.181 | 0.524 | 0.524 | 0.333 | — | 1 |
| C5ut | recall:own | two_hop/seen | 0.339 | +0.139 | 0.258 | 0.258 | 0.194 | — | 1 |
| C5ut | symbolic | reverse/all | 0.770 | +0.270 | — | — | — | 1.000 | 1 |
| C5ut | symbolic | reverse/heldout | 0.757 | +0.257 | — | — | — | 1.000 | 1 |
| C5ut | symbolic | reverse/new | 0.774 | +0.274 | — | — | — | 1.000 | 1 |
| C5ut | symbolic | reverse/rare | 0.755 | +0.255 | — | — | — | 1.000 | 1 |
| C5ut | symbolic | reverse/seen | 0.794 | +0.294 | — | — | — | 1.000 | 1 |
| C5ut | symbolic | two_hop/all | 0.848 | +0.648 | 1.000 | 0.951 | 1.000 | — | 1 |
| C5ut | symbolic | two_hop/heldout | 0.821 | +0.621 | 1.000 | 0.857 | 1.000 | — | 1 |
| C5ut | symbolic | two_hop/new | 0.864 | +0.664 | 1.000 | 0.955 | 1.000 | — | 1 |
| C5ut | symbolic | two_hop/rare | 0.857 | +0.657 | 1.000 | 1.000 | 1.000 | — | 1 |
| C5ut | symbolic | two_hop/seen | 0.855 | +0.655 | 1.000 | 1.000 | 1.000 | — | 1 |
| P0 | none | reverse/all | 0.566 | +0.066 | — | — | — | — | 1 |
| P0 | none | reverse/heldout | 0.571 | +0.071 | — | — | — | — | 1 |
| P0 | none | reverse/new | 0.483 | -0.017 | — | — | — | — | 1 |
| P0 | none | reverse/rare | 0.614 | +0.114 | — | — | — | — | 1 |
| P0 | none | reverse/seen | 0.596 | +0.096 | — | — | — | — | 1 |
| P0 | none | two_hop/all | 0.147 | -0.053 | — | — | — | — | 1 |
| P0 | none | two_hop/heldout | 0.232 | +0.032 | — | — | — | — | 1 |
| P0 | none | two_hop/new | 0.045 | -0.155 | — | — | — | — | 1 |
| P0 | none | two_hop/rare | 0.071 | -0.129 | — | — | — | — | 1 |
| P0 | none | two_hop/seen | 0.194 | -0.006 | — | — | — | — | 1 |
| P0 | recall:C5@1 | reverse/all | 0.497 | -0.003 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | reverse/heldout | 0.482 | -0.018 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | reverse/new | 0.483 | -0.017 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | reverse/rare | 0.509 | +0.009 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | reverse/seen | 0.514 | +0.014 | — | — | — | 1.000 | 1 |
| P0 | recall:C5@1 | two_hop/all | 0.652 | +0.452 | 0.853 | 0.833 | 0.588 | — | 1 |
| P0 | recall:C5@1 | two_hop/heldout | 0.643 | +0.443 | 0.786 | 0.750 | 0.536 | — | 1 |
| P0 | recall:C5@1 | two_hop/new | 0.523 | +0.323 | 0.818 | 0.773 | 0.591 | — | 1 |
| P0 | recall:C5@1 | two_hop/rare | 0.714 | +0.514 | 0.905 | 0.905 | 0.524 | — | 1 |
| P0 | recall:C5@1 | two_hop/seen | 0.710 | +0.510 | 0.903 | 0.903 | 0.677 | — | 1 |
| P0 | symbolic | reverse/all | 0.743 | +0.243 | — | — | — | 1.000 | 1 |
| P0 | symbolic | reverse/heldout | 0.753 | +0.253 | — | — | — | 1.000 | 1 |
| P0 | symbolic | reverse/new | 0.754 | +0.254 | — | — | — | 1.000 | 1 |
| P0 | symbolic | reverse/rare | 0.716 | +0.216 | — | — | — | 1.000 | 1 |
| P0 | symbolic | reverse/seen | 0.748 | +0.248 | — | — | — | 1.000 | 1 |
| P0 | symbolic | two_hop/all | 0.877 | +0.677 | 1.000 | 0.951 | 1.000 | — | 1 |
| P0 | symbolic | two_hop/heldout | 0.821 | +0.621 | 1.000 | 0.857 | 1.000 | — | 1 |
| P0 | symbolic | two_hop/new | 0.841 | +0.641 | 1.000 | 0.955 | 1.000 | — | 1 |
| P0 | symbolic | two_hop/rare | 0.881 | +0.681 | 1.000 | 1.000 | 1.000 | — | 1 |
| P0 | symbolic | two_hop/seen | 0.952 | +0.752 | 1.000 | 1.000 | 1.000 | — | 1 |

