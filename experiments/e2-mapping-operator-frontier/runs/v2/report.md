# E2 — mapping × operator frontier (frozen-host concept anchors)

## mesh

| Mapping | Operator | Val MRR | Test MRR | Test cosine | Test VE | Params |
|---|---|---:|---:|---:|---:|---:|
| induced_k4 | diagonal | 0.0979 | 0.0739 | 0.3242 | -1.4128 | 2297108 |
| induced_k16 | hrr | 0.0974 | 0.0753 | 0.3268 | -1.3987 | 2303312 |
| induced_k16 | hrr_identity | 0.0969 | 0.0743 | 0.3313 | -1.4807 | 2303312 |
| induced_k32 | random_fixed:hrr | 0.0968 | 0.0732 | 0.3220 | -1.3727 | 2310304 |
| hybrid_k8 | hrr | 0.0967 | 0.0716 | 0.3266 | -1.4004 | 2347176 |
| induced_k8 | hrr | 0.0966 | 0.0717 | 0.3265 | -1.4013 | 2299176 |
| induced_k16 | orthogonal | 0.0959 | 0.0717 | 0.3097 | -1.2193 | 2629712 |
| induced_k8 | random_fixed:hrr | 0.0953 | 0.0710 | 0.3220 | -1.3755 | 2297896 |
| induced_k16 | random_fixed:hrr | 0.0953 | 0.0747 | 0.3263 | -1.3782 | 2302032 |
| hybrid_k8 | hrr_identity | 0.0952 | 0.0735 | 0.3299 | -1.4790 | 2347176 |
| induced_k4 | untyped | 0.0951 | 0.0727 | 0.3227 | -1.4951 | 2295828 |
| hybrid_k8 | random_fixed:hrr | 0.0950 | 0.0716 | 0.3235 | -1.3747 | 2345896 |
| induced_k4 | hrr_identity | 0.0949 | 0.0730 | 0.3254 | -1.4991 | 2297108 |
| induced_k8 | hrr_identity | 0.0945 | 0.0735 | 0.3299 | -1.4765 | 2299176 |
| induced_k32 | diagonal | 0.0944 | 0.0715 | 0.3281 | -1.3754 | 2311584 |
| induced_k32 | untyped | 0.0940 | 0.0709 | 0.3230 | -1.4639 | 2310304 |
| induced_k4 | hrr | 0.0938 | 0.0738 | 0.3243 | -1.4057 | 2297108 |
| induced_k8 | orthogonal | 0.0937 | 0.0746 | 0.3099 | -1.2066 | 2625576 |
| hybrid_k8 | low_rank_tied | 0.0935 | 0.0730 | 0.3206 | -1.4727 | 2347181 |
| induced_k4 | orthogonal | 0.0935 | 0.0708 | 0.3107 | -1.2254 | 2623508 |
| induced_k32 | hrr | 0.0933 | 0.0752 | 0.3289 | -1.3904 | 2311584 |
| induced_k4 | random_fixed:hrr | 0.0928 | 0.0739 | 0.3158 | -1.3785 | 2295828 |
| induced_k4 | low_rank_tied | 0.0928 | 0.0720 | 0.3179 | -1.4911 | 2297113 |
| induced_k16 | low_rank_tied | 0.0926 | 0.0755 | 0.3268 | -1.4641 | 2303317 |
| induced_k8 | low_rank_tied | 0.0925 | 0.0735 | 0.3206 | -1.4747 | 2299181 |
| hybrid_k8 | orthogonal | 0.0923 | 0.0744 | 0.3113 | -1.2022 | 2673576 |
| binary | orthogonal | 0.0921 | 0.0717 | 0.3348 | -1.0880 | 2621440 |
| induced_k32 | orthogonal | 0.0920 | 0.0703 | 0.3116 | -1.2118 | 2637984 |
| induced_k8 | diagonal | 0.0919 | 0.0713 | 0.3216 | -1.3934 | 2299176 |
| hybrid_k8 | untyped | 0.0919 | 0.0737 | 0.3213 | -1.4841 | 2345896 |
| induced_k32 | hrr_identity | 0.0919 | 0.0755 | 0.3310 | -1.4698 | 2311584 |
| salience | orthogonal | 0.0914 | 0.0711 | 0.3236 | -1.3797 | 2621445 |
| binary | hrr | 0.0912 | 0.0703 | 0.3236 | -1.6103 | 2295040 |
| salience | hrr | 0.0911 | 0.0695 | 0.3220 | -1.5979 | 2295045 |
| hybrid_k8 | diagonal | 0.0906 | 0.0716 | 0.3232 | -1.3869 | 2347176 |
| salience | low_rank_tied | 0.0905 | 0.0672 | 0.3223 | -1.8398 | 2295050 |
| salience | hrr_identity | 0.0904 | 0.0682 | 0.3250 | -1.6580 | 2295045 |
| induced_k8 | untyped | 0.0903 | 0.0728 | 0.3208 | -1.4655 | 2297896 |
| induced_k16 | diagonal | 0.0903 | 0.0715 | 0.3253 | -1.3803 | 2303312 |
| binary | hrr_identity | 0.0902 | 0.0694 | 0.3270 | -1.6723 | 2295040 |
| induced_k32 | low_rank_tied | 0.0902 | 0.0720 | 0.3177 | -1.4675 | 2311589 |
| induced_k16 | untyped | 0.0901 | 0.0695 | 0.3215 | -1.4612 | 2302032 |
| salience | random_fixed:hrr | 0.0899 | 0.0699 | 0.3259 | -1.6455 | 2293765 |
| salience | untyped | 0.0892 | 0.0667 | 0.3200 | -1.9645 | 2293765 |
| binary | low_rank_tied | 0.0891 | 0.0667 | 0.3280 | -1.8200 | 2295045 |
| binary | diagonal | 0.0888 | 0.0676 | 0.3247 | -1.4157 | 2295040 |
| salience | diagonal | 0.0882 | 0.0673 | 0.3222 | -1.6415 | 2295045 |
| binary | random_fixed:hrr | 0.0874 | 0.0706 | 0.3345 | -1.6488 | 2293760 |
| binary | untyped | 0.0848 | 0.0671 | 0.3230 | -1.9438 | 2293760 |

Selected on validation: **induced_k4 / diagonal**; best induced rank for that operator: induced_k4|diagonal.
Induced − salience (test MRR): +0.0065 [-0.0021, +0.0151]; induced − binary: +0.0062 [-0.0032, +0.0156]; hybrid − induced k8: +0.0003 [-0.0017, +0.0023].
Selected − random_fixed operator: -0.0001 [-0.0054, +0.0052]; selected − untyped: +0.0011 [-0.0036, +0.0059].
True − shuffled relation labels (primary mapping): hrr: +0.0121 [-0.0060, +0.0302]; hrr_identity: +0.0057 [+0.0035, +0.0080]; diagonal: +0.0036 [-0.0042, +0.0115]; low_rank_tied: +0.0061 [+0.0043, +0.0078]; orthogonal: +0.0065 [-0.0007, +0.0137]; random_fixed:hrr: +0.0143 [+0.0079, +0.0208]; untyped: +0.0045 [+0.0030, +0.0059].
Gates: induced ≥ salience **True**; induced > binary **False**.

## wordnet

| Mapping | Operator | Val MRR | Test MRR | Test cosine | Test VE | Params |
|---|---|---:|---:|---:|---:|---:|
| binary | low_rank_tied | 0.0596 | 0.0342 | 0.1920 | -2.5927 | 2297872 |
| binary | hrr_identity | 0.0595 | 0.0350 | 0.1915 | -2.7351 | 2297856 |
| salience | hrr_identity | 0.0587 | 0.0355 | 0.1898 | -2.7079 | 2297872 |
| salience | low_rank_tied | 0.0580 | 0.0347 | 0.1889 | -2.9442 | 2297888 |
| salience | orthogonal | 0.0578 | 0.0394 | 0.1926 | -2.3185 | 3342352 |
| binary | orthogonal | 0.0567 | 0.0398 | 0.1978 | -1.6834 | 3342336 |
| binary | untyped | 0.0562 | 0.0346 | 0.1889 | -2.6181 | 2293760 |
| binary | diagonal | 0.0559 | 0.0347 | 0.1913 | -1.4274 | 2297856 |
| induced_k16 | hrr_identity | 0.0558 | 0.0352 | 0.1762 | -2.5256 | 2306304 |
| salience | diagonal | 0.0544 | 0.0351 | 0.1882 | -2.2598 | 2297872 |
| induced_k8 | hrr_identity | 0.0541 | 0.0354 | 0.1762 | -2.5188 | 2302080 |
| salience | untyped | 0.0539 | 0.0343 | 0.1890 | -2.9576 | 2293776 |
| binary | random_fixed:hrr | 0.0538 | 0.0356 | 0.1950 | -2.5780 | 2293760 |
| induced_k4 | hrr_identity | 0.0536 | 0.0362 | 0.1756 | -2.5020 | 2299968 |
| induced_k4 | hrr | 0.0536 | 0.0390 | 0.1758 | -2.4687 | 2299968 |
| induced_k16 | untyped | 0.0534 | 0.0341 | 0.1755 | -2.6712 | 2302208 |
| hybrid_k8 | hrr_identity | 0.0529 | 0.0358 | 0.1757 | -2.5305 | 2350080 |
| induced_k32 | hrr_identity | 0.0529 | 0.0364 | 0.1717 | -2.5170 | 2314752 |
| induced_k32 | orthogonal | 0.0528 | 0.0358 | 0.1675 | -2.3761 | 3359232 |
| induced_k16 | diagonal | 0.0527 | 0.0365 | 0.1770 | -2.4615 | 2306304 |
| induced_k4 | orthogonal | 0.0526 | 0.0353 | 0.1653 | -2.3936 | 3344448 |
| induced_k4 | low_rank_tied | 0.0524 | 0.0369 | 0.1756 | -2.7106 | 2299984 |
| induced_k8 | untyped | 0.0524 | 0.0349 | 0.1748 | -2.6898 | 2297984 |
| induced_k32 | diagonal | 0.0520 | 0.0379 | 0.1743 | -2.4665 | 2314752 |
| salience | random_fixed:hrr | 0.0520 | 0.0343 | 0.1890 | -2.6953 | 2293776 |
| induced_k8 | hrr | 0.0518 | 0.0337 | 0.1705 | -2.4659 | 2302080 |
| induced_k32 | untyped | 0.0518 | 0.0355 | 0.1746 | -2.6768 | 2310656 |
| induced_k16 | random_fixed:hrr | 0.0517 | 0.0333 | 0.1626 | -2.4796 | 2302208 |
| hybrid_k8 | untyped | 0.0512 | 0.0356 | 0.1776 | -2.7171 | 2345984 |
| hybrid_k8 | orthogonal | 0.0512 | 0.0365 | 0.1685 | -2.3928 | 3394560 |
| induced_k8 | diagonal | 0.0511 | 0.0358 | 0.1763 | -2.5055 | 2302080 |
| induced_k16 | orthogonal | 0.0507 | 0.0362 | 0.1678 | -2.3763 | 3350784 |
| induced_k4 | diagonal | 0.0507 | 0.0366 | 0.1777 | -2.5022 | 2299968 |
| binary | hrr | 0.0507 | 0.0354 | 0.1867 | -2.6626 | 2297856 |
| hybrid_k8 | hrr | 0.0506 | 0.0340 | 0.1717 | -2.4678 | 2350080 |
| induced_k32 | low_rank_tied | 0.0504 | 0.0368 | 0.1743 | -2.6960 | 2314768 |
| induced_k8 | orthogonal | 0.0503 | 0.0370 | 0.1664 | -2.3937 | 3346560 |
| induced_k8 | random_fixed:hrr | 0.0502 | 0.0337 | 0.1679 | -2.4512 | 2297984 |
| induced_k32 | random_fixed:hrr | 0.0498 | 0.0365 | 0.1683 | -2.4513 | 2310656 |
| hybrid_k8 | random_fixed:hrr | 0.0495 | 0.0334 | 0.1702 | -2.4558 | 2345984 |
| induced_k8 | low_rank_tied | 0.0494 | 0.0357 | 0.1805 | -2.7279 | 2302096 |
| induced_k16 | low_rank_tied | 0.0492 | 0.0353 | 0.1778 | -2.6931 | 2306320 |
| salience | hrr | 0.0491 | 0.0355 | 0.1847 | -2.6475 | 2297872 |
| induced_k4 | untyped | 0.0489 | 0.0358 | 0.1772 | -2.6947 | 2295872 |
| induced_k4 | random_fixed:hrr | 0.0487 | 0.0355 | 0.1686 | -2.4744 | 2295872 |
| induced_k16 | hrr | 0.0480 | 0.0355 | 0.1704 | -2.4493 | 2306304 |
| induced_k32 | hrr | 0.0477 | 0.0349 | 0.1720 | -2.4754 | 2314752 |
| hybrid_k8 | low_rank_tied | 0.0465 | 0.0368 | 0.1826 | -2.7488 | 2350096 |
| hybrid_k8 | diagonal | 0.0462 | 0.0360 | 0.1824 | -2.4453 | 2350080 |

Selected on validation: **binary / low_rank_tied**; best induced rank for that operator: induced_k4|low_rank_tied.
Induced − salience (test MRR): +0.0023 [-0.0057, +0.0102]; induced − binary: +0.0028 [-0.0084, +0.0140]; hybrid − induced k8: +0.0011 [-0.0068, +0.0090].
Selected − random_fixed operator: -0.0015 [-0.0094, +0.0064]; selected − untyped: -0.0005 [-0.0037, +0.0027].
True − shuffled relation labels (primary mapping): hrr: +0.0109 [+0.0042, +0.0176]; hrr_identity: +0.0073 [-0.0049, +0.0195]; diagonal: +0.0050 [-0.0109, +0.0209]; low_rank_tied: +0.0049 [-0.0039, +0.0137]; orthogonal: +0.0085 [-0.0058, +0.0228]; random_fixed:hrr: +0.0118 [-0.0026, +0.0262]; untyped: +0.0041 [-0.0126, +0.0209].
Gates: induced ≥ salience **True**; induced > binary **False**.
