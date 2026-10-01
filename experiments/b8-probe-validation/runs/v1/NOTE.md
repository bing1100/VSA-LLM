# Note

The INT4 column of this run failed (the default torchao INT4 layout needs the `mslk` package, which has no usable release) and the INT8 GPT-2 number quantized the tied output head. Quantization numbers are superseded by `../v1-ptq/` (tile-packed INT4 layout; output head kept in full precision). Probe numbers in this run are valid.
