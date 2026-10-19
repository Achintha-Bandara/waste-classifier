| Metric | Model A (Standard) | Model B (Depthwise Separable) |
|---|---|---|
| Trainable parameters | 1,143,305 | 29,316 |
| Total parameters | 1,143,689 | 29,700 |
| MACs per inference | 42,337,408 | 5,158,016 |
| Float32 weight size (KB) | 4467.5 | 116.0 |
| Model file on disk (KB) | 13462.6 | 411.6 |
| Training time / epoch (s) | 2.34 | 2.59 |
| Test accuracy | 0.6844 | 0.5835 |
| Test precision (macro) | 0.6876 | 0.5749 |
| Test recall (macro) | 0.6739 | 0.5500 |
| Test F1 (macro) | 0.6785 | 0.5444 |

Parameter reduction: 39.0x | MAC reduction: 8.2x