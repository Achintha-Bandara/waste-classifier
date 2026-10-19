# Saved Model Comparison

Each column is one saved model and optimizer run. Missing values are shown as `N/A`.

| Metric | modelA_adam | modelB_sgd_m0.9 |
|---|---:|---:|
| Total parameters | 1,338,217 | 88,105 |
| Trainable parameters | 1,337,321 | 86,729 |
| Estimated model size on disk (KB) | 15,783.8 | 802.2 |
| Float32 weight size (KB) | 5,227.4 | 344.2 |
| MACs per inference | 155,583,616 | 6,478,400 |
| Training time per epoch (s) | 4.11 | 1.96 |
| Median epoch time (s) | 3.47 | 1.12 |
| Test loss | 0.7326 | 0.6939 |
| Test accuracy | 0.7546 | 0.7658 |
| Test precision (macro) | 0.7633 | 0.7778 |
| Test recall (macro) | 0.7522 | 0.7720 |
| Test F1 (macro) | 0.7555 | 0.7734 |
| Test F1 (weighted) | 0.7525 | 0.7664 |
| Best validation loss | 0.6996 | 0.5944 |
| Best validation accuracy | 0.7756 | 0.8191 |
