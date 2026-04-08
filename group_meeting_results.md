# ChangeDINO Reproduction Summary

## Result Comparison

| Method | Key Setting | Acc | mIoU | mF1 | IoU_1 | F1_1 |
|---|---|---:|---:|---:|---:|---:|
| ChangeDINO-native256 | `256x256` non-overlap patch, original pipeline | 99.1949 | 92.1889 | 95.7988 | 85.2220 | 92.0215 |
| ChangeDINO + LoRA `r=4` | DINOv3 full LoRA | 99.2276 | 92.5237 | 95.9920 | 85.8578 | 92.3908 |
| ChangeDINO + LoRA `r=8` | DINOv3 full LoRA | 99.2299 | 92.5003 | 95.9783 | 85.8084 | 92.3622 |

## Main Conclusions

- The main reason the early reproduction underperformed was that the data preprocessing protocol was not aligned with the paper, especially the missing `256x256` non-overlap patch generation.
- After aligning the data protocol, native ChangeDINO can be stably reproduced at `F1_1 = 92.0215`.
- Adding full LoRA on DINOv3 brings a small but consistent gain over the native baseline.
- `r=4` and `r=8` are nearly tied, which suggests that LoRA rank is not the main bottleneck at this stage.
- The next priority should be decoder/fusion improvement rather than continuing to increase LoRA rank.

## One-Sentence Summary

Data protocol alignment solved the main reproduction issue, LoRA brought a modest gain, and the next likely bottleneck is decoder/fusion design.
