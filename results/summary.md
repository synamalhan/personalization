# Results summary

Source CSVs: `*_limit2000.csv` in `results/`. **These were run on a 2000-image random subset of Flowers102 only (about 20 images per class), not the full datasets, and Pets was not run.** Treat the numbers as a preview.
All intervals are 95% CIs (t-interval across independent episodes or seeds).

## Exp 1: top-1 accuracy vs photos per class
Episodes per cell: 100; 5-way.

| dataset | backbone | head | k | accuracy | macro-F1 |
| --- | --- | --- | --- | --- | --- |
| flowers102 | DINOv2-small | logistic | 1 | 0.996 [0.994, 0.999] | 0.996 [0.993, 0.999] |
| flowers102 | DINOv2-small | logistic | 3 | 0.997 [0.994, 1.000] | 0.997 [0.993, 1.000] |
| flowers102 | DINOv2-small | logistic | 5 | 0.998 [0.996, 1.000] | 0.998 [0.996, 1.000] |
| flowers102 | DINOv2-small | logistic | 10 | 0.999 [0.997, 1.000] | 0.999 [0.997, 1.000] |
| flowers102 | DINOv2-small | logistic | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| flowers102 | DINOv2-small | nearest class mean | 1 | 0.996 [0.992, 0.999] | 0.995 [0.992, 0.999] |
| flowers102 | DINOv2-small | nearest class mean | 3 | 0.997 [0.994, 1.000] | 0.997 [0.993, 1.000] |
| flowers102 | DINOv2-small | nearest class mean | 5 | 0.998 [0.995, 1.000] | 0.998 [0.995, 1.000] |
| flowers102 | DINOv2-small | nearest class mean | 10 | 0.999 [0.997, 1.000] | 0.999 [0.997, 1.000] |
| flowers102 | DINOv2-small | nearest class mean | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| flowers102 | MobileNetV3 | logistic | 1 | 0.845 [0.824, 0.866] | 0.833 [0.809, 0.857] |
| flowers102 | MobileNetV3 | logistic | 3 | 0.951 [0.939, 0.963] | 0.950 [0.938, 0.962] |
| flowers102 | MobileNetV3 | logistic | 5 | 0.961 [0.952, 0.970] | 0.961 [0.951, 0.970] |
| flowers102 | MobileNetV3 | logistic | 10 | 0.978 [0.972, 0.985] | 0.978 [0.972, 0.984] |
| flowers102 | MobileNetV3 | logistic | 20 | 0.976 [0.969, 0.982] | 0.975 [0.968, 0.982] |
| flowers102 | MobileNetV3 | nearest class mean | 1 | 0.834 [0.811, 0.857] | 0.822 [0.796, 0.847] |
| flowers102 | MobileNetV3 | nearest class mean | 3 | 0.946 [0.934, 0.959] | 0.945 [0.932, 0.958] |
| flowers102 | MobileNetV3 | nearest class mean | 5 | 0.952 [0.942, 0.963] | 0.951 [0.940, 0.962] |
| flowers102 | MobileNetV3 | nearest class mean | 10 | 0.971 [0.963, 0.979] | 0.970 [0.962, 0.979] |
| flowers102 | MobileNetV3 | nearest class mean | 20 | 0.965 [0.956, 0.974] | 0.964 [0.954, 0.974] |

**Worst case kept in:** MobileNetV3 / nearest class mean at k=1: accuracy 0.834 [0.811, 0.857]. At k=20 with `--n-query 5` only a few classes are eligible, so episodes reuse the same classes and the intervals are optimistic.

## Exp 2: logistic minus nearest class mean (paired by episode)

| dataset | backbone | k | d accuracy | d macro-F1 |
| --- | --- | --- | --- | --- |
| flowers102 | DINOv2-small | 1 | +0.001 [-0.001, +0.002] | +0.001 [-0.001, +0.003] |
| flowers102 | DINOv2-small | 3 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flowers102 | DINOv2-small | 5 | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.002] |
| flowers102 | DINOv2-small | 10 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flowers102 | DINOv2-small | 20 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flowers102 | MobileNetV3 | 1 | +0.010 [+0.003, +0.018] | +0.011 [+0.003, +0.019] |
| flowers102 | MobileNetV3 | 3 | +0.005 [-0.001, +0.010] | +0.005 [-0.001, +0.010] |
| flowers102 | MobileNetV3 | 5 | +0.009 [+0.004, +0.014] | +0.009 [+0.004, +0.015] |
| flowers102 | MobileNetV3 | 10 | +0.008 [+0.003, +0.012] | +0.008 [+0.003, +0.013] |
| flowers102 | MobileNetV3 | 20 | +0.010 [+0.006, +0.015] | +0.011 [+0.006, +0.016] |

## Exp 5: accuracy on earlier classes after adding class 10

| dataset | backbone | head | replay | sequential, no replay | newest class only |
| --- | --- | --- | --- | --- | --- |
| flowers102 | DINOv2-small | logistic | 0.999 [0.998, 0.999] | 0.049 [0.040, 0.059] | 0.000 [0.000, 0.000] |
| flowers102 | DINOv2-small | nearest class mean | 0.998 [0.997, 0.999] | 0.998 [0.997, 0.999] | 0.000 [0.000, 0.000] |
| flowers102 | MobileNetV3 | logistic | 0.935 [0.927, 0.942] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| flowers102 | MobileNetV3 | nearest class mean | 0.924 [0.916, 0.933] | 0.924 [0.916, 0.933] | 0.000 [0.000, 0.000] |

Sequential = keep the old head and add the new class from its own examples only. Nearest class mean cannot forget (its class means do not change), so it equals replay. Logistic forgets: training on the new class alone pushes old logits down. 'Newest class only' is 0 by construction (a head fit on one class can only predict that class); it is the Learner's `replay = false` setting.

## Exp 6: unknown detection (10 taught / 10 unseen classes)

| dataset | backbone | head | AUROC | ECE (15 bins) | top-1 on taught |
| --- | --- | --- | --- | --- | --- |
| flowers102 | DINOv2-small | cosine threshold rule | 0.997 [0.996, 0.998] | n/a | n/a |
| flowers102 | DINOv2-small | logistic | 0.997 [0.996, 0.997] | 0.119 [0.115, 0.123] | 0.999 [0.998, 0.999] |
| flowers102 | DINOv2-small | nearest class mean | 0.994 [0.992, 0.995] | 0.035 [0.032, 0.038] | 0.999 [0.998, 0.999] |
| flowers102 | MobileNetV3 | cosine threshold rule | 0.859 [0.850, 0.868] | n/a | n/a |
| flowers102 | MobileNetV3 | logistic | 0.891 [0.881, 0.900] | 0.396 [0.388, 0.404] | 0.931 [0.923, 0.939] |
| flowers102 | MobileNetV3 | nearest class mean | 0.885 [0.876, 0.895] | 0.507 [0.497, 0.517] | 0.918 [0.910, 0.927] |

Operating point of the PersonalizeCore threshold rule (max cosine to class means; threshold = mean - 2 std of leave-one-out similarities):

| dataset | backbone | threshold | taught accepted | unseen rejected |
| --- | --- | --- | --- | --- |
| flowers102 | DINOv2-small | 0.624 | 0.956 [0.949, 0.962] | 0.984 [0.978, 0.990] |
| flowers102 | MobileNetV3 | 0.698 | 0.976 [0.972, 0.981] | 0.277 [0.252, 0.302] |

**Worst case kept in:** lowest AUROC is MobileNetV3 / cosine threshold rule: 0.859 [0.850, 0.868]. MobileNetV3 nearest-class-mean ECE is large because its cosines are compressed and the softmax temperature (0.1) was not tuned.

## Exp 4: Core ML fp16 vs int8 (embedding cosine vs PyTorch)

| backbone | precision | mean cosine | worst image | max abs diff |
| --- | --- | --- | --- | --- |
| MobileNetV3 | fp16 | 0.9999 | 0.9998 | 0.071 |
| MobileNetV3 | int8 | 0.9224 | 0.8303 | 2.889 |
| DINOv2-small | fp16 | 0.9999 | 0.9998 | 0.157 |
| DINOv2-small | int8 | 0.9992 | 0.9986 | 0.471 |

300 Flowers102 test images, coremltools 9.0. This measures embedding similarity only; few-shot accuracy with int8 embeddings and latency were **not** measured.

## Gate check: phone vs laptop embeddings

| backbone | device / build | images | mean cosine | worst image | same predicted class (nearest mean / logistic) | max probability difference |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3 | arm64 / debug (simulator) | 20 | 0.9947 | 0.9855 | 100% / 100% | 0.029 |

Run in the iOS Simulator with CPU-only Core ML (the simulator returned all zeros with the default compute units), so this is **not** a phone result. The gap to the 0.999 target is most likely the resize method (Core Graphics vs Pillow). Not yet run on a real iPhone.

## Not available (not run yet)

- Exp 3 (accuracy vs latency) and Exp 7 (on-device cost): need the Release benchmark CSV from a real iPhone (missing).
- Exp 8 (real photos): needs `exp8_real.py` run on the app's exported vectors; none exist.
- Gate check on a real iPhone (only the simulator was run).
- Pets dataset and the full Flowers102 run for every experiment above.
