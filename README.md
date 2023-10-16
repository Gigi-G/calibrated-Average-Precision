![test](https://github.com/Gigi-G/calibrated-Average-Precision/workflows/test/badge.svg)

## Introduction



This README provides an explanation of calibrated Average Precision (cAP). The cAP metric is particularly useful when assessing the performance of online action detectors, which need to make decisions at every frame for every action. It aims to address some of the shortcomings of the traditional Average Precision (AP) metric in scenarios where the ratio of positive frames to negative background frames is not balanced.



## Traditional Average Precision (AP)

The traditional AP metric works as follows:

1. Frames are ranked in descending order of their confidence scores (from high to low).
2. Precision for a class at a given cutoff `k` in this ranked list is calculated using the formula: `Precision(k) = TP(k) / (TP(k) + FP(k))`, where `TP(k)` is the number of true positive frames, and `FP(k)` is the number of false positive frames at the cutoff.
3. The average precision of a class is defined as: `AP = Σ Precision(k) * Indicator(k) / P`, where `Indicator(k)` is an indicator function that equals 1 if frame `k` is a true positive and 0 otherwise, and `P` is the total number of positive frames.
4. The mean of the AP values across all classes (mAP) is the final performance metric for online action detection.

However, this traditional metric has a significant drawback. It is sensitive to changes in the ratio of positive frames to negative background frames, which can lead to variations in performance metrics. For instance, if there is a relatively larger amount of background data compared to true positives, the probability increases that some background frames are falsely detected with higher confidence than true positives, resulting in a decrease in AP.



## Introducing Calibrated Precision

To address the sensitivity of the traditional AP metric to imbalanced datasets, the calibrated precision (cPrec) is introduced. The calibrated precision takes into account the ratio of negative frames to positive frames. It is calculated as follows:

`cPrec = w * TP / (w * TP + FP)`

Here, `w` represents the ratio between the number of negative frames and positive frames. This ratio is chosen such that the total weight of the negatives becomes equal to the total weight of the positives.



## calibrated Average Precision (cAP)

Using calibrated precision, a new metric called calibrated Average Precision (cAP) is defined. It is calculated in a manner similar to traditional AP, but with calibrated precision:

`cAP = Σ cPrec(k) * Indicator(k) / P`

This metric evaluates the average precision as if there were an equal number of positive and negative frames, effectively making the random score 50%. This approach allows for a more equitable comparison of different classes and datasets with varying positive-to-negative ratios.



## Bibliography

[LINK](https://arxiv.org/abs/1604.06506)
```tex
@inproceedings{de2016online,
  title={Online action detection},
  author={De Geest, Roeland and Gavves, Efstratios and Ghodrati, Amir and Li, Zhenyang and Snoek, Cees and Tuytelaars, Tinne},
  booktitle={Computer Vision--ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14},
  pages={269--284},
  year={2016},
  organization={Springer}
}
```

