import numpy as np
from tqdm import tqdm
import warnings

from sklearn.metrics import average_precision_score as sk_average_precision_score

def precision_recall_curve(
    y_true:np.ndarray,
    y_score:np.ndarray,
    calibrated:bool=False
):
    """Compute precision-recall pairs for different probability thresholds.

    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold. This ensures that the graph starts on the
    y axis.

    The first precision and recall values are precision=class balance and recall=1.0
    which corresponds to a classifier that always predicts the positive class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    probas_pred : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, or non-thresholded measure of decisions (as returned by
        `decision_function` on some classifiers).

    calibrated : bool, default=False
        This is useful when there are a lots of negative labels and few positive
        labels. The precision will be weighted by the ratio between the number
        of negative labels and the number of positive labels.

    Returns
    -------
    precision : ndarray of shape (n_thresholds + 1,)
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : ndarray of shape (n_thresholds + 1,)
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : ndarray of shape (n_thresholds,)
        Increasing thresholds on the decision function used to compute
        precision and recall where `n_thresholds = len(np.unique(probas_pred))`.
    """
    # Check if type of y_true and y_score are np.ndarray
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        raise TypeError("y_true and y_score must be numpy.ndarray")
    
    # Check if y_true is an array of labels or label indicators
    if (
        np.ndim(y_true) > 1 and y_true.shape[1] > 1
    ):
        raise ValueError("precision_recall_curve expects y_true to be a 1d array of labels.")
    
    # Convert y_true and y_score to np.ndarray dtype=float64
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    
    # Initialize the result array with zeros to make sure that precision[ps == 0]
    # does not contain uninitialized values.
    precision = np.zeros_like(tps)

    # If 'calibrated' is True and there are both positive and negative labels,
    # calculate the weight 'w' as the ratio between the number of negative labels
    # and the number of positive labels. The negatives becomes equal to the total
    # weight of the positives.
    if calibrated and (len(y_true) - sum(y_true)) > 0 and sum(y_true) > 0:
        w = (len(y_true) - sum(y_true)) / sum(y_true)
        ps = w * tps + fps
        np.divide(w * tps, ps, out=precision, where=(ps != 0))
    else:
        ps = tps + fps
        np.divide(tps, ps, out=precision, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]

def average_precision_score(
    y_true:np.ndarray,
    y_score:np.ndarray,
    calibrated:bool=False
):
    """Compute average precision (AP) from prediction scores.

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    calibrated : bool, default=False
        Calculate the calibrated precision, which outputs the calibrated
        average precision.

    Returns
    -------
    average_precision : float
        Average precision score.
        
    """
    def _binary_uninterpolated_average_precision(
        y_true, y_score, calibrated=False
    ):
        precision, recall, _ = precision_recall_curve(
            y_true,
            y_score,
            calibrated=calibrated
        )
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
    
    # Check if type of y_true and y_score are np.ndarray
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        raise TypeError("y_true and y_score must be numpy.ndarray")
    
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    
    # Check if y_true is an array of labels or label indicators
    if (
        np.ndim(y_true) > 1 and y_true.shape[1] > 1
    ):
        y_true = np.transpose(y_true)
        y_score = np.transpose(y_score)
        mAP = 0.0
        for i in range(y_true.shape[0]):
            mAP += _binary_uninterpolated_average_precision(y_true[i], y_score[i], calibrated=calibrated)
        return mAP / y_true.shape[0]
    else:
        return _binary_uninterpolated_average_precision(y_true, y_score, calibrated=calibrated)

def testing_average_precision(length:int, calibrated=False, num_samples=1000, tolerance=1e-6) -> None:
    """
    Benchmark the performance of a custom average precision score implementation against sklearn's implementation.

    Parameters:
    - length: int
        The length of the random samples to use for benchmarking.
    - calibrated: bool, default=False
        Whether to use calibrated precision.
    - num_samples: int, default=1000
        The number of random samples to use for benchmarking.
    - tolerance: float, default=1e-6
        Tolerance for comparing the results.

    Returns:
    - result: str
        A message indicating whether the custom implementation matches sklearn's implementation within the specified tolerance.
    """

    # Create a set of random samples to benchmark
    for _ in tqdm(range(num_samples)):
        # The sample ground truth labels are length x length
        sample_y_true = np.random.randint(0, 2, size=(length, length))
        # The sample scores are length x length
        sample_y_score = np.random.rand(length, length)
        
        custom_ap_sample = average_precision_score(sample_y_true, sample_y_score, calibrated=calibrated)
        sklearn_ap_sample = sk_average_precision_score(sample_y_true, sample_y_score)

        # Check if the custom implementation matches sklearn's implementation
        if abs(custom_ap_sample - sklearn_ap_sample) > tolerance:
            assert False

    assert True

if __name__ == '__main__':
    np.random.seed(42)
    testing_average_precision(length=100)