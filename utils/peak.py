"""Deterministic event extraction, soft labels and equation (9)."""
import numpy as np
import torch
from torch.nn import functional as F


def events(values, threshold, w=2, separation=3):
    values = np.asarray(values)
    candidates = []
    for i, value in enumerate(values):
        if value <= threshold or not np.isfinite(value):
            continue
        if i and values[i - 1] == value:  # earliest point of a plateau
            continue
        if value == np.max(values[max(0, i-w):i+w+1]):
            candidates.append(i)
    keep = []
    for i in sorted(candidates, key=lambda i: (-values[i], i)):
        if all(abs(i-j) >= separation for j in keep):
            keep.append(i)
    return sorted(keep)


def soft_labels(target, thresholds, channels, w=2, separation=3):
    array = target.detach().cpu().numpy()
    labels = np.zeros_like(array)
    axis = np.arange(array.shape[1])
    for b in range(len(array)):
        for c in channels:
            for t in events(array[b, :, c], thresholds[c], w, separation):
                distance = abs(axis-t)
                labels[b, :, c] = np.maximum(labels[b, :, c], np.where(distance <= 3, np.exp(-distance**2/2), 0))
    return torch.as_tensor(labels, device=target.device, dtype=target.dtype)


def peak_loss(pred, target, logits, labels, mask, omega=5., lambda_loc=.1, lambda_peak=.5, return_parts=False):
    weights = mask * (1 + (omega-1) * labels)
    loc = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).sum((1, 2)) / (weights.sum((1, 2)) + 1e-8)
    peak_weights = mask * labels
    peak = ((pred-target).abs() * peak_weights).sum((1, 2)) / (peak_weights.sum((1, 2)) + 1e-8)
    all_loss=F.mse_loss(pred,target)
    total=all_loss + lambda_loc * loc.mean() + lambda_peak * peak.mean()
    return (total, {'mse':all_loss,'loc':loc.mean(),'peak':peak.mean()}) if return_parts else total


def match_events(truth, predicted, tolerance=1):
    # Ordered dynamic programming: maximise cardinality, then minimise distance.
    table = [[(0, 0, []) for _ in range(len(predicted)+1)] for _ in range(len(truth)+1)]
    for i, t in enumerate(truth, 1):
        for j, p in enumerate(predicted, 1):
            choices = [table[i-1][j], table[i][j-1]]
            if abs(t-p) <= tolerance:
                count, cost, pairs = table[i-1][j-1]
                choices.append((count+1, cost+abs(t-p), pairs+[(t, p)]))
            table[i][j] = max(choices, key=lambda x: (x[0], -x[1]))
    return table[-1][-1][2]


def select_threshold(target, scores, thresholds, channels, w=2, separation=3, tolerance=1):
    """Validation only. NMS survivors can be filtered at any score threshold."""
    prepared=[]
    for b,y in enumerate(target):
        for c in channels:
            cutoff=thresholds[b,c] if np.ndim(thresholds)==2 else thresholds[c]
            true=events(y[:,c],cutoff,w,separation)
            predicted=events(scores[b,:,c],-np.inf,w,separation)
            prepared.append((true,predicted,scores[b,:,c]))
    best=None
    for threshold in np.linspace(0,1,101):
        tp=fp=fn=0
        for true,candidates,score in prepared:
            predicted=[i for i in candidates if score[i]>threshold]
            matched=len(match_events(true,predicted,tolerance))
            tp+=matched; fp+=len(predicted)-matched; fn+=len(true)-matched
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        key=(f1,-fp,float(threshold))
        if best is None or key>best[0]: best=(key,float(threshold),dict(tp=tp,fp=fp,fn=fn,f1=f1))
    return best[1],best[2]


def diagnostics(target, prediction, scores, thresholds, channels, theta, w=2, separation=3, tolerance=1):
    result = {}; tp = fp = fn = 0
    for c in channels:
        bias = []; amplitude = []; timing = []; near = []; rest = []
        ctp = cfp = cfn = 0
        for b, (y, pred, score) in enumerate(zip(target, prediction, scores)):
            cutoff = thresholds[b,c] if np.ndim(thresholds) == 2 else thresholds[c]
            gt = events(y[:, c], cutoff, w, separation)
            detected = events(score[:, c], cutoff if theta is None else theta, w, separation)
            pairs = match_events(gt, detected, tolerance)
            ctp += len(pairs); cfp += len(detected)-len(pairs); cfn += len(gt)-len(pairs)
            region = np.zeros(len(y), dtype=bool)
            for t in gt:
                region[max(0,t-tolerance):t+tolerance+1] = True
            error = abs(pred[:,c]-y[:,c])
            near.extend(error[region]); rest.extend(error[~region])
            for t, p in pairs:
                bias.append(float(pred[p,c]-y[t,c])); amplitude.append(abs(bias[-1])); timing.append(abs(p-t))
        avg = lambda a: float(np.mean(a)) if len(a) else None
        result[str(c)] = dict(bias=avg(bias), amplitude=avg(amplitude), timing=avg(timing), neighbourhood=avg(near), rest=avg(rest))
        tp += ctp; fp += cfp; fn += cfn
    result.update(tp=tp, fp=fp, fn=fn, recall=tp/(tp+fn) if tp+fn else None, precision=tp/(tp+fp) if tp+fp else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    for key in ('amplitude', 'timing', 'neighbourhood', 'rest'):
        defined = [result[str(c)][key] for c in channels if result[str(c)][key] is not None]
        result[key] = float(np.mean(defined)) if defined else None
    return result
