"""
Evaluation functions as in UKPGAN (https://github.com/qq456cvb/UKPGAN/blob/master/eval_iou.py)
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors


def eval_det_cls(pred, gt, geo_dists, dist_thresh=0.1, confidence_thresh=0.0):
    npos = 0
    fp_sum = 0
    fn_sum = 0
    for mesh_name in gt.keys():
        gt_kps = np.array(gt[mesh_name]).astype(np.int32)
        npos += len(gt_kps)
        if confidence_thresh > 0:
            selection = np.array(pred[mesh_name]["confidence"]) > confidence_thresh
            pred_kps = np.array(pred[mesh_name]["indices"])[selection].astype(np.int32)
        else:
            pred_kps = np.array(pred[mesh_name]["indices"]).astype(np.int32)
        fp = np.count_nonzero(np.all(geo_dists[mesh_name][pred_kps][:, gt_kps] > dist_thresh, axis=-1))
        fp_sum += fp
        fn = np.count_nonzero(np.all(geo_dists[mesh_name][gt_kps][:, pred_kps] > dist_thresh, axis=-1))
        fn_sum += fn

    return (npos - fn_sum) / np.maximum(npos + fp_sum, np.finfo(np.float64).eps)


def eval_iou(pred_all, gt_all, geo_dists, dist_thresh=0.05, confidence_thresh=0.0):
    iou = {}
    for classname in gt_all.keys():
        iou[classname] = eval_det_cls(pred_all[classname], gt_all[classname], geo_dists, dist_thresh, confidence_thresh)

    return iou


def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode='distance', include_self=False, n_jobs=-1)
    return shortest_path(graph, directed=False)
