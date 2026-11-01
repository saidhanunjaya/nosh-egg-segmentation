import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - inter
    return inter / np.maximum(union, 1e-6)


def bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([cx, cy, s, r]).reshape((4, 1))


def z_to_bbox(z):
    cx, cy, s, r = z[0], z[1], z[2], z[3]
    w = np.sqrt(max(s * r, 0))
    h = s / (w + 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).flatten()


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, mask=None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = bbox_to_z(bbox)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.mask = mask
        self.last_bbox = np.array(bbox, dtype=float)  # last MEASURED box

    def update(self, bbox, mask=None):
        self.last_bbox = np.array(bbox, dtype=float)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_z(bbox))
        if mask is not None:
            self.mask = mask

    def predict(self):
        if self.time_since_update > 0:
            self.kf.x[4:7] = 0.0      # already coasting: freeze, don't drift
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return z_to_bbox(self.kf.x)

    def get_state(self):
        return z_to_bbox(self.kf.x)


def associate(detections, trackers, iou_threshold=0.25):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int)
        )

    iou_matrix = iou_batch(detections, trackers)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches, unmatched_dets, unmatched_trks = [], [], []
    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_dets.append(d)
    for t in range(len(trackers)):
        if t not in col_ind:
            unmatched_trks.append(t)
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_threshold:
            unmatched_dets.append(r)
            unmatched_trks.append(c)
        else:
            matches.append([r, c])

    return np.array(matches), np.array(unmatched_dets), np.array(unmatched_trks)


class Sort:
    """
    SORT tracker with a short-term 'lost track' memory for ID revival.

    Vanilla SORT permanently forgets a track once it exceeds max_age with
    no matching detection -- any later re-detection of that same physical
    object gets a brand new ID. In this video, eggs are frequently missed
    for short stretches (motion blur while spinning, brief occlusion by
    the hand), so vanilla SORT caused IDs to climb indefinitely.

    Fix: when a track dies, its last MEASURED bbox + id are kept in a
    'lost_tracks' buffer for `lost_buffer_age` frames. Any unmatched
    detection is first checked against this buffer (nearest centroid,
    via Hungarian assignment) and, if found, revives that old ID instead
    of minting a new one.

    Only tracks matched to a real detection in the current frame are
    returned, so there are no ghost boxes with stale masks.

    Note: revival uses no distance gate, which is valid only for this
    controlled scene (fixed, small number of eggs). It would need a
    distance/appearance gate to generalize.
    """

    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.25,
                 lost_buffer_age=120):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.lost_buffer_age = lost_buffer_age
        self.trackers = []
        self.lost_tracks = []  # list of {'id', 'bbox', 'frames_lost'}

    def update(self, detections):
        pred_boxes = np.array([t.predict() for t in self.trackers]) if self.trackers else np.empty((0, 4))
        det_boxes = np.array([d['bbox'] for d in detections]) if detections else np.empty((0, 4))

        matches, unmatched_dets, unmatched_trks = associate(
            det_boxes, pred_boxes, self.iou_threshold
        )

        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(detections[d_idx]['bbox'], detections[d_idx].get('mask'))

        # Try to revive IDs for unmatched detections using nearest lost track.
        revival_map = self._match_lost_tracks(det_boxes, unmatched_dets)

        for d_idx in unmatched_dets:
            bbox = detections[d_idx]['bbox']
            mask = detections[d_idx].get('mask')
            new_trk = KalmanBoxTracker(bbox, mask)
            if d_idx in revival_map:
                new_trk.id = revival_map[d_idx]
                new_trk.hits = self.min_hits
                new_trk.hit_streak = self.min_hits
            self.trackers.append(new_trk)

        # Move dead trackers into the lost-track buffer instead of discarding.
        alive = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                alive.append(t)
            else:
                self.lost_tracks.append({
                    'id': t.id,
                    'bbox': t.last_bbox,      # last measured box, not drifted prediction
                    'frames_lost': 0
                })
        self.trackers = alive

        for lt in self.lost_tracks:
            lt['frames_lost'] += 1
        self.lost_tracks = [
            lt for lt in self.lost_tracks if lt['frames_lost'] <= self.lost_buffer_age
        ]

        # Only report tracks matched to a real detection THIS frame.
        results = []
        for t in self.trackers:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                results.append({'id': t.id, 'bbox': t.get_state(), 'mask': t.mask})
        return results

    def _match_lost_tracks(self, det_boxes, unmatched_det_indices):
        if not self.lost_tracks or len(unmatched_det_indices) == 0:
            return {}

        def center(b):
            return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])

        lost_centers = np.array([center(lt['bbox']) for lt in self.lost_tracks])
        det_idx_list = list(unmatched_det_indices)
        det_centers = np.array([center(det_boxes[i]) for i in det_idx_list])

        dist_matrix = np.linalg.norm(
            det_centers[:, None, :] - lost_centers[None, :, :], axis=2
        )
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        revival_map = {}
        used_lost = []
        for r, c in zip(row_ind, col_ind):
            det_idx = det_idx_list[r]
            revival_map[det_idx] = self.lost_tracks[c]['id']
            used_lost.append(c)

        for c in sorted(used_lost, reverse=True):
            del self.lost_tracks[c]

        return revival_map