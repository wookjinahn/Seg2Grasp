"""Full official suctionnetAPI evaluation over test_seen (scenes 100-129, all 256 frames).

Uses the STOCK SuctionNetEval.eval_seen (which already covers scenes 100-129 x
range(256)), with SuctionGroup.nms monkey-patched to the numpy translation-only
fallback (suction_nms compiled op is unavailable; API calls nms(0.02,181deg) so
rotation never filters -> faithful 2cm greedy dedup). The monkeypatch is applied
before the multiprocessing Pool is created, so forked workers inherit it.

Saves the raw per-scene res array (30,256,50,4) to <dump>_res.npy for re-aggregation.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from suctionnetAPI import SuctionNetEval
from suctionnetAPI.suction import SuctionGroup


def nms_numpy(arr, translation_thresh=0.02):
    if len(arr) == 0:
        return arr
    trans = arr[:, 4:7]
    order = np.argsort(-arr[:, 0])
    suppressed = np.zeros(len(arr), dtype=bool)
    keep = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        suppressed |= (np.linalg.norm(trans - trans[idx], axis=1) < translation_thresh)
    return arr[keep]


SuctionGroup.nms = lambda self, t=0.02, r=None: SuctionGroup(nms_numpy(self.suction_group_array, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=config.GRASPNET_ROOT)
    ap.add_argument("--camera", default=config.CAMERA)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--proc", type=int, default=12)
    ap.add_argument("--res-out", default=None, help="path to np.save the raw res array")
    args = ap.parse_args()

    ev = SuctionNetEval(root=args.root, camera=args.camera)
    res, ap_top50, ap_top1 = ev.eval_seen(dump_folder=args.dump, proc=args.proc)
    res = np.asarray(res)

    res_out = args.res_out or (args.dump.rstrip("/") + "_res.npy")
    np.save(res_out, res)

    t1 = [float(np.mean(res[:, :, :1, i])) for i in range(4)]
    print("\n########## SUMMARY %s ##########" % os.path.basename(args.dump.rstrip("/")))
    print("AP_top50 = %.4f" % float(np.mean(res[:, :, :50, :])))
    print("AP_top1  = %.4f" % float(np.mean(res[:, :, :1, :])))
    print("AP_top1 per-thr [0.2,0.4,0.6,0.8] = [%s]" % ", ".join("%.4f" % x for x in t1))
    print("raw res saved -> %s  shape=%s" % (res_out, res.shape))


if __name__ == "__main__":
    main()
