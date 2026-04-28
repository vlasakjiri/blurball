import os
import os.path as osp
import matplotlib.pyplot as plt
import shutil
import torchvision.transforms as T
import pandas as pd
from pathlib import Path
import time
import logging
from collections import defaultdict
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from PIL import Image
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt

from dataloaders import build_dataloader
from detectors import build_detector
from trackers import build_tracker
from utils import mkdir_if_missing, draw_frame, draw_trail, gen_video, Center, Evaluator
from utils.image import get_affine_transform, affine_transform
from utils.preprocess import process_video

from .base import BaseRunner


# # Build the dataloader
# transform_train = T.Compose(
#     [
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )


def _build_inference_transforms(frame_width, frame_height, cfg, frames_out):
    center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
    scale = max(frame_height, frame_width) * 1.0
    input_size = [cfg["model"]["inp_width"], cfg["model"]["inp_height"]]
    trans_input = get_affine_transform(center, scale, 0, input_size, inv=0)
    trans_input_inv = get_affine_transform(center, scale, 0, input_size, inv=1)
    trans_output_inv = np.stack([trans_input_inv for _ in range(frames_out)], axis=0)
    trans_output_inv = torch.tensor(trans_output_inv, dtype=torch.float32)[None, :]
    return trans_input, trans_output_inv


def _fill_short_gaps(results_by_path, max_gap, max_step_distance):
    ordered_paths = list(results_by_path.keys())
    results = []
    for path in ordered_paths:
        result = dict(results_by_path[path])
        result["inferred"] = False
        results.append(result)

    num_results = len(results)
    left_idx = 0
    while left_idx < num_results:
        if not results[left_idx]["visi"]:
            left_idx += 1
            continue

        right_idx = left_idx + 1
        while right_idx < num_results and not results[right_idx]["visi"]:
            right_idx += 1

        gap = right_idx - left_idx - 1
        if right_idx >= num_results or gap <= 0 or gap > max_gap:
            left_idx = right_idx
            continue

        left_xy = np.array([results[left_idx]["x"], results[left_idx]["y"]], dtype=float)
        right_xy = np.array(
            [results[right_idx]["x"], results[right_idx]["y"]], dtype=float
        )
        step_distance = np.linalg.norm(right_xy - left_xy) / (gap + 1)
        if step_distance > max_step_distance:
            left_idx = right_idx
            continue

        for gap_idx in range(1, gap + 1):
            alpha = gap_idx / (gap + 1)
            interp_xy = (1.0 - alpha) * left_xy + alpha * right_xy
            result = results[left_idx + gap_idx]
            result["x"] = float(interp_xy[0])
            result["y"] = float(interp_xy[1])
            result["visi"] = True
            result["inferred"] = True
            result["score"] = min(results[left_idx]["score"], results[right_idx]["score"])
            if "angle" in result:
                result["angle"] = 0.0
            if "length" in result:
                result["length"] = 0.0

        left_idx = right_idx

    return ordered_paths, results


@torch.no_grad()
def inference_video(
    detector,
    tracker,
    input_video_path,
    frame_dir,
    cfg,
    vis_frame_dir=None,
    vis_hm_dir=None,
    vis_traj_path=None,
    dist_thresh=10.0,
):
    frames_in = detector.frames_in
    frames_out = detector.frames_out

    # +---------------
    t_start = time.time()

    det_results = []
    hm_results = []
    num_frames = 0
    print("Starting********")

    # Get all frames
    imgs_paths = sorted(Path(frame_dir).glob("*.png"))

    cap = cv2.VideoCapture(str(input_video_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    trans_input, trans = _build_inference_transforms(w, h, cfg, frames_out)
    normalize_frame = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    step = cfg["detector"]["step"]
    det_results = defaultdict(list)
    hm_results = defaultdict(list)
    img_paths_buffer = []
    frames_buffer = []
    for img_path in imgs_paths:
        # cv2.imshow("test", frame)
        # cv2.waitKey(1)
        # num_frames += imgs.shape[0] * frames_in
        frame = cv2.imread(str(img_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_buffer.append(frame)
        img_paths_buffer.append(str(img_path))
        if len(frames_buffer) == cfg["model"]["frames_in"]:
            # Preprocess the frames
            frames_processed = []
            for frame_rgb in frames_buffer:
                warped = cv2.warpAffine(
                    frame_rgb,
                    trans_input,
                    (cfg["model"]["inp_width"], cfg["model"]["inp_height"]),
                    flags=cv2.INTER_LINEAR,
                )
                frames_processed.append(normalize_frame(Image.fromarray(warped)))
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(
                0
            )  # .to(device)
            batch_results, hms_vis = detector.run_tensor(input_tensor, trans)

            for ie in batch_results[0].keys():
                path = img_paths_buffer[ie]
                preds = batch_results[0][ie]
                det_results[path].extend(preds)
                hm_results[path].extend(hms_vis[0][ie])
            if step == 1:
                frames_buffer.pop(0)
                img_paths_buffer.pop(0)
            elif step == 3:
                img_paths_buffer = []
                frames_buffer = []

    tracker.refresh()
    result_dict = {}
    print("Running tracker")
    for img_path, preds in det_results.items():
        result_dict[img_path] = tracker.update(preds)
    print("Finished tracking")
    ordered_paths, ordered_results = _fill_short_gaps(
        result_dict,
        max_gap=cfg["runner"]["gap_fill_max_gap"],
        max_step_distance=cfg["runner"]["gap_fill_max_step_distance"],
    )

    # print(result_dict)
    t_elapsed = time.time() - t_start
    # +---------------

    cm_pred = plt.get_cmap("Reds", len(result_dict))

    x_fin, y_fin, vis_fin = [], [], []
    if cfg["model"]["name"] == "blurball":
        l_fin, theta_fin = ([], [])

    trail_points = []
    max_trail_points = 18
    cnt = 0
    inferred_fin = []
    for cnt, img_path in enumerate(ordered_paths):
        # xy_pred = (result_dict[cnt]["x"], result_dict[cnt]["y"])
        current_result = ordered_results[cnt]
        x_pred = current_result["x"]
        y_pred = current_result["y"]
        visi_pred = current_result["visi"]
        score_pred = current_result["score"]
        inferred_pred = current_result["inferred"]
        if cfg["model"]["name"] == "blurball":
            angle_pred = current_result["angle"]
            length_pred = current_result["length"]

        # Save the predictions
        x_fin.append(int(min(max(x_pred, 0), 100000)))
        y_fin.append(int(min(max(y_pred, 0), 100000)))
        vis_fin.append(int(visi_pred))
        inferred_fin.append(int(inferred_pred))
        if cfg["model"]["name"] == "blurball":
            theta_fin.append(angle_pred)
            l_fin.append(length_pred)

        if visi_pred:
            trail_points.append((int(x_pred), int(y_pred)))
            trail_points = trail_points[-max_trail_points:]
        else:
            trail_points.append(None)
            trail_points = trail_points[-max_trail_points:]

        # cv2.imshow("test", 250 * hm_results[img_path][0]["hm"])
        # cv2.waitKey(800)

        if vis_frame_dir is not None:
            vis_frame_path = (
                osp.join(vis_frame_dir, osp.basename(img_path))
                if vis_frame_dir is not None
                else None
            )
            hm_path = (
                osp.join(vis_hm_dir, osp.basename(img_path))
                if vis_frame_dir is not None
                else None
            )
            vis_gt = cv2.imread(img_path)
            vis_pred = cv2.imread(img_path)

            for cnt2, img_path2 in enumerate(ordered_paths):
                if cnt2 != cnt:
                    continue
                if cnt2 > cnt:
                    break

                frame_result = ordered_results[cnt2]
                x_pred = frame_result["x"]
                y_pred = frame_result["y"]
                visi_pred = frame_result["visi"]
                score_pred = frame_result["score"]
                inferred_pred = frame_result["inferred"]
                if cfg["model"]["name"] == "blurball":
                    angle_pred = frame_result["angle"]
                    length_pred = frame_result["length"]

                color_pred = (
                    int(cm_pred(cnt2)[2] * 255),
                    int(cm_pred(cnt2)[1] * 255),
                    int(cm_pred(cnt2)[0] * 255),
                )

                color_pred = (255, 0, 0) if not inferred_pred else (0, 215, 255)
                vis_pred = draw_trail(vis_pred, trail_points)
                if cfg["model"]["name"] == "blurball":
                    vis_pred = draw_frame(
                        vis_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                        angle=angle_pred,
                        l=length_pred,
                    )
                    vis_hm_pred = cv2.cvtColor(
                        (255 * hm_results[img_path][0]["hm"]).astype(np.uint8),
                        cv2.COLOR_GRAY2RGB,
                    )
                    vis_hm_pred = cv2.resize(vis_hm_pred, (1280, 720))
                    vis_hm_pred = draw_frame(
                        vis_hm_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                        angle=angle_pred,
                        l=length_pred,
                    )
                else:
                    vis_pred = draw_frame(
                        vis_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                    )
                    vis_hm_pred = cv2.cvtColor(
                        (255 * hm_results[img_path][0]["hm"]).astype(np.uint8),
                        cv2.COLOR_GRAY2RGB,
                    )
                    vis_hm_pred = draw_frame(
                        vis_hm_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                    )

            # vis = np.hstack((vis_gt, vis_pred))
            vis = vis_pred
            cv2.imwrite(vis_frame_path, vis)
            cv2.imwrite(hm_path, vis_hm_pred)

        # if vis_traj_path is not None:
        #     color_pred = (
        #         int(cm_pred(cnt)[2] * 255),
        #         int(cm_pred(cnt)[1] * 255),
        #         int(cm_pred(cnt)[0] * 255),
        #     )
        #     vis = visualizer.draw_frame(
        #         vis,
        #         center_gt=center_gt,
        #         color_gt=color_gt,
        #     )

    if vis_frame_dir is not None:
        video_path = "{}.mp4".format(vis_frame_dir)
        gen_video(video_path, vis_frame_dir, fps=25.0)
        print("Saving video at " + video_path)

    # Save the evaluation results
    if cfg["model"]["name"] == "blurball":
        df = pd.DataFrame(
            {
                "Frame": x_fin,
                "X": x_fin,
                "Y": y_fin,
                "Visibility": vis_fin,
                "Inferred": inferred_fin,
                "L": l_fin,
                "Theta": theta_fin,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "Frame": x_fin,
                "X": x_fin,
                "Y": y_fin,
                "Visibility": vis_fin,
                "Inferred": inferred_fin,
            }
        )
    df["Frame"] = df.index
    df.to_csv(osp.join(frame_dir, "traj.csv"), index=False)
    print("Saving csv at " + osp.join(frame_dir, "traj.csv"))

    return {"t_elapsed": t_elapsed, "num_frames": num_frames}


@torch.no_grad()
def inference_video_memory(
    detector,
    tracker,
    input_video_path,
    output_dir,
    cfg,
    vis_frame_dir=None,
    vis_hm_dir=None,
    vis_traj_path=None,
    dist_thresh=10.0,
):
    frames_out = detector.frames_out
    t_start = time.time()

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    trans_input, trans = _build_inference_transforms(w, h, cfg, frames_out)
    normalize_frame = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    step = cfg["detector"]["step"]
    det_results = defaultdict(list)
    hm_results = defaultdict(list)
    frame_store = {}
    img_paths_buffer = []
    frames_buffer = []
    frame_idx = 0
    num_frames = 0
    print("Starting********")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_name = f"{frame_idx:05d}.png"
        if vis_frame_dir is not None:
            frame_store[frame_name] = frame_bgr.copy()
        frames_buffer.append(frame_rgb)
        img_paths_buffer.append(frame_name)
        frame_idx += 1
        num_frames += 1

        if len(frames_buffer) == cfg["model"]["frames_in"]:
            frames_processed = []
            for buffered_rgb in frames_buffer:
                warped = cv2.warpAffine(
                    buffered_rgb,
                    trans_input,
                    (cfg["model"]["inp_width"], cfg["model"]["inp_height"]),
                    flags=cv2.INTER_LINEAR,
                )
                frames_processed.append(normalize_frame(Image.fromarray(warped)))
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0)
            batch_results, hms_vis = detector.run_tensor(input_tensor, trans)

            for ie in batch_results[0].keys():
                path = img_paths_buffer[ie]
                preds = batch_results[0][ie]
                det_results[path].extend(preds)
                hm_results[path].extend(hms_vis[0][ie])

            if step == 1:
                frames_buffer.pop(0)
                img_paths_buffer.pop(0)
            elif step == 3:
                img_paths_buffer = []
                frames_buffer = []

    cap.release()

    tracker.refresh()
    result_dict = {}
    print("Running tracker")
    for img_path, preds in det_results.items():
        result_dict[img_path] = tracker.update(preds)
    print("Finished tracking")
    ordered_paths, ordered_results = _fill_short_gaps(
        result_dict,
        max_gap=cfg["runner"]["gap_fill_max_gap"],
        max_step_distance=cfg["runner"]["gap_fill_max_step_distance"],
    )

    t_elapsed = time.time() - t_start
    cm_pred = plt.get_cmap("Reds", len(result_dict))

    x_fin, y_fin, vis_fin = [], [], []
    if cfg["model"]["name"] == "blurball":
        l_fin, theta_fin = ([], [])

    trail_points = []
    max_trail_points = 18
    inferred_fin = []
    for cnt, img_path in enumerate(ordered_paths):
        current_result = ordered_results[cnt]
        x_pred = current_result["x"]
        y_pred = current_result["y"]
        visi_pred = current_result["visi"]
        inferred_pred = current_result["inferred"]
        if cfg["model"]["name"] == "blurball":
            angle_pred = current_result["angle"]
            length_pred = current_result["length"]

        x_fin.append(int(min(max(x_pred, 0), 100000)))
        y_fin.append(int(min(max(y_pred, 0), 100000)))
        vis_fin.append(int(visi_pred))
        inferred_fin.append(int(inferred_pred))
        if cfg["model"]["name"] == "blurball":
            theta_fin.append(angle_pred)
            l_fin.append(length_pred)

        if visi_pred:
            trail_points.append((int(x_pred), int(y_pred)))
            trail_points = trail_points[-max_trail_points:]
        else:
            trail_points.append(None)
            trail_points = trail_points[-max_trail_points:]

        if vis_frame_dir is not None:
            vis_frame_path = osp.join(vis_frame_dir, osp.basename(img_path))
            hm_path = osp.join(vis_hm_dir, osp.basename(img_path))
            vis_pred = frame_store[img_path].copy()

            frame_result = ordered_results[cnt]
            x_pred = frame_result["x"]
            y_pred = frame_result["y"]
            visi_pred = frame_result["visi"]
            inferred_pred = frame_result["inferred"]
            if cfg["model"]["name"] == "blurball":
                angle_pred = frame_result["angle"]
                length_pred = frame_result["length"]

            color_pred = (255, 0, 0) if not inferred_pred else (0, 215, 255)
            vis_pred = draw_trail(vis_pred, trail_points)
            if cfg["model"]["name"] == "blurball":
                vis_pred = draw_frame(
                    vis_pred,
                    center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                    color=color_pred,
                    radius=3,
                    angle=angle_pred,
                    l=length_pred,
                )
                vis_hm_pred = cv2.cvtColor(
                    (255 * hm_results[img_path][0]["hm"]).astype(np.uint8),
                    cv2.COLOR_GRAY2RGB,
                )
                vis_hm_pred = cv2.resize(vis_hm_pred, (1280, 720))
                vis_hm_pred = draw_frame(
                    vis_hm_pred,
                    center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                    color=color_pred,
                    radius=3,
                    angle=angle_pred,
                    l=length_pred,
                )
            else:
                vis_pred = draw_frame(
                    vis_pred,
                    center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                    color=color_pred,
                    radius=3,
                )
                vis_hm_pred = cv2.cvtColor(
                    (255 * hm_results[img_path][0]["hm"]).astype(np.uint8),
                    cv2.COLOR_GRAY2RGB,
                )
                vis_hm_pred = draw_frame(
                    vis_hm_pred,
                    center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                    color=color_pred,
                    radius=3,
                )

            cv2.imwrite(vis_frame_path, vis_pred)
            cv2.imwrite(hm_path, vis_hm_pred)

    if vis_frame_dir is not None:
        video_path = "{}.mp4".format(vis_frame_dir)
        gen_video(video_path, vis_frame_dir, fps=25.0)
        print("Saving video at " + video_path)

    if cfg["model"]["name"] == "blurball":
        df = pd.DataFrame(
            {
                "Frame": x_fin,
                "X": x_fin,
                "Y": y_fin,
                "Visibility": vis_fin,
                "Inferred": inferred_fin,
                "L": l_fin,
                "Theta": theta_fin,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "Frame": x_fin,
                "X": x_fin,
                "Y": y_fin,
                "Visibility": vis_fin,
                "Inferred": inferred_fin,
            }
        )
    df["Frame"] = df.index
    traj_path = osp.join(output_dir, "traj.csv")
    df.to_csv(traj_path, index=False)
    print("Saving csv at " + traj_path)

    return {"t_elapsed": t_elapsed, "num_frames": num_frames}


class NewVideosInferenceRunner(BaseRunner):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(cfg)
        # print(cfg["input_vid"])

        self._vis_result = cfg["runner"]["vis_result"]
        self._vis_hm = cfg["runner"]["vis_hm"]
        self._vis_traj = cfg["runner"]["vis_traj"]
        self._in_memory = cfg["runner"].get("in_memory", False)
        self._input_vid_path = Path(cfg["input_vid"])

    def run(self, model=None, model_dir=None):
        return self._run_model(model=model)

    def _run_model(self, model=None):
        detector = build_detector(self._cfg, model=model)
        tracker = build_tracker(self._cfg)

        t_elapsed_all = 0.0
        num_frames_all = 0

        vis_frame_dir, vis_hm_dir, vis_traj_path = None, None, None
        if self._vis_result:
            vis_frame_dir = osp.join(self._input_vid_path.parent, "frames")
            mkdir_if_missing(vis_frame_dir)
        if self._vis_hm:
            vis_hm_dir = osp.join(self._input_vid_path.parent, "hm")
            mkdir_if_missing(vis_hm_dir)
        # if self._vis_traj:
        #     vis_traj_dir = osp.join(self._output_dir, "vis_traj")
        #     mkdir_if_missing(vis_traj_dir)
        #     vis_traj_path = osp.join(vis_traj_dir, "{}_{}.png".format(match, clip_name))

        if self._in_memory:
            tmp = inference_video_memory(
                detector,
                tracker,
                self._input_vid_path,
                self._output_dir,
                self._cfg,
                vis_frame_dir=vis_frame_dir,
                vis_hm_dir=vis_hm_dir,
            )
        else:
            frame_dir = process_video(self._input_vid_path)
            print("Finished preprocess_video")
            tmp = inference_video(
                detector,
                tracker,
                self._input_vid_path,
                frame_dir,
                self._cfg,
                vis_frame_dir=vis_frame_dir,
                vis_hm_dir=vis_hm_dir,
            )

        t_elapsed_all += tmp["t_elapsed"]
        num_frames_all += tmp["num_frames"]

        return
