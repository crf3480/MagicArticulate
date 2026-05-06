#  Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import csv
import math
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs

from skeleton_models.skeletongen import SkeletonGPT
from utils.skeleton_data_loader import SkeletonData
from utils.save_utils import save_mesh, pred_joints_and_bones, save_skeleton_to_txt, save_args, \
                       merge_duplicate_joints_and_fix_bones, save_skeleton_obj, render_mesh_with_skeleton
from utils.eval_utils import chamfer_dist, joint2bone_chamfer_dist, bone2bone_chamfer_dist, joint_matching_metrics
from utils.medial_axis import compute_medial_axis_pts, snap_joints_to_medial_axis


def get_args():
    parser = argparse.ArgumentParser("SkeletonGPT", add_help=False)

    parser.add_argument("--input_pc_num", default=8192, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument('--llm', default="facebook/opt-350m", type=str, help="The LLM backend")
    parser.add_argument("--pad_id", default=-1, type=int, help="padding id")
    parser.add_argument("--n_discrete_size", default=128, type=int, help="size of discretized 3D space")
    parser.add_argument("--n_max_bones", default=100, type=int, help="max number of bones")
    parser.add_argument('--dataset_path', default="Articulation_xlv2.npz", type=str, help="data path")
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument('--save_name', default="infer_results", type=str)
    parser.add_argument("--save_render", default=False, action="store_true", help="save rendering results of mesh with skel")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)
    parser.add_argument('--pretrained_weights', default=None, type=str, help="path of pretrained models")
    parser.add_argument("--hier_order", default=False, action="store_true", help="use hier order")
    parser.add_argument("--max_samples", default=None, type=int, help="cap evaluation at N samples for a quick run")
    parser.add_argument("--threshold", default=0.05, type=float, help="distance threshold for precision/recall/iou")
    parser.add_argument("--metadata_path", default=None, type=str, help="optional CSV with uuid,category_label columns for per-category breakdown")
    parser.add_argument("--use_medial_axis", default=False, action="store_true", help="snap predicted joints to medial axis after generation")
    parser.add_argument("--snap_dist", default=0.05, type=float, help="max snap distance for medial axis refinement (in joint coordinate units)")
    parser.add_argument("--medial_grid_size", default=64, type=int, help="SDF grid resolution for medial axis extraction")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Load optional metadata for category-level breakdown
    uuid_to_category = {}
    if args.metadata_path and os.path.exists(args.metadata_path):
        with open(args.metadata_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                uuid_to_category[row['uuid']] = row['category_label'].strip().lower()
        print(f"[Metadata] Loaded categories for {len(uuid_to_category)} entries")

    dataset = SkeletonData.load(args, is_training=False)

    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
        print(f"[Dataset] Capped to {len(dataset)} samples via --max_samples")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        num_workers=2,
    )

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
        mixed_precision=args.precision,
    )

    model = SkeletonGPT(args).cuda()

    if args.pretrained_weights is not None:
        pkg = torch.load(args.pretrained_weights, map_location=torch.device("cpu"))
        model.load_state_dict(pkg["model"])
    else:
        raise ValueError("Pretrained weights must be provided.")

    set_seed(args.seed)
    dataloader, model = accelerator.prepare(dataloader, model)
    model.eval()

    output_dir = f'{args.output_dir}/{args.save_name}'
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    save_args(args, output_dir)

    gt_samples, pred_samples = [], []
    per_sample_rows = []
    infer_all_time = []
    num_skipped = 0

    for curr_iter, batch_data_label in tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True):
        uuid = batch_data_label['uuid'][0]
        start_time = time.time()
        with accelerator.autocast():
            pred_bone_coords = model.generate(batch_data_label)
        infer_time = time.time() - start_time
        infer_all_time.append(infer_time)

        if pred_bone_coords is None or pred_bone_coords.shape[1] == 0:
            num_skipped += 1
            continue

        gt_joints = batch_data_label['joints'].squeeze(0).cpu().numpy()
        gt_bones = batch_data_label['bones'].squeeze(0).cpu().numpy()

        pred_joints, pred_bones = pred_joints_and_bones(pred_bone_coords.cpu().numpy().squeeze(0))
        if pred_bones.shape[0] == 0:
            num_skipped += 1
            continue

        if args.hier_order:
            pred_root_index = pred_bones[0][0]
            pred_joints, pred_bones, pred_root_index = merge_duplicate_joints_and_fix_bones(pred_joints, pred_bones, root_index=pred_root_index)
        else:
            pred_joints, pred_bones = merge_duplicate_joints_and_fix_bones(pred_joints, pred_bones)
            pred_root_index = None

        gt_root_index = int(batch_data_label['root_index'][0])
        gt_joints, gt_bones, gt_root_index = merge_duplicate_joints_and_fix_bones(gt_joints, gt_bones, root_index=gt_root_index)

        if args.use_medial_axis:
            # Transform NPZ vertices into the same coordinate space as pred_joints
            # so medial axis points can be directly compared for snapping.
            tp = batch_data_label['transform_params'].squeeze(0).cpu().numpy()
            raw_verts = batch_data_label['vertices'].squeeze(0).cpu().numpy().astype(np.float32)
            raw_faces = batch_data_label['faces'].squeeze(0).cpu().numpy()
            verts_joint_space = (raw_verts - tp[:3]) * tp[3]
            verts_joint_space = (verts_joint_space - tp[4:7]) / tp[7]
            medial_pts = compute_medial_axis_pts(
                verts_joint_space, raw_faces, grid_size=args.medial_grid_size,
                verbose=(curr_iter == 0)
            )
            if curr_iter == 0:
                n_snapped = 0
                if medial_pts is not None:
                    dists = np.array([np.linalg.norm(medial_pts - j, axis=1).min() for j in pred_joints])
                    n_snapped = (dists < args.snap_dist).sum()
                print(f"[medial_axis] sample 0: medial_pts={'None' if medial_pts is None else len(medial_pts)}, "
                      f"joints={len(pred_joints)}, would_snap={n_snapped}")
            pred_joints = snap_joints_to_medial_axis(pred_joints, medial_pts, max_dist=args.snap_dist)
            # Snapping can collapse distinct joints onto the same voxel; re-merge.
            if args.hier_order:
                pred_joints, pred_bones, pred_root_index = merge_duplicate_joints_and_fix_bones(
                    pred_joints, pred_bones, root_index=pred_root_index)
            else:
                pred_joints, pred_bones = merge_duplicate_joints_and_fix_bones(pred_joints, pred_bones)

        if gt_bones.shape[0] == 0 or pred_bones.shape[0] == 0:
            num_skipped += 1
            continue

        j2j_cd = chamfer_dist(pred_joints, gt_joints)
        j2b_cd = joint2bone_chamfer_dist(pred_joints, pred_bones, gt_joints, gt_bones)
        b2b_cd = bone2bone_chamfer_dist(pred_joints, pred_bones, gt_joints, gt_bones)
        iou, precision, recall = joint_matching_metrics(pred_joints, gt_joints, threshold=args.threshold)

        if math.isnan(j2j_cd) or math.isnan(j2b_cd) or math.isnan(b2b_cd):
            print(f"NaN cd for {uuid}, skipping")
            num_skipped += 1
            continue

        category = uuid_to_category.get(uuid, 'unknown')
        per_sample_rows.append({
            'uuid': uuid,
            'category': category,
            'j2j': j2j_cd,
            'j2b': j2b_cd,
            'b2b': b2b_cd,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'infer_time': infer_time,
            'gt_joints': len(gt_joints),
            'gt_bones': len(gt_bones),
            'pred_joints': len(pred_joints),
            'pred_bones': len(pred_bones),
        })

        if len(gt_samples) <= 30:
            pred_samples.append((pred_joints, pred_bones, pred_root_index))
            gt_samples.append((gt_joints, gt_bones, batch_data_label['vertices'][0], batch_data_label['faces'][0], batch_data_label['transform_params'][0], uuid, gt_root_index))

    # ── Write per-sample CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['uuid', 'category', 'j2j', 'j2b', 'b2b', 'iou', 'precision', 'recall', 'infer_time', 'gt_joints', 'gt_bones', 'pred_joints', 'pred_bones'])
        writer.writeheader()
        writer.writerows(per_sample_rows)

    # ── Aggregate results ─────────────────────────────────────────────────────
    def mean(vals):
        return float(np.mean(vals)) if vals else float('nan')

    metric_keys = ['j2j', 'j2b', 'b2b', 'iou', 'precision', 'recall']
    agg = {k: mean([r[k] for r in per_sample_rows]) for k in metric_keys}
    n_valid = len(per_sample_rows)

    summary_lines = [
        "===== Aggregate Results =====",
        f"  {'j2j':<12}: {agg['j2j']:.4f}",
        f"  {'j2b':<12}: {agg['j2b']:.4f}",
        f"  {'b2b':<12}: {agg['b2b']:.4f}",
        f"  {'iou':<12}: {agg['iou']:.4f}",
        f"  {'precision':<12}: {agg['precision']:.4f}",
        f"  {'recall':<12}: {agg['recall']:.4f}",
        "",
        f"  Evaluated : {n_valid}",
        f"  Skipped   : {num_skipped}",
        f"  Avg infer : {mean(infer_all_time):.2f}s/sample",
        "",
        f"Per-sample results saved to: {csv_path}",
    ]

    # ── Per-category breakdown ────────────────────────────────────────────────
    categories = sorted(set(r['category'] for r in per_sample_rows))
    if per_sample_rows and not (len(categories) == 1 and categories[0] == 'unknown'):
        col_w = max(len(c) for c in categories) + 2
        header = f"\n===== Per-Category Results =====\n{'category_label':<{col_w}}" + \
                 "".join(f"{k:>12}" for k in metric_keys) + f"{'n':>6}"
        summary_lines.append(header)
        for cat in categories:
            rows = [r for r in per_sample_rows if r['category'] == cat]
            vals = "".join(f"{mean([r[k] for r in rows]):>12.6f}" for k in metric_keys)
            summary_lines.append(f"{cat:<{col_w}}{vals}{len(rows):>6}")

    summary = "\n".join(summary_lines)
    print(summary)

    results_file = os.path.join(output_dir, 'evaluate_results.txt')
    with open(results_file, 'w') as f:
        f.write(summary + "\n")

    # ── Save meshes and skeletons ─────────────────────────────────────────────
    for (pred_joints, pred_bones, pred_root_index), (gt_joints, gt_bones, vertices, faces, transform_params, file_name, gt_root_index) in zip(pred_samples, gt_samples):
        pred_skel_filename = f'{output_dir}/{file_name}_skel_pred.obj'
        gt_skel_filename = f'{output_dir}/{file_name}_skel_gt.obj'
        mesh_filename = f'{output_dir}/{file_name}.obj'
        pred_rig_filename = f'{output_dir}/{file_name}_pred.txt'

        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()
        trans = transform_params[:3].cpu().numpy()
        scale = transform_params[3].cpu().numpy()
        pc_trans = transform_params[4:7].cpu().numpy()
        pc_scale = transform_params[7].cpu().numpy()

        pred_joints_denorm = pred_joints * pc_scale + pc_trans
        pred_joints_denorm = pred_joints_denorm / scale + trans
        save_skeleton_to_txt(pred_joints_denorm, pred_bones, pred_root_index, args.hier_order, vertices=vertices, filename=pred_rig_filename)

        if args.hier_order:
            save_skeleton_obj(pred_joints, pred_bones, pred_skel_filename, pred_root_index, use_cone=True)
        else:
            save_skeleton_obj(pred_joints, pred_bones, pred_skel_filename, use_cone=False)
        save_skeleton_obj(gt_joints, gt_bones, gt_skel_filename, gt_root_index, use_cone=True)

        vertices_norm = (vertices - trans) * scale
        vertices_norm = (vertices_norm - pc_trans) / pc_scale
        save_mesh(vertices_norm, faces, mesh_filename)

        if args.save_render:
            if args.hier_order:
                render_mesh_with_skeleton(pred_joints, pred_bones, vertices_norm, faces, output_dir, file_name, prefix='pred', root_idx=pred_root_index)
            else:
                render_mesh_with_skeleton(pred_joints, pred_bones, vertices_norm, faces, output_dir, file_name, prefix='pred')
            render_mesh_with_skeleton(gt_joints, gt_bones, vertices_norm, faces, output_dir, file_name, prefix='gt', root_idx=gt_root_index)
