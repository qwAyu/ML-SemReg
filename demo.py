import open3d as o3d
import torch
from tqdm import tqdm
import numpy as np
from easydict import EasyDict
from sklearn.neighbors import KDTree

from SemReg.group_match import query_local_signature, GroupingMatcher 
from SemReg.gss_constructor import grab_gss, get_anchor_pts_with_labels_outdoor
from SemReg.mask_matcher import maskmatcher_nn

from utils_demo import vis_pcr_pcd
from SC2PCR import SC2PCR
from utils.evaluator import PCRResultEvaluator

evaluator = PCRResultEvaluator(5, 0.6, 0.6)

from utils_demo import get_cmd_config, DataLoader


def get_config():
    config_lsig = EasyDict()
    config_lsig.radiu_local_sig = 0.8
    config_lsig.label_unuse = np.asarray([30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259])
    config_lsig.label_undefine = np.asarray([0])

    config_gss = EasyDict()
    config_gss.label_usefull = [16, 18, 80, 81, 10]
    config_gss.max_num_label = 8
    config_gss.N = 33
    config_gss.L = 1.5

    config_mask_match = EasyDict()
    config_mask_match.topx = 2
    
    return  config_lsig, config_gss, config_mask_match


def matcher_nn(desc_src, desc_dst):
    desc_src_tensor = torch.from_numpy(desc_src).cuda()
    desc_dst_tensor = torch.from_numpy(desc_dst).cuda()
    dis_matrix = torch.norm(desc_src_tensor[:, None, ...] - desc_dst_tensor[None, ...], dim=-1)
    idx_dst = torch.argmin(dis_matrix, dim=1).cpu()
    corr_torch = torch.column_stack([torch.arange(len(desc_src_tensor)).reshape(-1, 1), idx_dst.reshape(-1, 1)])
    corres = corr_torch.cpu().numpy()   
    return corres



def ml_SemReg(pts_src, pts_dst, label_src, label_dst, kpts_src, kpts_dst, desc_src, desc_dst, trans_gt, config_lsig, config_gss, config_mask_match):
    """
    e.g: all numpy ndarray
        pts_src (18944, 3) : source raw points, voxel downsample with 0.3m.
        kpts_src (5000, 3) : source keypoints, random select from src_pts
        desc_src (5000, 32) : desc descriptors of kpts_src
        label_src (18944,) : semantic label of pts_src (ref Pointcept)
        pts_dst (18944, 3) : target raw points, voxel downsample with 0.3m.
        kpts_dst (5000, 3) : target keypoints, random select from dst_pts
        desc_dst (5000, 32) : desc descriptors of kpts_dst
        label_dst (18944,) : semantic label of pts_dst (ref Pointcept)
        trans_gt (4, 4) : ground truth T    
    """

    # 1. NSS
    nss_src = query_local_signature(pts_src, label_src, kpts_src, radiu=config_lsig.radiu_local_sig)
    nss_dst = query_local_signature(pts_dst, label_dst, kpts_dst, radiu=config_lsig.radiu_local_sig)

    # 2. SCM er
    scmer = GroupingMatcher(nss_src, nss_dst, config_lsig=config_lsig)

    # 3. BMR-SS
    (
        is_use_gss,
        guidpost_pts_src, guidpost_label_src,
        guidpost_pts_dst, guidpost_label_dst, num_label_type
    ) = get_anchor_pts_with_labels_outdoor(pts_src, label_src, pts_dst, label_dst, **config_gss)

    if not is_use_gss:
        print("No BMRSS, skip it!")
        return

    gss_src = grab_gss(kpts_src, guidpost_pts_src,
                        guidpost_label_src, **config_gss, num_label_type=num_label_type)
    gss_dst = grab_gss(kpts_dst, guidpost_pts_dst,
                        guidpost_label_dst, **config_gss, num_label_type=num_label_type)

    # 4. match
    corres = scmer.run_match_based_on_gss(maskmatcher_nn, desc_src, desc_dst, gss_src, gss_dst, **config_mask_match)

    return corres


def main():
    config_lsig, config_gss, config_mask_match = get_config()
    cmd_config = get_cmd_config()
    print(cmd_config)

    ds = DataLoader()
    rslt_eval_rc = []
    for idx in tqdm(range(len(ds))):
        data = ds.get_item(idx)
        pts_src, label_src, kpts_src, desc_src, pts_dst, label_dst, kpts_dst, desc_dst, trans_gt = data
        evaluator.update_pair(pts_src, pts_dst, trans_gt=trans_gt)

        # matching
        if cmd_config.is_use_baseline_nn:
            corres = matcher_nn(desc_src, desc_dst)
        else:
            corres = ml_SemReg(pts_src, pts_dst, label_src, label_dst, kpts_src, kpts_dst, desc_src, desc_dst, trans_gt, config_lsig, config_gss, config_mask_match)
        
        # evaluation
        kpts_src = kpts_src[corres[:, 0]]
        kpts_dst = kpts_dst[corres[:, 1]]

        IN, IP = evaluator.eval_corr(kpts_src, kpts_dst, is_print=False)

        # reg sc2pcr
        reger = SC2PCR()
        trans_pred = reger.forward(kpts_src, kpts_dst)

        # vis
        if cmd_config.is_vis:
            vis_pcr_pcd(pts_src, pts_dst, trans_pred)

        # eval
        rmse, rre, rte, is_success = evaluator.eval_trans(trans=trans_pred)

        # 
        
        rslt_eval = [IN, IP, rmse, rre, rte, is_success]
        # print(rslt_eval)
        rslt_eval_rc.append(rslt_eval)

    
    rslt_eval_rc = np.asarray(rslt_eval_rc)
    mean_in = np.mean(rslt_eval_rc[:, 0])
    mean_ir = np.mean(rslt_eval_rc[:, 1]) * 100
    is_success = rslt_eval_rc[:, 5] == 1
    mean_re = np.mean(rslt_eval_rc[is_success, 3])
    mean_te = np.mean(rslt_eval_rc[is_success, 4]) * 100
    RR = np.mean(is_success) * 100
    np.mean(rslt_eval_rc, axis=0)
    print( 
        "Inlier Number: {:.3f}".format(mean_in), 
        "Inlier Ratio: {:.3f}%".format(mean_ir), 
        "Rotation Error: {:.3f} degree".format(mean_re), 
        "Translation Error: {:.3f} cm".format(mean_te), 
        "Success Rate: {:.3f}%".format(RR), sep="\n"
    )

if __name__=="__main__":
    main()

"""
python -m demo
python -m demo -is_use_baseline
python -m demo -is_vis
python -m demo -is_vis -is_use_baseline
"""