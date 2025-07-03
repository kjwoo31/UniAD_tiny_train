import torch, math

def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


def denormalize_bbox_trt(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = custom_torch_atan2_trt(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()

    return if_normal_bboxes_size(normalized_bboxes, cx, cy, cz, w, l, h, rot)

def if_normal_bboxes_size(normalized_bboxes, cx, cy, cz, w, l, h, rot):
    vx = normalized_bboxes[:, 8:9]
    vy = normalized_bboxes[:, 9:10]
    denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    return denormalized_bboxes

def custom_torch_atan2_trt(y, x):
    '''
    reference: https://en.wikipedia.org/wiki/Atan2
    '''
    eps = 1e-8
    atan = torch.atan(y/(x+eps))
    x_eq_0 = x==0
    x_gt_0 = x>0
    x_ls_0 = x<0
    y_ge_0 = y>=0
    y_gt_0 = y>0
    y_ls_0 = y<0

    pi_div_2 = (torch.ones(atan.shape).to(atan.device))*(math.pi/2)
    negative_pi_div_2 = (torch.ones(atan.shape).to(atan.device))*(-math.pi/2)

    atan2 = (negative_pi_div_2)*(x_eq_0 & y_ls_0).int()\
            + (pi_div_2)*(x_eq_0 & y_gt_0).int()\
            + (atan-math.pi)*(x_ls_0 & y_ls_0).int()\
            + (atan+math.pi)*(x_ls_0 & y_ge_0).int()\
            + (atan) * x_gt_0.int()

    return atan2.float()