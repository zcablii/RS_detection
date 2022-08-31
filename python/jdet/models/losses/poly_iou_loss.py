import jittor as jt 
from jittor import nn 

from jdet.utils.registry import LOSSES
from jittor.nn import bmm as bmm
from copy import deepcopy


from jdet.ops.convex_sort import convex_sort
from jdet.ops.bbox_transforms import bbox2type, get_bbox_areas


def shoelace(pts):
    roll_pts = jt.roll(pts, 1, dims=-2)
    xyxy = pts[..., 0] * roll_pts[..., 1] - \
           roll_pts[..., 0] * pts[..., 1]
    areas = 0.5 * jt.abs(xyxy.sum(dim=-1))
    return areas


def convex_areas(pts, masks):
    nbs, npts, _ = pts.size()
    index = convex_sort(pts, masks)
    index[index == -1] = npts
    index = index[..., None].repeat(1, 1, 2)

    ext_zeros = jt.zeros((nbs, 1, 2),dtype=pts.dtype)
    ext_pts = jt.concat([pts, ext_zeros], dim=1)
    polys = jt.gather(ext_pts, 1, index)

    xyxy_1 = (polys[:, 0:-1, 0] * polys[:, 1:, 1])
    xyxy_2 = (polys[:, 0:-1, 1] * polys[:, 1:, 0])
    xyxy_1.sync()
    xyxy_2.sync()

    xyxy = xyxy_1 - xyxy_2
    areas = 0.5 * jt.abs(xyxy.sum(dim=-1))
    return areas


def poly_intersection(pts1, pts2, areas1=None, areas2=None, eps=1e-6):
    # Calculate the intersection points and the mask of whether points is inside the lines.
    # Reference:
    #    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    #    https://github.com/lilanxiao/Rotated_IoU/blob/master/box_intersection_2d.py
    lines1 = jt.concat([pts1, jt.roll(pts1, -1, dims=1)], dim=2)
    lines2 = jt.concat([pts2, jt.roll(pts2, -1, dims=1)], dim=2)
    lines1, lines2 = lines1.unsqueeze(2), lines2.unsqueeze(1)
    x1, y1, x2, y2 = lines1.unbind(dim=-1) # dim: N, 4, 1
    x3, y3, x4, y4 = lines2.unbind(dim=-1) # dim: N, 1, 4

    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    with jt.no_grad():
        den_u = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
        t, u = den_t / num, den_u / num
        mask_t = (t > 0) & (t < 1)
        mask_u = (u > 0) & (u < 1)
        mask_inter = jt.logical_and(mask_t, mask_u)

    t = den_t / (num + eps)
    x_inter = x1 + t * (x2 - x1)
    y_inter = y1 + t * (y2 - y1)
    pts_inter = jt.stack([x_inter, y_inter], dim=-1)

    B = pts1.size(0)
    pts_inter = pts_inter.view(B, -1, 2)
    mask_inter = mask_inter.view(B, -1)

    # Judge if one polygon's vertices are inside another polygon.
    # Use
    with jt.no_grad():
        areas1 = shoelace(pts1) if areas1 is None else areas1
        areas2 = shoelace(pts2) if areas2 is None else areas2

        triangle_areas1 = 0.5 * jt.abs(
            (x3 - x1) * (y4 - y1) - (y3 - y1) * (x4 - x1))
        sum_areas1 = triangle_areas1.sum(dim=-1)
        mask_inside1 = jt.abs(sum_areas1 - areas2[..., None]) < 1e-3 * areas2[..., None]

        triangle_areas2 = 0.5 * jt.abs(
            (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))
        sum_areas2 = triangle_areas2.sum(dim=-2)
        mask_inside2 = jt.abs(sum_areas2 - areas1[..., None]) < 1e-3 * areas1[..., None]

    all_pts = jt.concat([pts_inter, pts1, pts2], dim=1)
    masks = jt.concat([mask_inter, mask_inside1, mask_inside2], dim=1)
    return all_pts, masks


def poly_enclose(pts1, pts2):
    all_pts = jt.concat([pts1, pts2], dim=1)
    mask1 = pts1.new_ones((pts1.size(0), pts1.size(1)))
    mask2 = pts2.new_ones((pts2.size(0), pts2.size(1)))
    masks = jt.concat([mask1, mask2], dim=1)
    return all_pts, masks


def poly_iou_loss(pred, target, linear=False, eps=1e-6,weight=None, reduction='mean', avg_factor=None):
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    pred, target = bbox2type(pred, 'poly'), bbox2type(target, 'poly')

    pred_pts = pred.view(pred.size(0), -1, 2)
    target_pts = target.view(target.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(
        pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    ious = (overlap / (areas1 + areas2 - overlap + eps)).clamp(min_v=eps)
    if linear:
        loss = 1 - ious
    else:
        loss = -ious.log()

    if weight is not None:
        loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/avg_factor
    return loss


def poly_giou_loss(pred, target, eps=1e-6, weight=None, reduction='mean', avg_factor=None):
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    pred, target = bbox2type(pred, 'poly'), bbox2type(target, 'poly')

    pred_pts = pred.view(pred.size(0), -1, 2)
    target_pts = target.view(target.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(
        pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    union = areas1 + areas2 - overlap + eps
    ious = (overlap / union).clamp(min=eps)

    enclose_pts, enclose_masks = poly_enclose(pred_pts, target_pts)
    enclose_areas = convex_areas(enclose_pts, enclose_masks)

    gious = ious - (enclose_areas - union) / enclose_areas
    loss = 1 - gious

    if weight is not None:
        loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/avg_factor
    return loss


@LOSSES.register_module()
class PolyIoULoss(nn.Module):

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(PolyIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if weight is not None and weight.ndim > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * poly_iou_loss(
            pred,
            target,
            weight=weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class PolyGIoULoss(nn.Module):

    def __init__(self,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(PolyGIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not jt.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.ndim > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * poly_giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min_v=1e-7, max_v=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = jt.cos(r)
    sin_r = jt.sin(r)
    R = jt.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * jt.stack([jt.misc.diag(wh[i], diagonal=0) for i in range(wh.shape[0])]).reshape(-1, 2, 2)
    sigma = bmm(bmm(R,S*S),R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def postprocess(distance, fun='log1p', tau=1.0):

    if fun == 'log1p':
        distance = jt.log(distance+1)
    elif fun == 'sqrt':
        distance = jt.sqrt(distance.clamp(min_v=1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


def kld_loss_v0(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True, weight=None, reduction='mean', avg_factor=None):

    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target
    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)
    Sigma_p_inv = jt.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2) 
    
    Sigma_p_inv = Sigma_p_inv / jt.linalg.det(Sigma_p).unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * bmm(bmm(dxy.permute(0, 2, 1),(Sigma_p_inv)), dxy).view(-1)

    whr_distance = 0.5 * bmm(Sigma_p_inv,Sigma_t)
    whr_distance = jt.stack([jt.misc.diag(whr_distance[i]) for i in range(whr_distance.shape[0])]).sum(dim=-1)


    Sigma_p_det_log = jt.linalg.det(Sigma_p).log()
    Sigma_t_det_log = jt.linalg.det(Sigma_t).log()

    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(min_v=1e-7).sqrt()

    distance = distance.reshape(_shape[:-1])

    loss = postprocess(distance, fun=fun, tau=tau)
  
    if weight is not None:
        loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/avg_factor
    return loss




def kld_loss(pred, target, fun='log1p', tau=1.0, weight=None, reduction='mean', avg_factor=None):
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    
    delta = (xy_p - xy_t).unsqueeze(-1)
    sigma_t_inv = jt.linalg.inv(Sigma_t)
    term1 = delta.transpose(-1,
                            -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)
    x_ = sigma_t_inv.matmul(Sigma_p)
    term2_ = jt.stack([jt.misc.diag(x_[i]) for i in range(x_.shape[0])]).sum(dim=-1).reshape(-1,1)
    term2 = term2_ + jt.log(jt.linalg.det(Sigma_t) / jt.linalg.det(Sigma_p)).reshape(-1, 1)
    
    dis = term1 + term2 - 2
    kl_dis = dis.clamp(min_v=1e-6)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + jt.sqrt(kl_dis))
    else:
        loss = 1 - 1 / (tau + jt.log(kl_dis+1))
    # return kl_loss
 
    if weight is not None:
        loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        loss = loss.sum()/avg_factor
        # print('jt.grad(loss, Sigma_p)', jt.grad(loss, Sigma_p))
        return loss
    return loss


def gwd_loss_v0(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True, weight=None, reduction='mean', avg_factor=None):
    """Gaussian Wasserstein distance loss.

    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_distance = ((xy_p - xy_t)*(xy_p - xy_t)).sum(dim=-1)
    
    whr_distance = jt.stack([jt.misc.diag(Sigma_p[i]) for i in range(Sigma_p.shape[0])]).sum(dim=-1)
    whr_distance = whr_distance + jt.stack([jt.misc.diag(Sigma_t[i]) for i in range(Sigma_t.shape[0])]).sum(dim=-1)
    x_ = bmm(Sigma_p,Sigma_t)
    _t_tr =jt.stack([jt.misc.diag(x_[i]) for i in range(x_.shape[0])]).sum(dim=-1)
    _t_det_sqrt = (jt.linalg.det(Sigma_p) * jt.linalg.det(Sigma_t)).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())
    distance = (xy_distance + alpha * alpha * whr_distance).clamp(1e-7).sqrt()
    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    
    loss = postprocess(distance, fun=fun, tau=tau)

    
    if weight is not None:
        loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/avg_factor
    return loss

def gwd_loss(pred, target, fun='sqrt', tau=2.0, normalize=True, weight=None, reduction='mean', avg_factor=None):
    """Gaussian Wasserstein distance loss.
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_distance = ((xy_p - xy_t)*(xy_p - xy_t)).sum(dim=-1)

    whr_distance = jt.stack([jt.misc.diag(Sigma_p[i]) for i in range(Sigma_p.shape[0])]).sum(dim=-1)
    whr_distance = whr_distance + jt.stack([jt.misc.diag(Sigma_t[i]) for i in range(Sigma_t.shape[0])]).sum(dim=-1)
    x_ = bmm(Sigma_p,Sigma_t)
    _t_tr =jt.stack([jt.misc.diag(x_[i]) for i in range(x_.shape[0])]).sum(dim=-1)
    _t_det_sqrt = (jt.linalg.det(Sigma_p) * jt.linalg.det(Sigma_t)).clamp(0).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())
    distance = xy_distance + whr_distance
    gwd_dis = distance.clamp(min_v=1e-6)
    if fun == 'sqrt':
        loss = 1 - 1 / (tau + jt.sqrt(gwd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + jt.log(gwd_dis+1))
    else:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
        loss = jt.log1p(jt.sqrt(gwd_dis) / scale)
    
    if weight is not None:
        loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/avg_factor
    
    return loss

@LOSSES.register_module()
class GDLoss(nn.Module):

    BAG_GD_LOSS = {
        'gwd_v0': gwd_loss_v0,
        'kld_v0': kld_loss_v0,
        'gwd': gwd_loss,
        'kld': kld_loss,
    }
    BAG_PREP = {
        # 'xy_stddev_pearson': xy_stddev_pearson_2_xy_sigma,
        'xy_wh_r': xy_wh_r_2_xy_sigma
    }

    def __init__(self,
                 loss_type,
                 representation='xy_wh_r',
                 fun='log1p',
                 tau=1.0,
                 alpha=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'none', 'sqrt']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = self.BAG_PREP[representation]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not jt.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and len(weight.shape) > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        pred_ = self.preprocess(pred)
    
        target_ = self.preprocess(target)
        loss = self.loss(
            pred_,
            target_,
            fun=self.fun,
            tau=self.tau,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            **_kwargs) * self.loss_weight
        # print('loss',loss)
        return loss
