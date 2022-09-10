"""Script:
python tests/test_models/test_losses/test_seesaw_loss.py
"""

import jittor as jt
from jdet.models.losses import one_hot, SeesawLoss

N           = 5
use_sigmoid = False
p           = 0.8
q           = 2.0
num_classes = 3
eps         = 1e-2
reduction   = 'mean'
loss_weight = 1.0
return_dict = True

labels = jt.randint(0, num_classes, shape=(N,)) # (N,)
print("labels:\n", labels)

# test one_hot
coding = one_hot(labels, num_classes=3)
print("one-hot coding:\n", coding)

# test _split_cls_score
cls_score = jt.rand(N, num_classes+2) # (N, C+2)
print("original cls_score:\n", cls_score)
loss_cls = SeesawLoss(
    use_sigmoid=use_sigmoid,
    p=p,
    q=q,
    num_classes=num_classes,
    eps=eps,
    reduction=reduction,
    loss_weight=loss_weight,
    return_dict=return_dict)
cls_score_classes, cls_score_objectness = loss_cls._split_cls_score(cls_score)
print("splited cls_score_classes:\n", cls_score_classes)
print("splited cls_score_objectness:\n", cls_score_objectness)
scores = jt.concat([cls_score_classes, cls_score_objectness], dim=-1)
print("recovered scores:\n", scores)
assert (scores == cls_score).all()

# test SeesawLoss
print(loss_cls(cls_score, labels))


# test seesaw_ce_loss
# seesaw_ce_loss(cls_score,
#                labels,
#                label_weights,
#                cum_samples,
#                num_classes,
#                p,
#                q,
#                eps,
#                reduction='mean',
#                avg_factor=None):


# def test_seesaw_loss():
#     # only softmax version of Seesaw Loss is implemented
#     with pytest.raises(AssertionError):
#         loss_cfg = dict(type='SeesawLoss', use_sigmoid=True, loss_weight=1.0)
#         build_loss(loss_cfg)

#     # test that cls_score.size(-1) == num_classes + 2
#     loss_cls_cfg = dict(
#         type='SeesawLoss', p=0.0, q=0.0, loss_weight=1.0, num_classes=2)
#     loss_cls = build_loss(loss_cls_cfg)
#     # the length of fake_pred should be num_classes + 2 = 4
#     with pytest.raises(AssertionError):
#         fake_pred = torch.Tensor([[-100, 100]])
#         fake_label = torch.Tensor([1]).long()
#         loss_cls(fake_pred, fake_label)
#     # the length of fake_pred should be num_classes + 2 = 4
#     with pytest.raises(AssertionError):
#         fake_pred = torch.Tensor([[-100, 100, -100]])
#         fake_label = torch.Tensor([1]).long()
#         loss_cls(fake_pred, fake_label)

#     # test the calculation without p and q
#     loss_cls_cfg = dict(
#         type='SeesawLoss', p=0.0, q=0.0, loss_weight=1.0, num_classes=2)
#     loss_cls = build_loss(loss_cls_cfg)
#     fake_pred = torch.Tensor([[-100, 100, -100, 100]])
#     fake_label = torch.Tensor([1]).long()
#     loss = loss_cls(fake_pred, fake_label)
#     assert torch.allclose(loss['loss_cls_objectness'], torch.tensor(200.))
#     assert torch.allclose(loss['loss_cls_classes'], torch.tensor(0.))

#     # test the calculation with p and without q
#     loss_cls_cfg = dict(
#         type='SeesawLoss', p=1.0, q=0.0, loss_weight=1.0, num_classes=2)
#     loss_cls = build_loss(loss_cls_cfg)
#     fake_pred = torch.Tensor([[-100, 100, -100, 100]])
#     fake_label = torch.Tensor([0]).long()
#     loss_cls.cum_samples[0] = torch.exp(torch.Tensor([20]))
#     loss = loss_cls(fake_pred, fake_label)
#     assert torch.allclose(loss['loss_cls_objectness'], torch.tensor(200.))
#     assert torch.allclose(loss['loss_cls_classes'], torch.tensor(180.))

#     # test the calculation with q and without p
#     loss_cls_cfg = dict(
#         type='SeesawLoss', p=0.0, q=1.0, loss_weight=1.0, num_classes=2)
#     loss_cls = build_loss(loss_cls_cfg)
#     fake_pred = torch.Tensor([[-100, 100, -100, 100]])
#     fake_label = torch.Tensor([0]).long()
#     loss = loss_cls(fake_pred, fake_label)
#     assert torch.allclose(loss['loss_cls_objectness'], torch.tensor(200.))
#     assert torch.allclose(loss['loss_cls_classes'],
#                           torch.tensor(200.) + torch.tensor(100.).log())

#     # test the others
#     loss_cls_cfg = dict(
#         type='SeesawLoss',
#         p=0.0,
#         q=1.0,
#         loss_weight=1.0,
#         num_classes=2,
#         return_dict=False)
#     loss_cls = build_loss(loss_cls_cfg)
#     fake_pred = torch.Tensor([[100, -100, 100, -100]])
#     fake_label = torch.Tensor([0]).long()
#     loss = loss_cls(fake_pred, fake_label)
#     acc = loss_cls.get_accuracy(fake_pred, fake_label)
#     act = loss_cls.get_activation(fake_pred)
#     assert torch.allclose(loss, torch.tensor(0.))
#     assert torch.allclose(acc['acc_objectness'], torch.tensor(100.))
#     assert torch.allclose(acc['acc_classes'], torch.tensor(100.))
#     assert torch.allclose(act, torch.tensor([1., 0., 0.]))
