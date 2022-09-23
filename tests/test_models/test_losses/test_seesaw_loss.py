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

N           = 5
use_sigmoid = False
p           = 0.8
q           = 2.0
num_classes = 3
eps         = 1e-2
reduction   = 'mean'
loss_weight = 1.0
return_dict = True

labels = jt.Var([0, 0, 1, 1, 0])
print("labels:\n", labels)

# test _split_cls_score
cls_score = jt.Var(
   [[0.47881365, 0.420919  , 0.38578004, 0.8052165,  0.27370733],
    [0.19900735, 0.7165177 , 0.5128623,  0.6768294,  0.4716686 ],
    [0.33413434, 0.79596156, 0.7257531,  0.7326412,  0.5006631 ],
    [0.64515287, 0.08451196, 0.3924798,  0.40834078, 0.9835877 ],
    [0.1577293, 0.95628715, 0.3183534,  0.5656762,  0.3205039 ]]
)


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

# test SeesawLoss
print(loss_cls(cls_score, labels))