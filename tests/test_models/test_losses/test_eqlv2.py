"""Script:
python tests/test_models/test_losses/test_eqlv2.py
"""

from functools import partial
import imp
import numpy as np
import jittor as jt
from jdet.models.losses import EQLv2
from matplotlib import pyplot as plt

gamma = 12
mu    = 0.8
alpha = 4.0
def _func(x, gamma, mu):
    return 1 / (1 + np.exp(-gamma * (x - mu)))
map_func = partial(_func, gamma=gamma, mu=mu)

pos_neg = np.linspace(0, 2, 20)
neg_w = map_func(np.linspace(0, 2, 20))
pos_w = 1 + alpha * (1 - neg_w)
plt.plot(pos_neg, neg_w)
plt.plot(pos_neg, pos_w)
print("map_func(np.linspace(0, 2, 20):", map_func(np.linspace(0, 2, 20)))



num_classes = 4

labels = jt.Var([1,0,3,2,4])
print("labels:\n", labels)

cls_score = jt.Var(
   [[0.47881365, 0.420919  , 0.38578004, 0.8052165,  0.27370733],
    [0.19900735, 0.7165177 , 0.5128623,  0.6768294,  0.4716686 ],
    [0.33413434, 0.79596156, 0.7257531,  0.7326412,  0.5006631 ],
    [0.64515287, 0.08451196, 0.3924798,  0.40834078, 0.9835877 ],
    [0.1577293, 0.95628715, 0.3183534,  0.5656762,  0.3205039 ]]
)


# print("original cls_score:\n", cls_score)
loss_cls = EQLv2(num_classes=num_classes)

print("loss_cls 1:", loss_cls(cls_score, labels)) # 4.534896

# loss_cls.pos_grad = jt.rand(num_classes).stop_grad()
loss_cls.pos_grad = jt.Var([0.7910, 0.6721, 0.4647, 0.3477])
# loss_cls.neg_grad = jt.rand(num_classes).stop_grad()
loss_cls.neg_grad = jt.Var([0.3054, 0.9966, 0.7504, 0.2028])
# self.pos_neg = (jt.ones(self.num_classes) * 100).stop_grad()
loss_cls.pos_neg = jt.Var([0.0588, 0.2810, 0.2591, 0.9987])
loss_cls.pos_neg = jt.Var([0.0588, 0.2810, 0.2591, 0.9987])
# bce = jt.nn.binary_cross_entropy_with_logits(cls_score, labels, size_average=False)
# print("bce:", bce)

# cls_score = jt.Var(
#    [0.47881365]
# )
# labels = jt.Var([1])
# bce = jt.nn.binary_cross_entropy_with_logits(cls_score, labels, size_average=False)
# print("bce:", bce)
# test EQLv2
print("loss_cls 2:", loss_cls(cls_score, labels)) # 3.4176183
