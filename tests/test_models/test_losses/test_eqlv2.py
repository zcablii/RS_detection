"""Script:
python tests/test_models/test_losses/test_eqlv2.py
"""

import jittor as jt
from jdet.models.losses import EQLv2



labels = jt.Var([1,0,3,2,4])
print("labels:\n", labels)

cls_score = jt.Var(
   [[0.47881365, 0.420919  , 0.38578004, 0.8052165,  0.27370733],
    [0.19900735, 0.7165177 , 0.5128623,  0.6768294,  0.4716686 ],
    [0.33413434, 0.79596156, 0.7257531,  0.7326412,  0.5006631 ],
    [0.64515287, 0.08451196, 0.3924798,  0.40834078, 0.9835877 ],
    [0.1577293, 0.95628715, 0.3183534,  0.5656762,  0.3205039 ]]
)


print("original cls_score:\n", cls_score)
loss_cls = EQLv2(num_classes=4)
# bce = jt.nn.binary_cross_entropy_with_logits(cls_score, labels, size_average=False)
# print("bce:", bce)

# cls_score = jt.Var(
#    [0.47881365]
# )
# labels = jt.Var([1])
# bce = jt.nn.binary_cross_entropy_with_logits(cls_score, labels, size_average=False)
# print("bce:", bce)
# test EQLv2
print(loss_cls(cls_score, labels)) # 4.534896
