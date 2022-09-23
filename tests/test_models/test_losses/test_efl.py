"""Script:
python tests/test_models/test_losses/test_efl.py
"""

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import jittor as jt

N = 15
num_classes = 10
label = jt.randint(low=0, high=num_classes+1, shape=(N,))
# print("label:", label)

# z = np.array([1.0, 2.0, 3.0, 40.0,1.0, 2.0, 3.0, 4.0,1.0, 2.0])
z = np.array([262155, 36581, 105107, 238, 2327, 622, 7436, 1098, 18, 11837]) / 1000
T = 10
softmax_z = np.exp(z/T) / np.sum(np.exp(z/T))
linear_z = z / np.sum(z)
print("softmax: ", softmax_z)
print("linear_z:", linear_z)
# print("softmax: ", np.exp(np.mean(z)/T) / np.sum(np.exp(z/T)))

acc_loss = np.array([262155, 36581, 105107, 238, 2327, 622, 7436, 1098, 18, 11837])
nums = np.array([10671, 8689, 66017, 394, 731, 236, 252, 1549, 136, 311])
print("acc_loss / nums:", acc_loss / nums)