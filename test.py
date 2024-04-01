#%%
import torch
t = torch.linspace(2,6,3).expand(3, 3)
print(t)
# Perturb sampling along each ray.
mid = (t[:, :-1] + t[:, 1:]) / 2.0
# %%
mid
# %%
t[:, :-1]
# %%
t[:, 1:]
# %%
print(t[:, :1])
lower = torch.cat((t[:, :1], -mid , t[:,:-1]), 1)
print(lower)

# %%
print(t[:, :1])
lower = torch.cat((mid, -mid ), 0)
print(lower)
# %%
print(t[:, :1])
lower = torch.cat((mid, -mid ), -1)
print(lower)

# %%
delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10]).expand(3, 1)), -1)
# %%
delta
# %%
from torchsummary import summary
import torch
from torchrl.modules import ConvNet
import torchvision.models as models
model= models.alexnet()
model.cuda()
# model=AlexNet()
# batch_size=16
#summary(model,input_size=(1,1,28,28))
summary(model,(3,64,64))

# %%
from tqdm import tqdm
from time import sleep

data_loader = list(range(1000))

for i , j  in enumerate(tqdm(s)):
    print("i",i , "j" , j)
    sleep(0.01)
# %%

import numpy as np
np.shape(data_loader)
# %%
data_loader.reshape(100,10)
# %%
s =np.array(data_loader).reshape(100,10)
s.shape
# %%
!pip install numpy-indexed
# %%
import numpy_indexed as npi
import numpy as np
# %%

random_array = np.array(range(400*400*10))

# Displaying a summary of the array
random_array.shape, random_array.dtype
# %%
groupby = npi.group_by(random_array)
# %%
print(npi.group_by(random_array).mean(), npi.group_by(random_array).sum()) # mean and sum of each group
# %%
npi.indices(random_array,that=200,axis=-1) # indices of each group

# %%
import numpy_indexed as npi
s=npi.group_by(random_array[163:231])
# %%
print(s)
# %%
import numpy as np
import numpy_indexed as npi

# Create a large random array
random_array = np.array(range(400*400*10))

# Create the repeated sequence of group keys
group_keys_pattern = np.arange(163, 232)  # This creates an array from 163 to 231
num_repeats = len(random_array) // 400
print(num_repeats)
group_keys = np.tile(group_keys_pattern, num_repeats)

# Ensure group_keys is the same length as random_array
group_keys = group_keys[:len(random_array)]
print(group_keys)
# Group by the keys and calculate the sum of each group
grouped_data = npi.group_by(random_array, group_keys).sum()

# Print the results
print(grouped_data)

# %%


arr=np.arange(30).reshape(10,3)
print(arr,"ARRAY")
s=npi.group_by(arr[:,2]).mean(arr);
print("S",s)
# %%
s

# %%
# Index you want to access
index = (100)
random_array
# Accessing the value at the specified index
value = random_array[index]

value

# %%

from itertools import chain
import numpy as np
seq = np.array(range(400*400*4))
step1=32
step2=400
step3=int(step2/2)
subt=int(step1/2)
addt=subt-1
#print(seq[i:i+step] for i in range(0, len(seq), step2))
result = list(chain.from_iterable(seq[i:i+step1] for i in range(step3-subt, len(seq), step2)))
print(result)
# %%
seq[1]
# %%
import time 
start_time = time.time()
import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from itertools import chain
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
seq = np.array(range(400*400*100))
seq1 = np.array(range(400*400*100))
step1=32
step2=400
step3=int(step2/2)
subt=int(step1/2)
addt=subt-1
range_start=step3-subt
#print(seq[i:i+step] for i in range(0, len(seq), step2))
result = list(chain.from_iterable(seq[i:i+step1] for i in range(range_start, len(seq), step2)))
#print(result)
seq2=seq1[result]
print("seq2",seq2)

step11=step1*step1
range_start1=(range_start)*step1
step21=range_start1*2+step11

result1 = list(chain.from_iterable(seq2[i:i+step11] for i in range(range_start1, len(seq2), step21)))
print("\n\n\n\n result1:",result1)

print("\n\n len-seq2:",len(seq2),":step21:",step21,":range_start1:",range_start1,":step11:",step11,":len-result1:",len(result1))
end_time = time.time()
print("time:",end_time-start_time)
# %%
