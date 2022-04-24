
"""## Hyperparameters specification"""

#GPU selection 
information_patches=0.05
filter_Target=True#@param {type:"boolean"}
filter_Source=True#@param {type:"boolean"}
pretrain_model1=False#@param {type:"boolean"}
GPU_availability=True#@param {type:"boolean"}
show_images=False#@param {type:"boolean"}
GPU="0"#Perform and nvidia-smi to check which free gpus are available 
#Parameters to be modified:
plot_history=False#@param {type:"boolean"}
factor=4 #@param {type:"integer"}
noise=0.3 #@param {type:"number"}
factor_patches=1 #@param {type:"integer"}
random_patches=True#@param {type:"boolean"}
# === PreTraining parameters ===
# number of epochs
numEpochsPretrain =  200#@param {type:"integer"}
# patience
patiencePretrain =  200#@param {type:"integer"}
# learning rate
lrPretrain = 5e-4 #@param {type:"number"}
# batch size
batch_size_valuePretrain =  6#@param {type:"integer"}
# use one-cycle policy for super-convergence? Reduce on plateau?
no_schedule = None #@param {type:"raw"}
schedulePretrain = 'oneCycle' #@param [ "no_schedule","'oneCycle'","'reduce'"] {type:"raw"}

# Network architecture: UNet, ResUNet,MobileNetEncoder
model_namePretrain = 'AttentionUNET'#@param ['UNet','MobileNetEncoder','AttentionUNET']
# Optimizer name: 'Adam', 'SGD'
optimizer_namePretrain = 'Adam'#@param ['Adam','SGD']{type:"string"}
# Loss function name: 'BCE', 'Dice', 'W_BCE_Dice'
loss_acronymPretrain = 'mse' #@param ['mae','mse']{type:"string"}
max_poolingPretrain=True #@param {type:"boolean"}

#@title **Training Hyperparameters**

# === Training parameters ===
# number of epochs
numEpochs =  60#@param {type:"integer"}
# patience
patience = 60#@param {type:"integer"}
# learning rate
lr =1e-4#@param {type:"number"}
# batch size
batch_size_value = 5#@param {type:"integer"}
# use one-cycle policy for super-convergence? Reduce on plateau?
schedule = 'oneCycle' #@param [ "no_schedule","'oneCycle'","'reduce'"] {type:"raw"}
# Network architecture: UNet, ResUNet,MobileNetEncoder
model_name = 'AttentionUNET' #@param ['UNet','MobileNetEncoder','AttentionUNET']
# Optimizer name: 'Adam', 'SGD'
optimizer_name = 'Adam' #@param ['Adam','SGD']{type:"string"}
# Loss function name: 'BCE', 'Dice', 'W_BCE_Dice'
loss_acronym = 'BCE' #@param ['BCE','Dice','SEG']{type:"string"}
# create the network and compile it with its optimizer
max_pooling=True #@param {type:"boolean"}

repetitions=10#@param {type:"slider", min:1, max:30, step:1}
train_encoder=False #@param {type:"boolean"}
bottleneck_freezing=False #@param {type:"boolean"}
train_decoder=True
#Select dataset-route {'Lucchi++','Kasthuri++','Achucarro','VNC'}
#
Target='Lucchi++'#@param ['Lucchi++','Kasthuri++','Achucarro','VNC']
Source='Kasthuri++'#@param ['Lucchi++','Kasthuri++','Achucarro','VNC']
noisy_input = False #@param {type:"boolean"}
histogram_matching=False
if histogram_matching:
    hm='hm'
else:
    hm=''
testName=Target+'_'+Source+'_'+model_name+'_'+hm 


train_input_path1 = '/home/jpastor/Downloads/Datasets/'+Target+'/train/x'
train_label_path1 = '/home/jpastor/Downloads/Datasets/'+Target+'/train/y'
test_input_path1 = '/home/jpastor/Downloads/Datasets/'+Target+'/test/x'
test_label_path1 = '/home/jpastor/Downloads/Datasets/'+Target+'/test/y'

train_input_path2 = '/home/jpastor/Downloads/Datasets/'+Source+'/train/x'
train_label_path2 = '/home/jpastor/Downloads/Datasets/'+Source+'/train/y'
test_input_path2 = '/home/jpastor/Downloads/Datasets/'+Source+'/test/x'
test_label_path2 = '/home/jpastor/Downloads/Datasets/'+Source+'/test/y'

train_input_path1_hm = '/home/jpastor/Downloads/datasets-hist-matching/hist_match/'+Target+'/'+Target+'_s-t_'+Source+'/train/x'
train_label_path1_hm = '/home/jpastor/Downloads/datasets-hist-matching/hist_match/'+Target+'/'+Target+'_s-t_'+Source+'/train/y'
test_input_path1_hm = '/home/jpastor/Downloads/datasets-hist-matching/hist_match/'+Target+'/'+Target+'_s-t_'+Source+'/test/x'
test_label_path1_hm = '/home/jpastor/Downloads/datasets-hist-matching/hist_match/'+Target+'/'+Target+'_s-t_'+Source+'/test/y'

train_input_path2_hm = '/home/jpastor/Downloads/datasets-hist-matching/hist_match/'+Source+'/'+Source+'_s-t_'+Target+'/train/x'
train_label_path2_hm = '/home/jpastor/Downloads/datasets-hist-matching/hist_match/'+Source+'/'+Source+'_s-t_'+Target+'/train/y'
test_input_path2_hm = '/home/jpastor/Downloads/datasets-hist-matching/hist_match/'+Source+'/'+Source+'_s-t_'+Target+'/test/x'
test_label_path2_hm = '/home/jpastor/Downloads/datasets-hist-matching/hist_match/'+Source+'/'+Source+'_s-t_'+Target+'/test/y'

import pandas as pd
import numpy as np
hyperparameters=[['Source','Target','Epochs Pre','patience Pre','lr Pre','batch size Pre','scheduler pre','model','optimizer pre','loss pre','Epochs','Patience','Lr','Batch size','Scheduler','Optimizer','Iterations','Train Encoder','BottleNeck freezing','Train decoder'],[Source,Target,numEpochsPretrain,patiencePretrain,lrPretrain,batch_size_valuePretrain,schedulePretrain,model_name,optimizer_namePretrain,loss_acronymPretrain,numEpochs,patience,lr,batch_size_value,schedule,optimizer_name,repetitions,train_encoder,bottleneck_freezing,train_decoder]]
df=pd.DataFrame(hyperparameters)
new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = new_header 
print(df)
