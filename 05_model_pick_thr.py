import argparse
import pickle
import logging
import operator
import torch
from tools import mlauc, METRICS, auc_thr, exam_f1,exam_recall, exam_precision, binary_f1
import os
import time
import numpy as np
from data_loader import DataLoader


torch.backends.cudnn.benchmark = True

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)



parser = argparse.ArgumentParser(description='select args')
parser.add_argument('--gpu', '-g',type= int, default=6)
parser.add_argument('--model', '-m',type= int, default=0)
parser.add_argument("--year","-y",type= int, default=0)  # 0 : 2018, 1 : 2019
parser.add_argument("--embedding","-e",type= int, default=0)  # 0 : 300LG , 1 : 300NLG, 2: 100LG
args = parser.parse_args()

gpu = args.gpu
model = args.model
year = "2018" if args.year ==0 else "2019"
if args.embedding == 0:
    embedding = "300LG"
elif args.embedding == 1:
    embedding = "300NLG"
elif args.embedding == 2:
    embedding = "100LG"



device = torch.device(gpu) if torch.cuda.device_count()>gpu  else torch.device(0)





with_images= (model ==4)


#Getting data
logging.info("Loading Data")
data = DataLoader(year = year, embedding=embedding,with_image = with_images)
X_train, Y_train = data.get_train()
if (model !=2) and (model !=4) :
    X_train = X_train[:,:300]

logging.info("Data loaded")


nlabels = Y_train.shape[1]
ndim = X_train.shape[1]


    


####################
#NETWORK DEF
####################
if model ==0 :
    from BPMLL_train import BPMLL
    ccn = BPMLL(nlabels)
    st = "BPMLL"

elif model==1 :
    from Baseline_train import BaseLineML
    ccn = BaseLineML(nlabels,ndim)
    st = "Baseline"

elif model == 2:
    from Wiki2Prop_train_final import Wiki2Prop
    ccn  = Wiki2Prop(nlabels)
    st ="Wiki2Prop"
elif model==3 :
    from Wiki2Prop_train_partial import Wiki2Prop_Part
    ccn  = Wiki2Prop_Part(nlabels,"EN")
    st = thrn = "Wiki2Prop_EN"
elif model == 4:
    from Wiki2Prop_train_final import Wiki2Prop
    ccn  = Wiki2Prop(nlabels,with_img = with_images)
    st ="Wiki2Prop_IMAGE"
    
ccn.load_state_dict(torch.load(os.path.join('saves', st+"_year{}_embedding{}.pt".format(year,embedding))))


print("Computing image for {}".format(ccn.__class__.__name__))


ccn.eval()
ccn.to(device)



#Pick best Thr:

nb = (len(X_train)//100)  
y_t= np.zeros_like(Y_train,dtype=float)
for i in range(101):
    features_ = torch.Tensor(X_train[i*nb:(i+1)*nb]).to(device)
    y_ = ccn(features_).cpu().detach().numpy()
    y_t[i*nb:(i+1)*nb] = y_

del features_,y_
y_=y_t 
#best_thr = auc_thr(y_,Y_train)

possible_thresholds = list(np.arange(0.1,1,0.1))+list(np.arange(0.3,0.5,0.02))



print("picking thr for {}".format(ccn.__class__.__name__))

bthr=.5
bf1=0
for ip,thrp in enumerate(possible_thresholds):
    recall = exam_recall(y_,Y_train,threshold=thrp)
    precision = exam_precision(y_,Y_train,threshold=thrp)
    f1 = exam_f1(y_,Y_train,threshold=thrp)
    print("{} - value {:.2f}, Precision {:.2f}, Recall {:.2f},F1 {:.2f}".format(ip, thrp, precision, recall, f1))
    if f1 >bf1 :
        bf1 = f1
        bthr = thrp

print("selected thr : {} (F1 : {})".format(bthr, bf1))
np.save(os.path.join('saves', st+"_thr_year{}_embedding{}.npy".format(year,embedding)),bthr)

# else :

#     bthresholds = .5 * np.ones(Y_train.shape[1])
#     bf1=1e-4* np.ones(Y_train.shape[1])

#     for thrp in possible_thresholds:
#         f1 = binary_f1(y_,Y_train,threshold=thrp)
#         bthresholds[f1>bf1] = thrp
#         bf1[f1>bf1] = f1[f1>bf1]
    
#     thresholds = bthresholds[None,:]

#     print("selected thr : {}".format(thresholds))
#     np.save(os.path.join("saves", st+"_complex_thr.npy"), thresholds)