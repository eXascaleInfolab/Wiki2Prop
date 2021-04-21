import argparse
import pickle
import logging
import operator
import torch
from tools import mlauc, METRICS
import os
import time
import numpy as np
from data_loader import DataLoader

torch.backends.cudnn.benchmark = True

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)



parser = argparse.ArgumentParser(description='select args')
parser.add_argument('--gpu', '-g',type= int, default=6)
parser.add_argument('--model', '-m',type= int, default=0)
parser.add_argument('--save', '-s',type= int, default=0)
parser.add_argument('--targetset', '-t',type= int, default=0)
parser.add_argument("--year","-y",type= int, default=0)  # 0 : 2018, 1 : 2019
parser.add_argument("--embedding","-e",type= int, default=0)  # 0 : 300LG , 1 : 300NLG, 2: 100LG
parser.add_argument("--deploy","-d",type= int, default=0)  # 0 : regular train/test,  1: Deply mode : train on train+test

args = parser.parse_args()

model = args.model
gpu = args.gpu
save_prediction = args.save
target_set =args.targetset
year = "2018" if args.year ==0 else "2019"
if args.embedding == 0:
    embedding = "300LG"
elif args.embedding == 1:
    embedding = "300NLG"
elif args.embedding == 2:
    embedding = "100LG"


with_images= (model ==4)



TRAINING_DEPLOY = (args.deploy == 1)
if TRAINING_DEPLOY : 
    logging.info("WARNING : DEPLOY MODE TRAINING. NO TEST SET.")
    if save_prediction==0 :
        logging.info("ERROR : COMPUTING METRICS ON DEPLOY MODE IS NOT ALLOWED")
        raise Exception()




device = torch.device(gpu) if torch.cuda.device_count()>gpu  else torch.device(0)





#Getting data
logging.info("Loading Data")
data = DataLoader(year = year, embedding=embedding,with_image = with_images)
properties= data.get_properties()

if TRAINING_DEPLOY:
    X_test, Y_test, ID_test = data.get_all()
    logging.info("Loaded Complete DataSet")
    data_name = ""

else :
    #ID_test = np.load(os.path.join("..", "data",'cleaned_data',"indexes_test.npy"))
    ID_test = data.get_test_index()
    if target_set == 0 : #Whole test set
        X_test,Y_test = data.get_test()
        logging.info("Loaded Complete Test Set")
        data_name = ""
    if target_set == 1 : #TopClass Subset
        X_test, Y_test = data.get_topclass()
        logging.info("Loaded TopClass set")
        data_name = 'TopClass'

    if target_set == 2 : #Q
        X_test, Y_test = data.get_q(7270702)
        logging.info("Loaded Q set")
        data_name = 'Q7270702'


    if (model !=2) and (model !=4) :
        X_test = X_test[:,:300]


logging.info("Data loaded")
data.clean()





nlabels = Y_test.shape[1]
ndim = X_test.shape[1]




####################
#NETWORK DEF
####################

if model ==0 :
    from BPMLL_train import BPMLL
    ccn = BPMLL(nlabels)
    st = thrn =  "BPMLL"

elif model==1 :
    from Baseline_train import BaseLineML
    ccn = BaseLineML(nlabels,ndim)
    st = thrn ="Baseline"

elif model==2 :
    from Wiki2Prop_train_final import Wiki2Prop
    ccn  = Wiki2Prop(nlabels)
    st = thrn = "Wiki2Prop"

elif model==3 :
    from Wiki2Prop_train_partial import Wiki2Prop_Part
    ccn  = Wiki2Prop_Part(nlabels,"EN")
    st = thrn = "Wiki2Prop_EN"

elif model == 4:
    from Wiki2Prop_train_final import Wiki2Prop
    ccn  = Wiki2Prop(nlabels,with_img = with_images)
    st = thrn = "Wiki2Prop_IMAGE"

if TRAINING_DEPLOY :
    version = st +"DEPLOY_year{}_embedding{}.pt".format(year,embedding) + data_name
    ccn.load_state_dict(torch.load(os.path.join('saves',version.replace("IMAGE",''))))
  

else : 
    version = st +"_year{}_embedding{}_".format(year,embedding) + data_name
    ccn.load_state_dict(torch.load(os.path.join('saves',st+"_year{}_embedding{}.pt".format(year,embedding))))
    thr = np.load(os.path.join("saves",thrn+"_thr_year{}_embedding{}.npy".format(year,embedding)))


ccn.eval()
ccn.to(device)





print("launching XP eval for {}".format(version))

n_test = len(X_test)

nb = (len(X_test)//100)  
y_t= np.zeros_like(Y_test,dtype=float)
for i in range(nb):
    print("Done {}%".format(10*i),end="\r")
    features_ = torch.Tensor(X_test[i*nb:(i+1)*nb]).to(device)
    y_ = ccn(features_).cpu().detach().numpy()
    y_t[i*nb:(i+1)*nb] = y_

del features_,y_
  
y_true = (Y_test )


if save_prediction :
    ROOT_PATH = os.path.join("//var","cache","fscache","audiffre")
    PATH_TO_CLEANED_DATA = os.path.join(ROOT_PATH, "cleaned_data")
    import pandas as pd
    filename = os.path.join(PATH_TO_CLEANED_DATA,"prediction_ranked_{}.feather").format(version)
    print("Saving prediction to {}".format(filename))
    y_pred = pd.DataFrame(data = y_t, index=ID_test, columns = properties)
    y_pred = y_pred.reset_index()
    y_pred.columns = [ "c_{}".format(c) for c in y_pred.columns]
    y_pred.to_feather(filename)

    #filename = os.path.join("..","data","prediction_{}.pkl.bz2").format(version)
    #y_pred = pd.DataFrame(data = y_t> thr, index=ID_test, columns = properties)
    #y_pred.to_pickle(filename)
else :

    print("Computing error")
    errors = {}
    for m in METRICS :
        error_pred = m(y_t, y_true, threshold=thr )
        errors[m.__name__] = error_pred
        print(m.__name__,error_pred)


    errors["N"]=n_test
    m=mlauc
    try :
        error_pred = m(y_t, y_true, threshold=thr )
        error_pred = np.mean(error_pred)
    except ValueError :
        error_pred = None

        errors[m.__name__] = error_pred
        print(m.__name__,error_pred)




    ans = input("Should I write the results for {}?".format(version))
    if (len(ans)>0) & (int(ans)==1):
        try :
            scores = pickle.load(open("saves/scores_Nov_2019.pickle", "rb"))
        except :
            scores = {}
        scores[version] = errors
        pickle.dump(scores, open("saves/scores_Nov_2019.pickle", "wb"))
