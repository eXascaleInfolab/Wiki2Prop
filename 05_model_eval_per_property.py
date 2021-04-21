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



if __name__== "__main__":


    torch.backends.cudnn.benchmark = True



    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)



    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--gpu', '-g',type= int, default=6)
    parser.add_argument('--model', '-m',type= int, default=0)



    args = parser.parse_args()

    model = args.model
    gpu = args.gpu

    year = "2018"
    embedding = "300LG"

    with_images= (model ==4)



    device = torch.device(gpu) if torch.cuda.device_count()>gpu  else torch.device(0)





    #Getting data
    logging.info("Loading Data")
    data = DataLoader(year = year, embedding=embedding,with_image = with_images)
    properties= data.get_properties()



    ID_test = data.get_test_index()

    _,Y_test = data.get_test()
    logging.info("Loaded Complete Test Set")
    data_name = ""
        


    logging.info("Data loaded")
    data.clean()





    nlabels = Y_test.shape[1]
    



    ####################
    #NETWORK DEF
    ####################

    thrn = st = ""


    if model ==0 :

        st = thrn =  "BPMLL"

    elif model==1 :

        st = thrn ="Baseline"

    elif model==2 :

        st = thrn = "Wiki2Prop"

    elif model==3 :

        st = thrn = "Wiki2Prop_EN"

    elif model == 4:
        
        
        st = thrn = "Wiki2Prop_IMAGE"

    elif model == 5 :
        st = thrn = "Recoin"


    thr = np.load(os.path.join("saves",thrn+"_thr_year{}_embedding{}.npy".format("2018","300LG")))

    version = st +"_year{}_embedding{}_".format(year,embedding) + data_name


    ROOT_PATH = os.path.join("//var","cache","fscache","audiffre")
    PATH_TO_CLEANED_DATA = os.path.join(ROOT_PATH, "cleaned_data")
    import pandas as pd
    filename = os.path.join(PATH_TO_CLEANED_DATA,"prediction_ranked_{}.feather").format(version)
    print("Loading prediction from {}".format(filename))
    y_pred = pd.read_feather(filename)

    y_pred = y_pred.drop(columns=['c_C_index'])
    y_pred = y_pred.values


    print( 'P = 1 : Test {}, Pred {}'.format(Y_test.shape,y_pred.shape))


    for P in [5,10,15,20,25]:


        version=st+"_"+str(P)

        is_selected = (np.sum(Y_test,axis=1)>= P)
        Y_test =Y_test[is_selected]
        y_pred = y_pred[is_selected]
        print( 'P = {} : Test {}, Pred {}'.format(P, Y_test.shape,y_pred.shape))
        n_test=len(Y_test)



        print("Computing error")
        errors = {}
        errors["N"]=n_test
        for m in METRICS :
            error_pred = m(y_pred, Y_test, threshold=thr )
            errors[m.__name__] = error_pred
            print(m.__name__,error_pred)




                

        try :
            scores = pickle.load(open("saves/scores_P_var_Oct_2020.pickle", "rb"))
        except :
            scores = {}
        scores[version] = errors
        pickle.dump(scores, open("saves/scores_P_var_Oct_2020.pickle", "wb"))
