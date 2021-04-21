import pickle
import logging
from tools import  METRICS, exam_f1
import os
import pandas as pd
import numpy as np
from data_loader import DataLoader



if __name__== "__main__":


    



    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)



    logging.info("Loading Data Recoin")
    ROOT_PATH = os.path.join("/net","dataset", "audiffre","cleaned_data")
    filename_recoin= os.path.join(ROOT_PATH,"prediction_RECOIN_new.pkl.bz2")

    ID_test = np.load(os.path.join(ROOT_PATH, "indexes_test.npy"))
    y_pred = pd.read_pickle(filename_recoin).values


    
    logging.info("Loading True Y ")
    ROOT_PATH = os.path.join("/net","dataset", "pp","data","cleaned_data")
    filename_test= os.path.join(ROOT_PATH,"Y_cleaned_20180813.npy")
    Y_test = pd.read_pickle(filename_test)


    Y_test = Y_test.loc[ID_test].values

    print(y_pred.shape, Y_test.shape)

    logging.info("Loaded Everythin'")


    print( ' Finding threshold. ')

    possible_thresholds = list(np.arange(0.1,1,0.1))+list(np.logspace(1e-5,1e-1,num=10)) 

    
    bthr=.4
    #bf1=0
    #for ip,thrp in enumerate(possible_thresholds):
    #    f1 = exam_f1(y_pred,Y_test,threshold=thrp)
    #    print("{} - value {:.2f}, F1 {:.2f}".format(ip, thrp, f1))
    #    if f1 >bf1 :
    #        bf1 = f1
    #        bthr = thrp

    #print("selected thr : {} (F1 : {})".format(bthr, bf1))



    
    

    print( "Evaluating Recoin for different P")


    for P in [5,10,15,20,25]:


        version="Recoin"+"_"+str(P)

        is_selected = (np.sum(Y_test,axis=1)>= P)
        Y_test =Y_test[is_selected]
        y_pred = y_pred[is_selected]
        print( 'P = {} : Test {}, Pred {}'.format(P, Y_test.shape,y_pred.shape))
        n_test=len(Y_test)



        print("Computing error")
        errors = {}
        errors["N"]=n_test
        for m in METRICS :
            error_pred = m(y_pred, Y_test, threshold=bthr )
            errors[m.__name__] = error_pred
            print(m.__name__,error_pred)




                

        try :
            scores = pickle.load(open("saves/scores_P_var_Oct_2020.pickle", "rb"))
        except :
            scores = {}
        scores[version] = errors
        pickle.dump(scores, open("saves/scores_P_var_Oct_2020.pickle", "wb"))
