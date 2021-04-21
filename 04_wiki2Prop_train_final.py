import argparse
import logging
import operator
import torch
from tools import mlauc, METRICS, auc_thr
import os
import time
import numpy as np
from slackpush import slack_message
from Wiki2Prop_train_partial import Wiki2Prop_Part, Wiki2Prop, SmartEarlyStopping
from data_loader import DataLoader

torch.backends.cudnn.benchmark = True



if __name__ == "__main__":



    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--gpu', '-g',type= int, default=7)
    parser.add_argument('--lr', '-l',type= int, default=1)
    parser.add_argument('--reg', '-r',type= int, default=7)
    parser.add_argument("--year","-y",type= int, default=0)  # 0 : 2018, 1 : 2019
    parser.add_argument("--embedding","-e",type= int, default=0)  # 0 : 300LG , 1 : 300NLG, 2: 100LG
    parser.add_argument("--deploy","-d",type= int, default=0)  # 0 : regular train/test,  1: Deply mode : train on train+test
    parser.add_argument("--with_image","-w",type= int, default=1)  # use image mode
    args = parser.parse_args()


    with_image = (args.with_image == 1)
    gpu = args.gpu
    lr = 10**(-args.lr)
    reg = 10**(-args.reg)
    year = "2018" if args.year ==0 else "2019"
    if args.embedding == 0:
        embedding = "300LG"
    elif args.embedding == 1:
        embedding = "300NLG"
    elif args.embedding == 2:
        embedding = "100LG"

    TRAINING_DEPLOY = (args.deploy == 1)
    if TRAINING_DEPLOY : 
        logging.info("WARNING : DEPLOY MODE TRAINING. NO TEST SET.")

    device = torch.device(gpu) if torch.cuda.device_count()>gpu  else torch.device(0)




    #Getting data
    logging.info("Loading Data")
    data = DataLoader(year = year, embedding=embedding, with_image = with_image)
    
    X_train, Y_train = data.get_train()
    if TRAINING_DEPLOY :
        X_test, Y_test = data.get_test()
        X_train = np.concatenate([X_train, X_test],axis=0)
        Y_train = np.concatenate([Y_train, Y_test],axis=0)
        del X_test, Y_test
            
    X_valid, Y_valid = data.get_valid()
    data.clean()

    w =np.sum(Y_valid, axis=0)
    w = w / np.sum(w)

    logging.info("Data loaded")


    nlabels = Y_train.shape[1]
    ndim = X_train.shape[1]


    string_filename_template = "Wiki2Prop_{}_DEPLOY_year{}_embedding{}.pt" if TRAINING_DEPLOY else "Wiki2Prop_{}_year{}_embedding{}.pt"
    

    complete_dict = {}

    for lang in ["EN",'DE','FR']:
        complete_dict.update(torch.load(os.path.join("saves",string_filename_template.format(lang ,year, embedding))))


    ccn = Wiki2Prop(nlabels)
    keysans = ccn.load_state_dict(complete_dict,strict=False)
    logging.info(keysans)
    ccn.freeze()
    ccn.to(device)

    logging.info("Model Loaded")


    closs = torch.nn.BCELoss()

    #========================================================
    # CLASSIFICATION Network
    #========================================================

    BATCH_SIZE = 4096

    result_watcher = SmartEarlyStopping(start_epoch= 10, starting_lr=lr, increase=False, minimum_improvement=1e-3, patience=10)

    

    optimizer = torch.optim.SGD(
        ccn.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=reg)

    n_train = len(X_train)
    n_valid = len(X_valid)

    if TRAINING_DEPLOY :
           n_train = int(len(X_train)*0.95)
           X_train =X_train[:n_train]
           Y_train = Y_train[:n_train]

    
 

    features_ = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(Y_train)#.to(device)


    features_valid = torch.Tensor(X_valid)#.to(device)
    y_valid = torch.Tensor(Y_valid)


    indexes = list(range(n_train))
    epoch = 0
    clr = lr
    logging.info("Start training Wiki2Prop. Number of items : {} ".format(n_train))
    input("Continue?")

    ndivid = 3
    while clr>0 :
        epoch+=1
        np.random.shuffle(indexes)
        ccn.train()
        mloss=0

        
        for batch_start in np.arange(0,n_train,BATCH_SIZE ):
            
            print("Batch {:.2f} %".format(100*batch_start/n_train), end="\r")

            batch = features_[indexes[int(batch_start):int(batch_start+BATCH_SIZE)]]#.to(device)
            y_batch = y_train[indexes[int(batch_start):int(batch_start+BATCH_SIZE)]].to(device)

            optimizer.zero_grad()
            
            y_ = ccn(batch)
            loss = closs(y_, y_batch)
            loss.backward()
            optimizer.step()
            mloss += float(loss)
        
        del batch, y_batch, y_

        mloss = mloss / (n_train//BATCH_SIZE)

        # validation
        ccn.eval()

        vloss=0
        for batch_start in np.arange(0,n_valid,BATCH_SIZE ):
            
            print("Batch Valid {:.2f} %".format(100*batch_start/n_valid), end="\r")

            batch = features_valid[int(batch_start):int(batch_start+BATCH_SIZE)].to(device)
            y_batch = y_valid[int(batch_start):int(batch_start+BATCH_SIZE)]#.to(device)
            
            y_ = ccn(batch).cpu()
            vloss += float(closs(y_,y_batch))
        
        del batch, y_batch, y_

        vloss = vloss / (n_valid//BATCH_SIZE) 
        
        msg_str = "Wiki2Prop lr = {:.1E}({:.1E}), r={:.1E},  Epoch {}; Errors : Train {:.2E};  Valid {:.2E}".format(clr, lr,reg, str(epoch),mloss,vloss)
        logging.info(msg_str)#, end="")
        

        nlr = result_watcher.feed(vloss)

        if (nlr<clr):
            clr= nlr
            if clr >0 :
                for param_group in optimizer.param_groups:
                    param_group['lr'] = clr



    string_filename_save_template = ""
    if TRAINING_DEPLOY :
        string_filename_save_template = "Wiki2Prop_DEPLOY_year{}_embedding{}.pt" 
    elif with_image :
        string_filename_save_template = "Wiki2Prop_IMAGE_year{}_embedding{}.pt" 
    else :
        string_filename_save_template = "Wiki2Prop_year{}_embedding{}.pt"
    torch.save(ccn.state_dict(),os.path.join("saves",string_filename_save_template.format(year, embedding)))

    y_v = []
    for batch_start in np.arange(0,n_valid,BATCH_SIZE ):

        print("Batch Valid {:.2f} %".format(100*batch_start/n_valid), end="\r")

        batch = features_valid[int(batch_start):int(batch_start+BATCH_SIZE)]#.to(device)
        y_batch = y_valid[int(batch_start):int(batch_start+BATCH_SIZE)]#.to(device)
        
        y_ = ccn(batch).detach().cpu().numpy()
        y_v.append(y_)
    y_v = np.concatenate(y_v,axis=0)

    vauc = mlauc(y_v, Y_valid)
    

    msg_str = "Wiki2Prop lr ={:.1E},  END TRAINING; Errors : Train {:.2E};  Valid {:.2E}; Valid AvgAUC {:.4f} (+- {:.4f})  ### WAUC {:.4f}".format( lr, mloss, vloss, np.mean(vauc), np.std(vauc), np.sum(vauc*w))
    slack_message(msg_str)







