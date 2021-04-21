import argparse
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





####################
#NETWORK DEF
####################


class BPMLL (torch.nn.Module):
    """
    From the seminal paper 
            Multi-Label Neural Networks with Applications to
            Functional Genomics and Text Categorization
    """

    def __init__(self, nlabels, ndims=300):

        super(BPMLL, self).__init__()

        self.dense_layer1 = torch.nn.Linear(in_features=ndims,out_features=512)
        self.dense_layer2 = torch.nn.Linear(in_features=512,out_features=1024)
        self.dense_layer3 = torch.nn.Linear(in_features=1024,out_features=nlabels)
        
        self.dropout = torch.nn.Dropout(p=.5)

    def forward(self, x):

        middle_x = self.dropout(torch.relu(self.dense_layer1(x)))
        middle_x = self.dropout(torch.relu(self.dense_layer2(middle_x)))
        final_x = torch.sigmoid(self.dense_layer3(middle_x))
        return final_x


def custom_loss_bpmll(y_, y_true):



    error = 0

    for i in range(len(y_true)):
        error_i = 0
        y_p = y_[i,y_true[i]]
        y_n = y_[i,~y_true[i]]
        error_i = y_p[:,None] - y_n[None,:]
        error_i = torch.mean(torch.exp(-error_i)) 
        error = error + error_i
    return error




if __name__ == "__main__":

        
    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--gpu', '-g',type= int, default=6)
    parser.add_argument('--lr', '-l',type= int, default=1)
    parser.add_argument("--year","-y",type= int, default=0)  # 0 : 2018, 1 : 2019
    parser.add_argument("--embedding","-e",type= int, default=0)  # 0 : 300LG , 1 : 300NLG, 2: 100LG
    args = parser.parse_args()

    gpu = args.gpu
    lr = 10**(-args.lr)
    year = "2018" if args.year ==0 else "2019"
    if args.embedding == 0:
        embedding = "300LG"
    elif args.embedding == 1:
        embedding = "300NLG"
    elif args.embedding == 2:
        embedding = "100LG"

    device = torch.device(gpu) if torch.cuda.device_count()>gpu  else torch.device(0)


    #Getting data
    logging.info("Loading Data")
    data = DataLoader(year = year, embedding=embedding)
    X_train, Y_train = data.get_train()
    X_valid, Y_valid = data.get_valid()
    data.clean()

    #Change dims
    X_train = X_train[:,:300]
    X_valid = X_valid[:,:300]


    ndims = X_train.shape[1]
    nlabels = Y_train.shape[1]



    if 0:
        closs = custom_loss_bpmll
    else :
        closs = torch.nn.BCELoss()

    #========================================================
    # CLASSIFICATION Network
    #========================================================

    TOTAL_EPOCHS = 200 
    SAVEEVERY = TOTAL_EPOCHS/10
    BATCH_SIZE = 4096


    ccn = BPMLL(nlabels)


    optimizer = torch.optim.SGD(
        ccn.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    scheduler_optim = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10000, gamma=.99)

    ccn.to(device)
    train_losses = []
    valid_losses = []
    test_losses = []
    auc = []


    n_train = len(X_train)
    n_valid = len(X_valid)

    features_ = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(Y_train).to(device)
    #if args.model == 0 :
    #    y_train = y_train.bool()


    features_valid = torch.Tensor(X_valid).to(device)
    y_valid = torch.Tensor(Y_valid)
    #if args.model == 0 :
    #    y_valid = y_valid.bool()



    indexes = list(range(n_train))
    for epoch in range(TOTAL_EPOCHS):
        strat = time.time()

        np.random.shuffle(indexes)
        ccn.train()
        mloss=0
        strat = time.time()
        for batch_start in np.arange(0,n_train,BATCH_SIZE ):
            
            print("Batch {:.2f} %".format(100*batch_start/n_train), end="\r")

            batch = features_[indexes[int(batch_start):int(batch_start+BATCH_SIZE)]]#.to(device)
            y_batch = y_train[indexes[int(batch_start):int(batch_start+BATCH_SIZE)]]#.to(device)

        
            optimizer.zero_grad()
            
            y_ = ccn(batch)
            loss = closs(y_,y_batch)
            loss.backward()
            optimizer.step()
            mloss += float(loss)
        
        del batch, y_batch, y_
        logging.info("Done epoch training")

        mloss = mloss / n_train
        train_losses.append(mloss)

        # validation
        # validation
        ccn.eval()

        vloss=0
        for batch_start in np.arange(0,n_valid,BATCH_SIZE ):
            
            print("Batch Valid {:.2f} %".format(100*batch_start/n_valid), end="\r")

            batch = features_valid[int(batch_start):int(batch_start+BATCH_SIZE)]#.to(device)
            y_batch = y_valid[int(batch_start):int(batch_start+BATCH_SIZE)]#.to(device)
            
            y_ = ccn(batch).cpu()
            vloss += float(closs(y_,y_batch))
        
        del batch, y_batch, y_

        vloss = vloss / (n_valid//BATCH_SIZE) 
        valid_losses.append(vloss)

        logging.info("{:.2f}s  Epoch {}  Train Error  {:.2E}  Valid Error  {:.2E}  ".format(time.time()-strat, str(epoch),mloss,vloss))#, end="")



    torch.save(ccn.state_dict(),os.path.join("saves","BPMLL_year{}_embedding{}.pt".format(year, embedding)))











