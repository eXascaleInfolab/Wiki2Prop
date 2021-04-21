import argparse
import logging
import operator
import torch
from tools import mlauc, METRICS, auc_thr
import os
import time
import numpy as np
from slackpush import slack_message
from data_loader import DataLoader

torch.backends.cudnn.benchmark = True



class SmartEarlyStopping():
    def __init__(self, start_epoch=0, starting_lr = 1, increase=False,minimum_improvement=0, patience = 50, fails=20 ):
        self.start_epoch = start_epoch
        self.minimum_improvement = minimum_improvement
        self.patience = patience
        self.increase = increase
        self.max_fails = fails

        self.ref_value = None
        self.observed_valued_since_improvement = 0
        self.fails = 0
        self.lr = starting_lr

    def feed(self, value):
        
        if self.ref_value is None :
            self.observed_valued_since_improvement +=1
            if self.observed_valued_since_improvement > self.start_epoch :
                self.ref_value = value
                self.observed_valued_since_improvement = 0
        else :
            if  ((~self.increase) & (value < (1-self.minimum_improvement)*self.ref_value)) | \
                    ((self.increase) & (value > (1+self.minimum_improvement)*self.ref_value)):
                self.ref_value = value
                self.observed_valued_since_improvement = 0
            else :
                self.observed_valued_since_improvement +=1
                if self.observed_valued_since_improvement> self.patience:
                    self.observed_valued_since_improvement= 0
                    self.fails +=1
                    self.lr = 0.5*self.lr
                    if self.fails > self.max_fails :
                        self.lr = 0
        return self.lr



class Wiki2Prop_Part(torch.nn.Module):
    """
    Part of Wiki2Prop,for training purpose

    """

    def __init__(self, nlabels, langage = "EN"):

        super(Wiki2Prop_Part, self).__init__()
        self.lang = langage

        setattr(self, "dense_layer1_{}".format(langage),torch.nn.Linear(in_features=300,out_features=512))
        setattr(self, "dense_layer2_{}".format(langage),torch.nn.Linear(in_features=512,out_features=2048))
        setattr(self, "dense_layer3_{}".format(langage),torch.nn.Linear(in_features=2048,out_features=nlabels))


    def forward(self, x):

        final_x =   torch.relu(getattr(self, "dense_layer1_{}".format(self.lang))(x))
        final_x =   torch.relu(getattr(self, "dense_layer2_{}".format(self.lang))(final_x))
        final_x =   torch.sigmoid(getattr(self, "dense_layer3_{}".format(self.lang))(final_x))
        return final_x




class Wiki2Prop (torch.nn.Module):
    """
    Seemless Embedding Fusion for Multi Label Prediction

    """

    def __init__(self, nlabels,with_img= True):

        self.with_img =with_img

        super(Wiki2Prop, self).__init__()


        self.dense_layer1_EN = torch.nn.Linear(in_features=300,out_features=512)
        self.dense_layer2_EN = torch.nn.Linear(in_features=512,out_features=2048)

        self.dense_layer1_DE = torch.nn.Linear(in_features=300,out_features=512)
        self.dense_layer2_DE = torch.nn.Linear(in_features=512,out_features=2048)

        self.dense_layer1_FR = torch.nn.Linear(in_features=300,out_features=512)
        self.dense_layer2_FR = torch.nn.Linear(in_features=512,out_features=2048)

        fusion_dimension = 3*2048
        if with_img : fusion_dimension+=1024


        self.fusion_layer = torch.nn.Linear(in_features=fusion_dimension,out_features=2048)
        self.post_layer = torch.nn.Linear(in_features=2048,out_features=nlabels)


    def forward(self, x):

        w1 = torch.norm(x[:,:300],dim=1)
        w2 = torch.norm(x[:,300:600],dim=1)
        w3 = torch.norm(x[:,600:900],dim=1)
        if self.with_img :
            w_img = x[:,900:]

        middle_x1 =   torch.relu(self.dense_layer2_EN(torch.relu(self.dense_layer1_EN(x[:,:300]))))
        middle_x2 =   torch.relu(self.dense_layer2_DE(torch.relu(self.dense_layer1_DE(x[:,300:600]))))
        middle_x3 =   torch.relu(self.dense_layer2_FR(torch.relu(self.dense_layer1_FR(x[:,600:900]))))

        middle_x1[w1==0]=0
        middle_x2[w2==0]=0
        middle_x3[w3==0]=0

        if self.with_img :
            fused_x = torch.cat((middle_x1,middle_x2,middle_x3,w_img), dim=1)
        else :
            fused_x = torch.cat((middle_x1,middle_x2,middle_x3), dim=1)
        
        fused_x = torch.relu(self.fusion_layer(fused_x))
        final_x = torch.sigmoid(self.post_layer(fused_x))
        return final_x

    def freeze(self):
        for layer in [self.dense_layer1_EN,self.dense_layer2_EN, self.dense_layer1_DE,self.dense_layer2_DE,\
                        self.dense_layer1_FR,self.dense_layer2_FR ]:
            for param in layer.parameters():
                param.requires_grad = False




if __name__ == "__main__":

    





    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)




    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--gpu', '-g',type= int, default=6)
    parser.add_argument('--lr', '-l',type= int, default=1)
    parser.add_argument('--reg', '-r',type= int, default=7)
    parser.add_argument('--lang', '-t',type= int, default=0)
    parser.add_argument("--year","-y",type= int, default=0)  # 0 : 2018, 1 : 2019
    parser.add_argument("--embedding","-e",type= int, default=0)  # 0 : 300LG , 1 : 300NLG, 2: 100LG
    parser.add_argument("--deploy","-d",type= int, default=0)  # 0 : regular train/test,  1: Deply mode : train on train+test
    
    args = parser.parse_args()



    gpu = args.gpu
    lr = 10**(-args.lr)
    reg = 10**(-args.reg)
    lang = int(args.lang)

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
    data = DataLoader(year = year, embedding=embedding)
    X_train, Y_train = data.get_train()
    if TRAINING_DEPLOY :
        X_test, Y_test = data.get_test()
        X_train = np.concatenate([X_train, X_test],axis=0)
        Y_train = np.concatenate([Y_train, Y_test],axis=0)
        del X_test, Y_test



    X_valid, Y_valid = data.get_valid()
    data.clean()



    logging.info("Data shape {}".format(X_train.shape))

  
    



    
    if lang==1 :
        langage ="DE"
    elif lang==2 :
        langage = "FR"
    elif lang == 0:
        langage = "EN"


    x_train = X_train[:,lang*300:(lang+1)*300]
    vx_train = np.linalg.norm(x_train,axis=1)
    x_train = x_train[vx_train>0]
    y_train = Y_train[vx_train>0]


    x_valid = X_valid[:,lang*300:(lang+1)*300]
    vx_valid = np.linalg.norm(x_valid,axis=1)
    x_valid = x_valid[vx_valid>0]
    y_valid = Y_valid[vx_valid>0]


    nlabels = Y_train.shape[1]
    ndim = X_train.shape[1]


    closs = torch.nn.BCELoss()

    #========================================================
    # CLASSIFICATION Network
    #========================================================

    BATCH_SIZE = 4096

    result_watcher = SmartEarlyStopping(start_epoch= 10, starting_lr=lr, increase=False, minimum_improvement=1e-3, patience=10)

    ccn = Wiki2Prop_Part(nlabels,langage=langage)
    optimizer = torch.optim.SGD(
        ccn.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=reg)

    ccn.to(device)

    n_train = len(x_train)
    n_valid = len(x_valid)

    features_ = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)


    features_valid = torch.Tensor(x_valid).to(device)
    y_valid = torch.Tensor(y_valid)


    indexes = list(range(n_train))
    epoch = 0
    clr = lr
    logging.info("Start training {}. Number of items : {} / {}".format(langage, n_train, len(X_train)))
    input("Continue?")
    while clr>0 :
        epoch+=1
        np.random.shuffle(indexes)
        ccn.train()
        mloss=0
        for batch_start in np.arange(0,n_train,BATCH_SIZE ):
            
            print("Batch {:.2f} %".format(100*batch_start/n_train), end="\r")

            batch = features_[indexes[int(batch_start):int(batch_start+BATCH_SIZE)]]#.to(device)
            y_batch = y_train[indexes[int(batch_start):int(batch_start+BATCH_SIZE)]]#.to(device)

            optimizer.zero_grad()
            
            y_ = ccn(batch)
            loss = closs(y_, y_batch)
            loss.backward()
            optimizer.step()
            mloss += float(loss)
        
        del batch, y_batch, y_

        mloss = mloss / (n_train//BATCH_SIZE)

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
        
        msg_str = "Wiki2Prop LANG {} lr = {:.1E}({:.1E}), r={:.1E},  Epoch {}; Errors : Train {:.2E};  Valid {:.2E}".format(langage,clr, lr,reg, str(epoch),mloss,vloss)
        logging.info(msg_str)#, end="")
        

        nlr = result_watcher.feed(vloss)

        if (nlr<clr):
            clr= nlr
            if clr >0 :
                for param_group in optimizer.param_groups:
                    param_group['lr'] = clr

    if TRAINING_DEPLOY :
        savepath = os.path.join("saves","Wiki2Prop_{}_DEPLOY_year{}_embedding{}.pt".format(langage ,year, embedding))
    else :
        savepath = os.path.join("saves","Wiki2Prop_{}_year{}_embedding{}.pt".format(langage ,year, embedding))


    torch.save(ccn.state_dict(),savepath)

   

    msg_str = "Wiki2Prop LANG {} lr = {:.1E}, r={:.1E},  END TRAINING; Errors : Train {:.2E};  Valid {:.2E}".format(langage, args.lr,args.reg,mloss,vloss)
    slack_message(msg_str)






