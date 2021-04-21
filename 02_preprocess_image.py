from __future__ import print_function
from __future__ import division
from urllib.parse import ParseResult
from matplotlib import image
from numpy.core.defchararray import endswith
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import tarfile

import multiprocessing
from joblib import Parallel, delayed



import argparse
import logging

from PIL import Image
from tqdm import tqdm
import pandas

import pickle as pkl


from data_loader import get_all_indexes

torch.backends.cudnn.benchmark = True



#ROOT_PATH = os.path.join("//net","dataset")
ROOT_PATH = os.path.join("//var","cache","fscache","audiffre")


TMPL_raw_images = os.path.join(ROOT_PATH, 'images')
TMPL_to_processed_images = os.path.join(ROOT_PATH, "processed_images")
TMPL_to_embedded_images = os.path.join(ROOT_PATH, "cleaned_data","embedded_img.feather")
TMPL_to_embedded_images2 = os.path.join(ROOT_PATH, "cleaned_data","embedded_img2.feather")
TMPL_to_cleaned_data_input = os.path.join(ROOT_PATH,"cleaned_data","X_cleaned_20180813.npy")
TMPL_to_cleaned_data_output = os.path.join(ROOT_PATH, "cleaned_data","X_full_2018.npy")
TMPL_to_cleaned_data_input_2019 = os.path.join(ROOT_PATH,"cleaned_data","X_cleaned_20190916.npy")
TMPL_to_cleaned_data_output_2019 = os.path.join(ROOT_PATH, "cleaned_data","X_full_2019.npy")
INPUT_SIZE = 224

embedding = "300LG"
year = "2018"


if __name__ == "__main__":

    
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)




    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--gpu', '-g',type= int, default=7)
    parser.add_argument('--task', '-t',type= int, default=0) # 0 : preprocess, 1 : embedd
    parser.add_argument("--cpu", "-c",type= int, default=8) # number of cpu used
        
    args = parser.parse_args()

    gpu = args.gpu
    task_number =args.task


    device = torch.device(gpu) if torch.cuda.device_count()>gpu  else torch.device(0)


    
    if task_number == 0 : #goal; preprocess
        #all_indexes = get_all_indexes()


        with open(os.path.join(ROOT_PATH,  "aws_list_q.pickle"),"rb") as file :
            dict_name_to_q = pkl.load(file)

        images =list(os.listdir(TMPL_raw_images))
        
        n_img = len(images)
        nloc = n_img//100

        logging.info("Preprocessing {} images".format(n_img))

        image_preprocessing = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            #transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        def process_image(image_name) :

            

            if "." not in image_name :
                extension ="png"
            else :
                extension = image_name.split(".")[-1]

            #try :
            q_number =image_name+"\n" #dict_name_to_q[image_name+"\n"]
            #except :
            #    return

            input_path = os.path.join(TMPL_raw_images,image_name)
            output_path = os.path.join(TMPL_to_processed_images,"Q{}.{}".format(q_number,extension))
            
            
            if os.path.isfile(output_path):
                return 

            try :
                img = Image.open(input_path)
                img = img.convert('RGB')
                img =  image_preprocessing(img)
                img.save(output_path)
            except Exception as e:
                logging.info("skipping img : {} because : {}".format(image_name, e))

            
        current_list = tqdm(images)
        
        
        Parallel(n_jobs=args.cpu)(delayed(process_image)(image_name) for image_name in current_list)
        
        




    elif task_number == 1 : #goal; embedding
        
  
        
        images = list( os.listdir(TMPL_to_processed_images))

        #images= images[:3000]
        
        
        n_img = len(images)

        logging.info("Processing {} images".format(n_img))

        image_preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        def process_image(image_name) :
            image_path = os.path.join(TMPL_to_processed_images,image_name)
            
            try :
                img = Image.open(image_path)
                img =  image_preprocessing(img)
                
                
            except Exception as e:
                logging.info("skipping img : {} because : {}".format(image_name, e))
                img= None

            return (image_name,img)

            
        current_list = tqdm(images)
        
        
        tensor_images_list= Parallel(n_jobs=args.cpu)(delayed(process_image)(image_name) for image_name in current_list)

        tensor_images_list = [ u for u in tensor_images_list if u[1] is not None]

        id_images_list= [u[0] for u in tensor_images_list]
        tensor_images_list= [u[1] for u in tensor_images_list]
        encoded_list = []


        logging.info("Encoding {} images".format(len(tensor_images_list)))


        n_img = len(tensor_images_list)

    

        BATCH_SIZE = 1024

        model_densenet = models.densenet121(pretrained=True)
        for param in model_densenet.parameters():
            param.requires_grad = False

        model_densenet.classifier = torch.nn.Identity()

        model_densenet.to(device)

    
        for batch_start in np.arange(0,n_img,BATCH_SIZE):
            
            print("Batch {:.2f} %".format(100*batch_start/n_img), end="\r")

            batch =torch.stack(tensor_images_list[int(batch_start):int(batch_start+BATCH_SIZE)]).to(device)
        

            y_ = model_densenet(batch).cpu().numpy()
            encoded_list.append(y_)
            

        encoded_list = np.vstack(encoded_list)

        

        results = pandas.DataFrame(data = encoded_list, index=id_images_list)
        results = results.reset_index()
        results.columns = [ "C_{}".format(c) for c in results.columns]

        results.to_feather(TMPL_to_embedded_images)


    elif task_number == 2 :
        logging.info("updating indexes of image")

        with open(os.path.join(ROOT_PATH,  "aws_list_q.pickle"),"rb") as file :
            dict_name_to_q = pkl.load(file)


        
        def update_index(index):
            index = index[1:-5]  #remove Q and double extension
            if index[-1]!='\n' :
               index = index+'\n' 
            try :
                return "Q_"+str(dict_name_to_q[index])
            except :
                #print(index)
                return "UNK"


        embedded_img = pandas.read_feather(TMPL_to_embedded_images)  

        indexes = embedded_img.C_index.map(update_index)
        embedded_img.C_index = indexes

        print(sum( embedded_img.C_index == 'UNK'))


        embedded_img.to_feather(TMPL_to_embedded_images2)
        
        

    elif task_number == 3 :
        logging.info("Merging data with image")


        embedded_img = pandas.read_feather(TMPL_to_embedded_images2)  

        # preprocessed the index of the embedded images

        indexes = embedded_img.C_index
        index_type, index_count = np.unique(indexes,return_counts = True)
        multi_index = index_type[ index_count>1]

        unique_indexes = [not (i in multi_index) for i in indexes]

        embedded_img = embedded_img[unique_indexes]

        indexes = embedded_img.C_index
        indexes = [ int(c.split("_")[1]) for c in indexes if '_' in c]
        embedded_img = embedded_img[embedded_img.C_index!="UNK"]
        
        embedded_img.index = indexes
        embedded_img = embedded_img.drop(columns = ["C_index"])
        
        #load wikipedia embeddings

        X = pandas.read_pickle(TMPL_to_cleaned_data_input)
        print("Prior to merge : {}".format(X.shape))

        # merge the two modes
        X = X.merge(right=embedded_img, how="left", left_index = True, right_index = True).fillna(value =0 )
        print("Post merge : {}".format(X.shape))
        


        
        #save 
        X = X.reset_index()
        X.columns = [ "C_{}".format(c) for c in X.columns]
        X.to_feather(TMPL_to_cleaned_data_output)
        

        print("Entities w/ english embedding: {}".format(sum(X["C_0"]!=0)))
        print("Entities w/ german embedding: {}".format(sum(X["C_300"]!=0)))
        print("Entities w/ english embedding: {}".format(sum(X["C_600"]!=0)))
        print("Entities w/ image embedding: {}".format(sum(X["C_C_1000"]!=0)))


    elif task_number == 4 :
        logging.info("Merging data with image v 2019")


        embedded_img = pandas.read_feather(TMPL_to_embedded_images2)  

        # preprocessed the index of the embedded images

        indexes = embedded_img.C_index
        index_type, index_count = np.unique(indexes,return_counts = True)
        multi_index = index_type[ index_count>1]

        unique_indexes = [not (i in multi_index) for i in indexes]

        embedded_img = embedded_img[unique_indexes]

        indexes = embedded_img.C_index
        indexes = [ int(c.split("_")[1]) for c in indexes if '_' in c]
        embedded_img = embedded_img[embedded_img.C_index!="UNK"]
        
        embedded_img.index = indexes
        embedded_img = embedded_img.drop(columns = ["C_index"])
        
        #load wikipedia embeddings

        X = pandas.read_pickle(TMPL_to_cleaned_data_input_2019)
        print("Prior to merge : {}".format(X.shape))

        # merge the two modes
        X = X.merge(right=embedded_img, how="left", left_index = True, right_index = True).fillna(value =0 )
        print("Post merge : {}".format(X.shape))
        


        
        #save 
        X = X.reset_index()
        X.columns = [ "C_{}".format(c) for c in X.columns]
        X.to_feather(TMPL_to_cleaned_data_output_2019)
        

        print("Entities w/ english embedding: {}".format(sum(X["C_0"]!=0)))
        print("Entities w/ german embedding: {}".format(sum(X["C_300"]!=0)))
        print("Entities w/ english embedding: {}".format(sum(X["C_600"]!=0)))
        print("Entities w/ image embedding: {}".format(sum(X["C_C_1000"]!=0)))


