
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def mlauc(y_, y_true, **kwargs):
    """
    Compute the AUC for each dimension. The Higher, the Better
    """

    err = [] 
    for i in range(y_.shape[1]):
        err.append( roc_auc_score(y_true=y_true[:,i], y_score=y_[:,i]) )
    
    return err

def auc_thr(y_, y_true):
    """
    Compute a Threshold for each dimension, in order to maximize TPR - FPR
    """

    thrs = [] 
    for i in range(y_.shape[1]):

        fpr,tpr,thr = roc_curve(y_true=y_true[:,i], y_score=y_[:,i])
        value_gain = (tpr-fpr)*(tpr>.8)
        if max(value_gain)>0 :
            best_thr = thr[np.argmax(value_gain)] 
        else :
            best_thr = .5
        thrs.append(best_thr)
    
    return np.array(thrs)



def subset_accuracy(y_, y_true,threshold =None):
    """
    Average subset Accuracy. Subset accuracy is 1 if perfect prediction, 0 otherwise. The Higher, the Better
    """
    if threshold is None :
        threshold = .5

    y_2 = (y_>threshold)
    subset_ = np.sum(np.abs(y_true - y_2), axis=1)
    valid_subset = (subset_==0)

    return np.mean(valid_subset)


def hamming_loss(y_, y_true,threshold = None):
    """
    Average Hamming Loss. Hamming loss is the proportion of addition + deletion necessary to recover y_true from y_. The Lower, the Better
    """
    if threshold is None :
        threshold = .5

    y_true = (y_true==1)
    y_2 = (y_>threshold)
    errors = y_2 ^ y_true
    errors = np.sum(errors, axis=1)
    return np.mean(errors)

def exam_accuracy(y_, y_true,threshold = None):
    """
    Average Accuracy. Accuracy is the number of correct label predicted divided by number of labels predicted OR true. The Higher, the Better
    """
    if threshold is None :
        threshold = .5
    y_true = (y_true==1)
    y_2 = (y_>threshold)
    inter = np.sum(y_2 & y_true ,axis=1)
    outer = np.sum(y_2 | y_true, axis=1)

    return np.mean(inter/outer)


def exam_precision(y_, y_true,threshold = None):
    """
    Average Precision. Precision is the number of correct label predicted divided by number of labels predicted. The Higher, the Better
    """
    if threshold is None :
        threshold = .5
    y_true = (y_true==1)
    y_2 = (y_>threshold)
    inter = np.sum(y_2 & y_true ,axis=1)
    outer = np.sum(y_2, axis=1)
    score = np.zeros_like(outer,dtype=float)
    score[outer>0] = inter[outer>0]/outer[outer>0]
    return np.mean(score)

def exam_recall(y_, y_true,threshold = None):
    """
    Average Recall. Recall is the number of correct label predicted divided by number of correct labels . The Higher, the Better
    """
    if threshold is None :
        threshold = .5
    y_true = (y_true==1)
    y_2 = (y_>threshold)
    inter = np.sum(y_2 & y_true ,axis=1)
    outer = np.sum(y_true , axis=1)
    score = np.zeros_like(outer,dtype=float)
    score[outer>0] = inter[outer>0]/outer[outer>0]
    return np.mean(score)


def exam_f1(y_, y_true,threshold = None):
    """
    Average F1. . The Higher, the Better
    """
    if threshold is None :
        threshold = .5
    p = exam_precision(y_,y_true,threshold=threshold)
    r = exam_recall(y_,y_true,threshold=threshold)

    
    outer = (2*p*r)/(p+r) if p+r>0 else 0
    return outer

def one_error(y_, y_true,**kwargs):
    """
    Average One Error.  Rated Correct if and only if the most probable label is true. The Higher, the Better
    """
    p = np.max(y_,axis=1,keepdims=True)
    p = (y_ == p)
    valid = p*(1-y_true)
    valid = np.sum(valid, axis=1)
    return np.mean(valid)

def coverage(y_, y_true,**kwargs):
    """
    Average Coverage.  Coverage = rank of the least probable true label. The Lower, the Better
    """
    rk_ = np.argsort(y_, axis=1)
    y_true_sorted = np.take_along_axis(y_true,rk_,axis=1)
    first_mistake = np.argmax(y_true_sorted, axis=1)
    first_mistake = rk_.shape[1] - first_mistake
    return np.mean(first_mistake)


def ranking_loss(y_, y_true,**kwargs):
    """
    Average Ranking Loss.  Ranking Loss = proportion of pairs of wrongly ordered labels ; i.e. true label less probable than false label. The Lower, the Better
    """
    rk_ = np.argsort(y_, axis=1)
    y_true_sorted = np.take_along_axis(y_true,rk_,axis=1)
    nb_1_left = np.cumsum( y_true_sorted, axis=1)
    ranking_loss = np.sum((1-y_true_sorted) * nb_1_left, axis=1)

    ranking_loss = ranking_loss /( np.sum(y_true, axis=1)*np.sum(1-y_true, axis=1) )
    ranking_loss = np.nan_to_num(ranking_loss)
    return np.mean(ranking_loss)

def mmr(y_, y_true,**kwargs):
    """
    Average MMR.  MMR = sum of the inverse of the rank of the the true labels. The Higher, the Better
    """
    rk_ = np.argsort(y_, axis=1)

    y_true_sorted = np.take_along_axis(y_true,rk_,axis=1)
    y_true_sorted= y_true_sorted[:,::-1]

    y_true_rank = y_true_sorted * (np.arange(1,y_true_sorted.shape[1]+1)[None,:])

    y_true_rank[y_true_rank>0] = 1/y_true_rank[y_true_rank>0] 

    mmr_loss = np.sum(y_true_rank, axis=1)
    return np.mean(mmr_loss)


METRICS = [subset_accuracy, hamming_loss, exam_accuracy, exam_precision, \
                exam_recall, exam_f1, one_error, coverage, ranking_loss,mmr]

METRICS_THR_DEPENDENT = [subset_accuracy, hamming_loss, exam_accuracy, exam_precision, \
                exam_recall, exam_f1]
METRICS_THR_INDEPENDENT = [ one_error, coverage, ranking_loss,mmr]


def binary_f1(y_, y_true, threshold = None):

    if threshold is None :
        threshold = .5

    y_true = (y_true==1)
    y_2 = (y_>threshold)

    class_true = np.sum(y_true,axis=0)
    class_predicted = np.sum(y_2,axis=0)
    class_true_n_predicted = np.sum( (y_2 * y_true),axis=0)

    precision = np.nan_to_num(class_true_n_predicted / class_predicted)
    recall = np.nan_to_num(class_true_n_predicted / class_true)
    f1 = np.nan_to_num((2*precision*recall)/(precision+recall))

    return f1






if __name__ == "__main__" :
    y_true = np.zeros((1,50))
    y_true[0,7]=y_true[0,21] = y_true[0,44] = y_true[0,1] =  1
    y_pred = np.zeros((1,50)) +0.1
    y_pred[0,44] =  0.6
    y_pred[0,21] = 0.55
    y_pred[0,7] = 0.5
    y_pred[0,1] = 0.051
    y_pred[0,3] = 0.7
    y_pred[0,13] = 0.97
    
    print(binary_f1(y_=y_pred, y_true = y_true))

    assert 0