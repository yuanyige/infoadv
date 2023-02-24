import numpy as np
import functools
import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def get_data_from_mask(mask, embeddings:np.array, y):
    index = torch.nonzero(mask).reshape(-1).tolist()
    X = embeddings[index]
    y = y[index]
    return X, y

def trans_emb_y(embeddings:torch.Tensor, y):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
    X = normalize(X, norm='l2')
    return X, Y


def label_classification_cv(embeddings, y, ratio):
    X, Y = trans_emb_y(embeddings, y)
    ret = []
    for i in range(3):#Cora10
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio, random_state = i+1)
        #print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 10)

        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                        param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                        verbose=0)

        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = prob_to_one_hot(y_pred)

        micro = f1_score(y_test, y_pred, average="micro")
        macro = f1_score(y_test, y_pred, average="macro")

        metirx =  [micro,macro]
        ret.append(metirx)

    ret = np.array(ret)
    mean = np.mean(ret,axis=0)
    std = np.std(ret,axis=0)
    return mean,std


def label_classification(embeddings, y, train_mask, val_mask, test_mask):
    X, Y = trans_emb_y(embeddings, y)

    X_train, y_train = get_data_from_mask(train_mask, X, Y)
    X_val, y_val = get_data_from_mask(val_mask, X, Y)
    X_test, y_test = get_data_from_mask(test_mask, X, Y)

    train_indices = np.full((X_train.shape[0],), -1, dtype=int)
    val_indices = np.full((X_val.shape[0],), 0, dtype=int)
    train_val_fold = np.append(train_indices, val_indices)
    X_train_val = np.concatenate((X_train, X_val),0)
    y_train_val = np.concatenate((y_train, y_val),0)
    ps = PredefinedSplit(train_val_fold)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=ps,
                       verbose=0)
    
    for train,test in clf.cv.split(X_train_val):
        print('TRAIN: ', train, ' TEST: ', test)

    clf.fit(X_train_val, y_train_val)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return micro, macro


def mean_classifier(embeddings, y):
    #print(torch.nonzero(y==1).shape)
    #print(embeddings[[1,2,5,8]].shape)
    #print(y.max())
    mc=torch.zeros((y.max().item()+1),embeddings.shape[1]).to(embeddings.device)
    #print(mc)
    for i in range(y.max().item()+1):
        index=torch.nonzero(y==i).reshape(-1)
        w=(embeddings[index]).mean(dim=0)
        mc[i]=w
    #print(mc.t().shape)
    #print('result',torch.mm(embeddings,mc.t()).argmax(dim=1))
    input=torch.mm(embeddings,mc.t())
    label_pre=input.argmax(dim=1)
    loss = F.cross_entropy(input, y)
    acc=torch.nonzero(label_pre-y).shape[0]/y.shape[0]
    # print('loss',loss)
    # print('acc',)
    return acc, loss

