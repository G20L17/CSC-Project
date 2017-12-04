import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
import sklearn.metrics as sklm
from datetime import datetime

RANDOM_SEED = 193
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape, name):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=name)


def forwardprop(X, w_1, b_1, w_2, b_2, keep_prob):
    h=tf.nn.relu(tf.add(tf.matmul(X,w_1), b_1))
    drop_out=tf.nn.dropout(h, keep_prob)
    yhat=tf.add(tf.matmul(drop_out,w_2), b_2)
    return yhat

def forwardprop_score(X, w_1, b_1, w_2, b_2):
    h=tf.nn.relu(tf.add(tf.matmul(X,w_1), b_1))
    yhat=tf.nn.sigmoid(tf.add(tf.matmul(h,w_2), b_2))
    return yhat

def Tlinear(S, w_t):
    w=tf.transpose([w_t[1:,0]])
    tl=tf.add(tf.matmul(S, w), w_t[0,0])
    return tl


def Threshold(y_true, score):
    nt,lt=score.shape
    thresh=np.zeros(nt)
    for i in range(nt): #range(nt):
        f1t = 0
        t_candidate=np.array([0,1])
        t_candidate=np.append(t_candidate, score[i,:])
        for j in range(lt+2):
            f1t_j=sklm.f1_score(y_true[i, :], Predict(score[i,:], t_candidate[j]), average='micro')
            if f1t_j>=f1t:
                f1t=f1t_j
                thresh[i]=t_candidate[j]
        print(thresh[i])
    return thresh


def Predict(xscore, threshold):
    lp=len(xscore)
    pred=np.zeros(lp)
    for j in range(lp):
        if xscore[j]>threshold:
            pred[j]=1
    return pred



path='/home/machinelearningstation/PycharmProjects/CSC project'
df=pd.read_table(path+'/data/enron.dat', delimiter=',', header=None, skiprows=1058)
label_num=53
all_accuracy=[]
all_mi_auc=[]
all_sa_auc=[]
all_mi_precision=[]
all_ma_precision=[]
all_sa_precision=[]
all_mi_recall=[]
all_ma_recall=[]
all_sa_recall=[]
all_mi_f1=[]
all_ma_f1=[]
all_sa_f1=[]

print("%s", str(datetime.now()))

target = df[df.columns[-label_num:]].astype(int)
data=df.drop(df.columns[-label_num:],axis=1)
print('X shape: '+str(data.shape))
print('Y shape: '+str(target.shape))

all_X=pd.DataFrame.as_matrix(data)

all_Y=pd.DataFrame.as_matrix(target)
#train_size=0.7
#train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size=1-train_size, random_state=RANDOM_SEED)
kf=KFold(n_splits=5)
for train, test in kf.split(all_Y):
    train_X, test_X, train_Y, test_Y = all_X[train], all_X[test], all_Y[train], all_Y[test]
    print('trainX shape: ' + str(train_X.shape))
    print('testX shape: ' + str(test_X.shape))
    print('trainY shape: ' + str(train_Y.shape))
    print('testY shape: ' + str(test_Y.shape))
    x_size=train_X.shape[1]
    h_size=2048
    y_size=train_Y.shape[1]
    t_size=1

    X=tf.placeholder("float", shape=[None, x_size], name='X')
    Y=tf.placeholder("float", shape=[None, y_size], name='Y')
    S=tf.placeholder('float', shape=[None, y_size], name='S')
    T=tf.placeholder('float', shape=[None, t_size], name='T')

    w_1=init_weights((x_size,h_size), 'w_1')
    b_1=init_weights((1,h_size),'b_1')
    w_2=init_weights((h_size,y_size), 'w_2')
    b_2=init_weights((1,y_size),'b_2')
    w_t=init_weights((y_size+1, t_size), 'w_t')

    yhat=forwardprop(X, w_1, b_1, w_2, b_2, 0.7)
    score=forwardprop_score(X, w_1,b_1, w_2,b_2)
    t=Tlinear(S, w_t)
    cost_s=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))
    cost_t=tf.reduce_mean(tf.pow(Tlinear(S, w_t)-T, 2))
    #cost_t=tf.losses.mean_squared_error(T, t)
    updates_s=tf.train.AdamOptimizer(0.01).minimize(cost_s, name='Adam_LabelScores')
    updates_t=tf.train.AdamOptimizer(0.01).minimize(cost_t, name='Adam_Thresholds')

    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)

    J0s = 0.00
    J1s = 10.00
    epoch=1
    tolerance=1e-10
    while tolerance <= abs(J0s - J1s) and epoch<2000:
        J0s=J1s
        sess.run(updates_s,feed_dict={X:train_X,Y:train_Y})

        J1s=sess.run(cost_s, feed_dict={X:train_X,Y:train_Y})
        epoch += 1
        print('score epoch= '+str(epoch)+', partial loss='+str(J1s))

    train_score=sess.run(score, feed_dict={X: train_X, Y: train_Y})
    train_t=Threshold(train_Y, train_score).reshape((train_X.shape[0], t_size))

    J0t=0.00
    J1t=10.00
    epoch=1
    tolerance=1e-24
    while tolerance <= abs(J0t - J1t) and epoch<10000:
        J0t=J1t
        sess.run(updates_t,feed_dict={S:train_score,T:train_t})
        J1t=sess.run(cost_t, feed_dict={S:train_score,T:train_t})
        epoch += 1
        print('thresh epoch= '+str(epoch)+', partial loss='+str(J1t))


    test_score=sess.run(score, feed_dict={X: test_X,Y: test_Y})

    test_t=sess.run(t, feed_dict={S:test_score})
    y_predict=np.zeros((len(test_score),y_size))
    for i in range(len(test_score)):
        y_predict[i]=Predict(test_score[i], test_t[i])
    y_true = test_Y

    accuracy=sklm.accuracy_score(y_true, y_predict)
    mi_auc=sklm.roc_auc_score(y_true, test_score, average='micro')
    #ma_auc=sklm.roc_auc_score(y_true, test_score, average='macro')
    sa_auc=sklm.roc_auc_score(y_true, test_score, average='samples')
    mi_precision=sklm.precision_score(y_true, y_predict, average='micro')
    ma_precision=sklm.precision_score(y_true, y_predict, average='macro')
    sa_precision=sklm.precision_score(y_true, y_predict, average='samples')
    mi_recall=sklm.recall_score(y_true, y_predict, average='micro')
    ma_recall=sklm.recall_score(y_true, y_predict, average='macro')
    sa_recall=sklm.recall_score(y_true, y_predict, average='samples')
    mi_f1=sklm.f1_score(y_true, y_predict, average='micro')
    ma_f1=sklm.f1_score(y_true, y_predict, average='macro')
    sa_f1=sklm.f1_score(y_true, y_predict, average='samples')
    print(' accuracy = %.2f%% '% (100.*accuracy))
    print(" mi-auc = %.2f%%, mi-auc =%.2f%%, sa-auc=%.2f%% "
              % ( 100.* mi_auc, 100.*mi_auc, 100.*sa_auc))
    print(" mi-precision = %.2f%%, mi-recall =%.2f%%, mi-f1_score=%.2f%% "
              % ( 100.* mi_precision, 100.*mi_recall, 100.*mi_f1))
    print(" ma-precision = %.2f%%, ma-recall =%.2f%%, ma-f1_score=%.2f%% "
              % ( 100.* ma_precision, 100.*ma_recall, 100.*ma_f1))
    print(" sa-precision = %.2f%%, sa-recall =%.2f%%, sa-f1_score=%.2f%% "
              % ( 100.* sa_precision, 100.*sa_recall, 100.*sa_f1))
    all_accuracy.append(accuracy)
    all_mi_auc.append(mi_auc)
    all_sa_auc.append(sa_auc)
    all_mi_precision.append(mi_precision)
    all_ma_precision.append(ma_precision)
    all_sa_precision.append(sa_precision)
    all_mi_recall.append(mi_recall)
    all_ma_recall.append(ma_recall)
    all_sa_recall.append(sa_recall)
    all_mi_f1.append(mi_f1)
    all_ma_f1.append(ma_f1)
    all_sa_f1.append(sa_f1)

    sess.close()

print("%s", str(datetime.now()))

""""
print("mean(precision)= %.2f%%, mean(recall)= %.2f%%, mean(f1_score)= %.2f%%"
      % ( 100.*np.mean(all_precision), 100.*np.mean(all_recall), 100.*np.mean(all_f1)))
print("std(precision)= %.4f, std(recall)= %.4f, std(f1_score)= %.4f"
      % ( np.std(all_precision), np.std(all_recall), np.std(all_f1)))
"""
print(' mean(accuracy) = %.2f%% '% (100.*np.mean(all_accuracy)))
print(' std(accuracy) = %.2f%% '% (100.*np.std(all_accuracy)))
print(" mean(mi-auc) = %.2f%%, mean(sa-auc)=%.2f%% "
              % ( 100.* np.mean(all_mi_auc), 100.*np.mean(all_sa_auc)))
print(" std(mi-auc) = %.2f%%, std(sa-auc)=%.2f%% "
              % ( 100.* np.std(all_mi_auc), 100.*np.std(all_sa_auc)))
print(" mean(mi-precision) = %.2f%%, mean(mi-recall) =%.2f%%, mean(mi-f1_score)=%.2f%% "
              % ( 100.* np.mean(all_mi_precision), 100.*np.mean(all_mi_recall), 100.*np.mean(all_mi_f1)))
print(" std(mi-precision) = %.2f%%, std(mi-recall) =%.2f%%, std(mi-f1_score)=%.2f%% "
              % ( 100.* np.std(all_mi_precision), 100.*np.std(all_mi_recall), 100.*np.std(all_mi_f1)))
print(" mean(ma-precision) = %.2f%%, mean(ma-recall) =%.2f%%, mean(ma-f1_score)=%.2f%% "
              % ( 100.* np.mean(all_ma_precision), 100.*np.mean(all_ma_recall), 100.*np.mean(all_ma_f1)))
print(" std(ma-precision) = %.2f%%, std(ma-recall) =%.2f%%, std(ma-f1_score)=%.2f%% "
              % ( 100.* np.std(all_ma_precision), 100.*np.std(all_ma_recall), 100.*np.std(all_ma_f1)))
print(" mean(sa-precision) = %.2f%%, mean(sa-recall) =%.2f%%, mean(sa-f1_score)=%.2f%% "
              % ( 100.* np.mean(all_sa_precision), 100.*np.mean(all_sa_recall), 100.*np.mean(all_sa_f1)))
print(" std(sa-precision) = %.2f%%, std(sa-recall) =%.2f%%, std(sa-f1_score)=%.2f%% "
              % ( 100.* np.std(all_sa_precision), 100.*np.std(all_sa_recall), 100.*np.std(all_sa_f1)))