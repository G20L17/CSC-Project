import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
from datetime import datetime

RANDOM_SEED = 193
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape, name):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=name)


def forwardprop(X, w_1, b_1, w_2, b_2):
    h=tf.nn.relu(tf.add(tf.matmul(X,w_1), b_1))
    yhat=tf.add(tf.matmul(h,w_2), b_2)
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
    return thresh


def Predict(xscore, threshold):
    lp=len(xscore)
    pred=np.zeros(lp)
    for j in range(lp):
        if xscore[j]>threshold:
            pred[j]=1
    return pred



path='/home/machinelearningstation/PycharmProjects/CSC project'
df=pd.read_table(path+'/data/medical.dat', delimiter=',', header=None, skiprows=1498)
label_num=45
all_precision=[]
all_recall=[]
all_f1=[]

print("%s", str(datetime.now()))

target = df[df.columns[-label_num:]].astype(int)
data=df.drop(df.columns[-label_num:],axis=1)
print('X shape: '+str(data.shape))
print('Y shape: '+str(target.shape))

all_X=data

all_Y=pd.DataFrame.as_matrix(target)
train_size=0.6
train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size=1-train_size, random_state=RANDOM_SEED)

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

yhat=forwardprop(X, w_1, b_1, w_2, b_2)
score=forwardprop_score(X, w_1,b_1, w_2,b_2)
t=Tlinear(S, w_t)

cost_s=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))
cost_t=tf.reduce_mean(tf.pow(Tlinear(S, w_t)-T, 2))

updates_s=tf.train.AdamOptimizer(0.01).minimize(cost_s, name='Adam_LabelScores')
updates_t=tf.train.AdamOptimizer(0.01).minimize(cost_t, name='Adam_Thresholds')

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

J0s = 0.00
J1s = 10.00
epoch=1
tolerance=1e-12
while tolerance <= abs(J0s - J1s) and epoch<15000:
    J0s=J1s
    sess.run(updates_s,feed_dict={X:train_X,Y:train_Y})

    J1s=sess.run(cost_s, feed_dict={X:train_X,Y:train_Y})
    epoch += 1
    print('score epoch= '+str(epoch)+', partial loss='+str(J1s))

'''''
train_yhat=np.zeros((len(train_X),y_size))
for i in range(len(train_X)/batch_size):
    train_yhat[batch_size*i:batch_size*(i+1)] = sess.run(yhat, feed_dict={X: train_X[batch_size*i:batch_size*(i+1)],
                                          Y: train_Y[batch_size*i:batch_size*(i+1)]})

train_yhat[(len(train_X)/batch_size+1)*batch_size:]=sess.run(yhat, feed_dict={X: train_X[(len(train_X)/batch_size+1)*batch_size:],
                                          Y: train_Y[(len(train_X)/batch_size+1)*batch_size:]})
'''

train_score=sess.run(score, feed_dict={X: train_X, Y: train_Y})
train_t=Threshold(train_Y, train_score).reshape((train_X.shape[0], t_size))

J0t=0.00
J1t=10.00
epoch=1
tolerance=1e-22
while tolerance <= abs(J0t - J1t) and epoch<2000:
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

mi_precison=sklm.precision_score(y_true, y_predict, average='micro')
ma_precison=sklm.precision_score(y_true, y_predict, average='macro')
mi_recall=sklm.recall_score(y_true, y_predict, average='micro')
ma_recall=sklm.recall_score(y_true, y_predict, average='macro')
mi_f1=sklm.f1_score(y_true, y_predict, average='micro')
ma_f1=sklm.f1_score(y_true, y_predict, average='macro')
print(" mi-precision = %.2f%%, mi-recall =%.2f%%, mi-f1_score=%.2f%% "
              % ( 100.* mi_precison, 100.*mi_recall, 100.*mi_f1))
print(" ma-precision = %.2f%%, ma-recall =%.2f%%, ma-f1_score=%.2f%% "
              % ( 100.* ma_precison, 100.*ma_recall, 100.*ma_f1))

"""
all_precision.append(precison)
all_recall.append(recall)
all_f1.append(f1)
"""

sess.close()

print("%s", str(datetime.now()))

""""
print("mean(precision)= %.2f%%, mean(recall)= %.2f%%, mean(f1_score)= %.2f%%"
      % ( 100.*np.mean(all_precision), 100.*np.mean(all_recall), 100.*np.mean(all_f1)))
print("std(precision)= %.4f, std(recall)= %.4f, std(f1_score)= %.4f"
      % ( np.std(all_precision), np.std(all_recall), np.std(all_f1)))
"""