import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
#题目：拟合y = sinx

Mid_Num = 120#中间层神经元个数
class getTensor:
    def __init__(self):
        self.x = tf.placeholder(tf.float32,[None],"x")
        self.z = tf.placeholder(tf.float32,[None],"z")
        self.lr = tf.placeholder(tf.float32,[],"lr")

        # t = tf.layers.dense(self.x,Mid_Num,activation=tf.nn.relu)
        t = tf.expand_dims(self.x,1)    # [None,1]
        w1 = tf.get_variable("w1",[1,Mid_Num],tf.float32)
        b1 = tf.get_variable("b1",[Mid_Num],tf.float32)
        t = tf.matmul(t,w1)  # [None,Mid_Num]
        t = t+b1
        t= tf.nn.relu(t)

        # t = tf.layers.dense(t,1)
        w2 = tf.get_variable("w2",[Mid_Num,1],tf.float32)
        b2 = tf.get_variable("b2",[1],tf.float32)
        t = tf.matmul(t,w2)
        t = t+b2
        self.predict = tf.reshape(t,[-1])
        loss = tf.square(self.z - self.predict)#[-1]
        self.loss = tf.reduce_mean(loss)

        opt = tf.train.AdamOptimizer(self.lr)#优化器设置
        self.train_opt = opt.minimize(loss)#优化
def train(session,Tesnsor,xs,zs,lr,epoch):
    for i in range(epoch):
        _,loss = session.run((Tesnsor.train_opt,Tesnsor.loss),{Tesnsor.lr:lr,Tesnsor.x:xs,Tesnsor.z:zs})
        if i%10 ==0:
            print("epoc = %s,loss:%s"%(i,loss))



def predict(session,Tesnsor,Test_x):
    p = session.run(Tesnsor.predict,{Tesnsor.x:Test_x})
    return p








if __name__ == '__main__':
    xs = np.arange(0,2*math.pi,0.001)#样本
    zs = np.sin(xs)#标签
    plt.plot(xs,zs)
    lr = 0.001#步长
    epoch = 1000


    #  创建图
    with tf.Session() as session:
        Tesnsor = getTensor()  # 获取三层神经网络
        session.run(tf.global_variables_initializer())
        train(session,Tesnsor,xs,zs,lr,epoch)  # 训练网络
        Test_x = np.random.uniform(0,2*math.pi,[500]) # 样本
        Test_x = np.sort(Test_x)
        Test_y = predict(session,Tesnsor,Test_x)

        plt.plot(Test_x,Test_y)
    plt.show()