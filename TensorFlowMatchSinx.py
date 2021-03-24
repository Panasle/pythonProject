import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

#   拟合y=sinx
Mid_Num = 120  # 中间层神经元个数


class getTensor:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None], "x")
        self.y = tf.placeholder(tf.float32, [None], "y")
        self.lr = tf.placeholder(tf.float32, [], "lr")

        # 第一层，从特征到隐含层
        t = tf.expand_dims(self.x, 1)  # 对的x增加一个维度(?,1)
        w1 = tf.get_variable("w1", [1, Mid_Num], tf.float32)  # 创建tf权重(1,120)
        b1 = tf.get_variable("b1", [Mid_Num], tf.float32)  # 创建tf截距(120,)
        t = tf.matmul(t, w1)  # 特征*权重(?,120)
        t = t + b1  # y=wx+b(?,120)
        t = tf.nn.relu(t)  # 放入激活函数(?,120)

        # 第二层，从隐含层到输出层
        w2 = tf.get_variable("w2", [Mid_Num, 1], tf.float32)  # 创建tf权重（120,1）
        b2 = tf.get_variable("b2", [1], tf.float32)  # 创建tf截距（1,）
        t = tf.matmul(t, w2)  # 隐含层*权重（?，1）
        t = t + b2  # （?，1）

        self.predict = tf.reshape(t, [-1])  # (?,)合并为1维
        loss = tf.square(self.y - self.predict)  # 平方误差(?,)
        self.loss = tf.reduce_mean(loss)  # 求平均误差()

        opt = tf.train.AdamOptimizer(self.lr)  # 优化器Adam，lr学习率,寻找全局最优点的优化算法
        self.train_opt = opt.minimize(loss)  # 最小损失优化？？？？？？？？？？？？？？？？？？？？？？？？？？？？？


def train(session, tensor, xs, ys, lr, epoch):
    for i in range(epoch):  # 循环计算损失
        drop_, loss = session.run((tensor.train_opt, tensor.loss),
                                  {tensor.lr: lr, tensor.x: xs, tensor.y: ys})  # ？？？？？？？？？？？？？？？？？？？？
        if i % 10 == 0:
            print("epoc = %s,loss:%s" % (i, loss))


def predict(session, tensor, Test_x):  # ????????????????????????????????????????????????????????????
    p = session.run(tensor.predict, {tensor.x: Test_x})  # ????????????????????????????????????????????????????????????
    return p


if __name__ == '__main__':
    xs = np.arange(0 * math.pi, 2 * math.pi, 0.001)  # 样本
    ys = np.sin(xs)  # 标签
    plt.plot(xs, ys)
    lr = 0.001  # 步长
    epoch = 2000  # 训练次数
    # plt.show()

    #  创建图
    with tf.Session() as session:
        tensor = getTensor()  # 获取三层神经网络
        session.run(tf.global_variables_initializer())  # 初始化tf变量
        train(session, tensor, xs, ys, lr, epoch)  # 训练网络
        Test_x = np.random.uniform(0 * math.pi, 2 * math.pi, [500])  # 样本
        Test_x = np.sort(Test_x)  # ??????????????????????????????????????????????????
        Test_y = predict(session, tensor, Test_x)

        plt.plot(Test_x, Test_y)
    plt.show()


