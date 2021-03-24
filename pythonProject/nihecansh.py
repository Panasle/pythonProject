import tensorflow as tf
import math
class Tensor:#网络结构
    def __init__(self):
        self.value = tf.placeholder(tf.float32,[None],"value")
        self.lable = tf.placeholder(tf.float32,[None],"lable")
        self.lr = tf.placeholder(tf.float32,[],"lr")

        a = tf.get_variable("a",[],tf.float32)
        b1 = tf.get_variable("b1",[],tf.float32)
        b2 = tf.get_variable("b2",[],tf.float32)
        g = tf.get_variable("g",[],tf.float32)
        predict = a+b1*tf.sin(((2*math.pi)/g)*self.value)+b2*tf.cos(((2*math.pi)/g)*self.value)
        #predict = tf.cast(predict,tf.float32,"cast")


        self.loss = tf.reduce_mean(tf.square(predict-self.lable))

        opt = tf.train.GradientDescentOptimizer(self.lr)
        self.train_opt = opt.minimize(self.loss)
        self.summary = tf.summary.scalar("loss",self.loss)



class Model:#训练和预测，模型保存，可视化
    def __init__(self,lr=0.001,epoch = 200):
        self.Tensor = Tensor()
        self.session = tf.Session()
        self.savePath = "./model/nihe"
        self.lr = lr
        self.epoch = epoch
        self.saver = tf.train.Saver()
        self.logdir = "./logs"


        try:
            self.saver.restore(self.session,self.savePath)
            print("Sucess restor model %s"%self.savePath)
        except:
            print("Fail restore model %s"%self.savePath)
            self.session.run(tf.global_variables_initializer())

    def train(self,value,lable):
        print("Start train...")
        fw = tf.summary.FileWriter(self.logdir,graph=tf.get_default_graph())
        globalStep=0
        with fw:
            for epoc in range(self.epoch):
                _,loss,summary = self.session.run([self.Tensor.train_opt,self.Tensor.loss,self.Tensor.summary],{self.Tensor.value:value,self.Tensor.lable:lable,self.Tensor.lr:self.lr})
                fw.add_summary(summary,globalStep)
                if epoc%10 ==0:
                    print("epoch:%s,loss:%s",epoc,loss)
                    self.saver.save(self.session, self.savePath)
                    print("Save model success")
            print("Finish train!")


if __name__ == '__main__':
    #read dataset
    value=[0,00,0.004]
    lable = []
    #=====
    model = Model()
    model.train(value,lable)