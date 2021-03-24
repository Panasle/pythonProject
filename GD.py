'''
梯度下降求解 y = sqrt(x)


'''
def sqrt(a):
    lr = 0.001
    x = 1
    y = lambda x:x**2
    y_dao = lambda x:2*x
    y_delta = lambda x:a-y(x)
    x_delta = lambda x,lr:lr*y_delta(x)*y_dao(x)

    for _ in range(5000):
        x+=x_delta(x,lr)
    return x






if __name__ == '__main__':
    for i in range(2,10):
        print("sqrt(%s) = %s"%(i,sqrt(i)))