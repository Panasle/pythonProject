
def sqrt(y_label, lr):
    x = 1
    y = lambda x:x**2
    y_dao = lambda x:2*x
    y_delta = lambda x:y_label-y(x)
    x_delta = lambda x:lr*y_delta(x)*y_dao(x)

    for time in range(1000):
        x += x_delta(x)
    return x




if __name__ == '__main__':
    lr = 0.001
    for x2 in range(10):
        print("x=%s, sqrt(%s)=%s"%(x2, x2, sqrt(x2, lr)))