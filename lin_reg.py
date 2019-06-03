import pandas as pd
x = [1.0,1.6,3.4,4.0,5.2]
y = [1.2,2.0,2.4,3.5,3.5]

def suma(a):
    sum = 0
    for i in range(len(a)):
        sum = a[i] + sum
    return sum

def lin_reg(m,n):
    x1= pd.array(m)
    y1 = pd.array(n)
    x2=x1*x1
    n= len(m)

    a= (suma(x1*y1)*n)-(suma(x1)*suma(y1))/(suma(x2)*n)-pow(suma(x1),2)
    b= ((suma(y1)*suma(x2))-(suma(x1*y1))*suma(x1))/(suma(x2)*n)-pow(suma(x1),2)
    return a,b

print(lin_reg(x,y))
