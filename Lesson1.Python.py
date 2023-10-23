# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:52:55 2023

@author: Lucien C
"""

import math as m
import pandas

print('goodbye')
type('goodby')
a = 3
b = 2
print(a/b)
print(a**b)
c = 3.4
print(a+c)
print(5**5)
print("coffee \nkaffe")
f = float(input())
print(f + 2.5)
print(223/73)
print((1+1/10)**10)
'sparrow' > 'eagle'
a = 2
b = 3
bob = a*b
print("ushfudhgfugdugyu {} ehhhufuhefhu {} jjfh {}".format (a,b,bob))
print(7//2)
print((2*5+3**2)/2.5)

R = 0.08206
n = input("moles")
n = float(n)
P = input("meef")
P= float(P)
a = R*n*P
print (" ijhghuguigj {}".format(a))

r = float(input("r"))
h = float(input("h"))
v = 1/3*math.pi*r**2*h
print("trgeghe {} ".format(v))


import math as m
ang = float(input("Enter an angle"))
ang_rad = ang*m.pi/180
si = m.sin(ang_rad)
print("The ine of {} is {}".format(ang,si))

num1 = float(input("EIUE"))
num2 = input("Enter")
sum = num1 + num2
product = num1*num2
print("hufbfbh {} {} {}".format(num1,num2,sum))
print("hufbfbh {} {} {}".format(num1,num2,product))

T = float(input("Température:"))
K = 273.15 + T
print("résultat {} ".format(K))

C = float(input("Coté"))
V = C**3
A = 6*(C**2)
print("jbhfebhyrbg {} {} ".format(V,A))


nb10 = float(input("10"))
nb20 = float(input("20"))
nb50 = float(input("50"))
total = nb10*10+nb20*20+nb50*50
print("jirfhghu {} ".format(total))

Ea = float(input("EA"))
R = 8.3144621
A = 3.718*10**13
k = A*math.exp(-Ea/(R*T))
print("resultat {} ".format(k))

a = float(input("A"))
b = float(input("B"))
L = float(input("L"))

c = a**2 + b**2-2*(a*b*m.cos(L))
print ("resulat {}".format(m.sqrt(c)))

weight = float(input("weight"))
height = float(input("height"))
BMI = weight/height**2
if BMI<18.5:
    print("normal")
else :
    print("obese")
    
n1 = int(input("1"))
n2 = int(input("2"))
if n1%n2 == 0 :
    print("divisible")
else :
    print(n1%n2)

n1 = int(input("1"))
n2 = int(input("2"))
print(m.min(n1,n2))


def price (a):
    if a<100 :
        return 0
    elif 100<a<200 :
        return (a-100)*5
    elif a>200:
        return  500 + (a-200)*10

num =int(input("nombre"))
while num>0:
    res=num//3
    print("The division of {} by 3 gives {}".format(num,res))
    num=int(input("enter an integer value "))
print("where done")

num = int(input("Enter"))
ndiv = 0
while num>0:
    res = num// 3
    print("The integer division of {} by 3 gives {}".format(num,res))
    ndiv = ndiv+1
    print("number of divisions so far : {}".format(ndiv))
    num=int(input("enter an integer value:"))
print("We're done")
print ("Total number of divisions: {}".format(ndiv))

num = 1
ndiv = 0
while num>0 and num<211:
    if num%3==0:
        print(f"the number is {num}")
        ndiv = ndiv+1
    num=num+1
print("We're done")
print ("Total number of divisions: {}".format(ndiv))

n=0
s=0
while n<10:
    a = int(input("nombre?"))
    s = s+a
    n = n+1
print("ok c'est {}".format(s/10))

f=1
n=5
i=1
while i<n+1:
    f = f*i
    i = i+1
print(f)
    
n= int(input("enter"))
for i in range (1,n+1):
    q_i=i**2
    print(q_i)

n= int(input("enter"))
sum=0
for i in range (1,n+1):
    sum=sum+i
print(sum)

n= int(input("enter"))
sum=0.0
for j in range (1,n+1):
    q=((j+1)/j**2)
    sum=sum+q
print("the sum is {: .2f}".format(sum))

n = int(input("Entrer"))
sum=1
for i in range (1,n+1):
    sum = sum*i
print(sum)    

for i in range(1,10):
    for j in range(1,10):
        print("{}{}".format(i,j))
        
for i in range(1,10):
    for j in range(1,10):
       if i!=j:
          print("{}{}".format(i,j))
          
a = int(input("enter"))

for i in range (1,99):
    for j in range (1,4):
        print("{}".format(i),end="")
        
for i in range (1,10):
    for j in range (1,10):
        for k in range (1,10):
             if (i+j+k)==22:
                print("{}{}{}".format(i,j,k))

for i in range (0,10):
    for j in range (0,10):
        for k in range (0,10):
             if i**3+j**3+k**3==(i*100+j*10+k):
                print("{}{}{}".format(i,j,k))
                
                
n = int(input("ENtrer"))
for i in range (1,n):
    if(n%i==0):
        print(i)
        
num = int(input("Entrer"))
sum =0
for i in range (0, num+1):
    odd_num = 2*i+1
    print("the odd number is {}".format(odd_num))
    sum = sum +odd_num
print("The sum of the first {} odd numbers is {}".format(num,sum))

num = int(input("Entrer"))
for i in range (2,n):
    if num%i==0:
        print("non")
        break
print("yes")


num = int(input("Entrer"))
a=0
b=1
c=0
for i in range (1,num+1):
    c=b
    b=b+a
    a=c
    print(b)
    

LIIISST = ['ihfuhfufufh',2]
LIIISST[0] 

list = ['test1','test2','test3','test4','test5','test6']
print(list[2])
list.append('test7')
print(list)
list[0]="heggr"
tuple_pres = ('Joe','Biden','2021-01-20','Democratic')

integer = [1,2,3,4]
print(integer)
smooth=[3.14,7,-2+3j,'water',False,[1,2]]
print(smooth)
long_smooth = len(smooth)
print(long_smooth)
smooth[2:5]
smooth[::2]
print(smooth[5] [1])
smooth[3][4]

n = int(input("Entrez"))

list1=[]
for i in range (1,n+1):
    list1.append(1/i)
print(sum(list1))

line = input("Enter in a single and seperated by spaces,  the temperatures")
smooth_txt = line.split()
print("List is now {}".format(smooth_txt))
temp=[]
for element in smooth_txt:
    value = float(element)
    temp.append(value)
print("the final list is {}".format(temp))  

n = int(input("Entrer"))
list2 =[]
for i in range(1,n+1):
    list2.append(i**2)
list2    

n = int(input("Entrer"))
list2 =[]
list3 = list2(range(n))   
list3

list9 =[]
list90 =[]
summm=0
name = "a"
while (name !="") :
    name = input("Enter a name :")
    if(name==""):
       break
    list9.append(name)
    name = int(input("Enter a grade :"))
    list90.append(name)
print(list9)
print(list90)
for e in list90:
    summm = summm + e
print(summm/len(list90))

summm=0
list_num = []
numb = 2
while (numb!=""):
    numb = input("Number")
    if(numb==""):
      break
    list_num.append(int(numb))

print(list_num)
for e in list_num:
    summm = summm + e
print(summm/len(list_num))

phrase = input("Entrez les noms")
list_phrases = phrase.split('@')
for e in list_phrases:
    
    print("Hi" ,e )
len(list_phrases)

list_a = ["H","C","N","O","S","CI"]
list_b = [1,12,14,16,32,35]
list_c = []
t=0
somme =0
m=0
for e in list_a:
   t= input("Combien de {}".format(e))
   list_c.append(int(t))
for i in range (0,len(list_c)):
    somme = somme + list_b[i]*list_c[i]
    
print(somme)     

n = input("n?")
list_coef=[]
res=0
x = float(input("x"))
for i in range (0,int(n)):
    coef = input("Coef de {}".format(i))
    list_coef.append(int(coef))
for i in range (0,len(list_coef)):
    res = res + list_coef[i]*x**i
print(res)


Dict = dict([(1,'Geeks'),(2,'For')])
print(Dict)
    
    
Dict = {'Dict1':{1:'Geeks'},'Dict2':{'Name':'For'}}

keys = ['Ten','Twenty','Thirty']
values = [10,20,30]

sample_dict = {
    "name":"Kelly",
    "age":25,
    "salary":8000,
    "city":"New york"}


Atome_dict = {
    "H":{"Melting point":14,"Boiling point":20},
    "He":{"Melting point":1,"Boiling point":4},
    "Li":{"Melting point":453,"Boiling point":1603},
    "Be":{"Melting point":1560,"Boiling point":2742},
    "B":{"Melting point":2349,"Boiling point":4200},
    "C":{"Melting point":3915,"Boiling point":3915},
    "N":{"Melting point":63,"Boiling point":77},
    "O":{"Melting point":54,"Boiling point":90},
    "F":{"Melting point":53,"Boiling point":85},
    "Ne":{"Melting point":25,"Boiling point":27}}

a = input("Entrer le symbole de l'élement que vous voulez rechercher: ")
b = int(input("Entrer la temperature: "))
if (b<Atome_dict[a]["Melting point"]):
    print("SOLID")
elif (b>Atome_dict[a]["Boiling point"]):
    print("GAS")
else:
    print("LIQUID")

r=0
Bank_dict = {
    "ANZ":{"Years 1 & 2":2.3,"Years 3":4.1},
    "Bank of Australia":{"Years 1 & 2":0.1,"Years 3":5},
    "Commonwealth Bank":{"Years 1 & 2":3.5,"Years 3":3.8},
    "Westpac":{"Years 1 & 2":3.7,"Years 3":3.7}}
t = input("Entrez le nom de la banque")
p = int(input("Entrez le montant du mortgage"))
a = int(input("Entrer le nombre d'année à rembourser"))
r = p
if(a<=2):
    for i in range (1,a):
        r = r + r*Bank_dict[t]["Years 1 & 2"]*0.01
        
else:
    for i in range (1,a):
        r = r + r*Bank_dict[t]["Years 1 & 2"]*0.01   
    for i in range (1,a-2):
         r = r + r*Bank_dict[t]["Years 3"]*0.01  
print( r)

def my_function(**kid):
    print("his last name is"+kid["lname"])
my_function(fname="Tobias",lname="hgedfety")

def max(a,b):
    if a>b:
        return a 
    else :
        return b
def fonc(a,b,c,d,e):
    max=a
    min =a
    list_a = [a,b,c,d,e]
    for el in list_a:
        if max<el:
            max = el
        if min>el:
            min=el
    print(max,min)
    return list_a
        
try: 
    num = int(input("Enter: "))
except :
    print("Erreur")
else:  
    if (num%2)==0:
        print("{0} is Even".format(num))
    else :
        print("{0} is Odd".format(num))

def Test ():
   try:
      num=int(input("Enter a number : "))
   except:
    print("Wrong number")
    Test()
   else:
    if (num%2)==0:
        print("{} is Even".format(num))
    else:
        print("{} is not Even".format(num))

def Test2 ():
   try:
      num=int(input("Enter a number : "))
   except:
    print("Wrong number")
    Test2()
   else:
       for i in range (2,num):
           if num%i==0:
               print("non")
               break
       print("yes")
Test2()


def longest_prefix(list_a):
    # your code here
    
    verif2=""
    verif = list_a[0]
    n=0
    if (len(list_a)>1):
        for k in range (1,len(list_a)):
            for i in range (0,len(verif)-1):
                if i<len(list_a[k]):
                    if verif[i] == list_a[k][i]:
                        n= n+1
                    else :               
                        break
                else :
                    break
        
            verif2 =""
        
            for j in range (0,n):
                    verif2=verif2+verif[j]
                    
            verif = verif2
            n=0
    else :
        verif2 = list_a[0]
    
    return verif2 

print(longest_prefix(["flow", "flower", "flight"]))


import numpy
nel = int(input("entrer le bn d'element"))
lvec = []
for i in range (nel):
    comp = input("Enter the value of component{}".format(i))
    lvec.append(float(comp))
vec = numpy.array(lvec)
print(vec)
 
import numpy as np
nel=int(input("data"))
vec=np.zeros(nel)
for i in range (nel):
    comp=input("value {}".format(i))
    vec[i]=float(comp)
vec=np.array(lvec)
print(vec)

import numpy as np
a = np.array([[2,3],[1,2]])
b = np.array([[1,0],[3,1]])

print(np.linalg.inv(a*b) == np.linalg.inv(a)*np.linalg.inv(b))
np.linalg.inv(a*b)
np.linalg.inv(a)*np.linalg.inv(b)

a = np.array([[1,1],[1,2]])
b = np.array([[4,1],[3,1]])
c = np.array([[24,7],[31,9]])

np.linalg.inv(a).dot(c).dot(np.linalg.inv(b))

H = [2.07,2.62,3.22,2.59,4.87,1.19,3.95]
vect = np.array(H)


# S09_4 Convert Angstroms to Nanometers
import numpy as np
exo1 = np.linspace(1,5,21)
vect = exo1*0.1
print(vect)




# S09_5 Table of values
x0 = int(input("x0"))
s = float(input("s"))
init = int(input("Initial value?"))
fin = int(input("Final value?"))
n = int(input("Number of points that the table must have: "))

import math  

vect_x = np.linspace(init,fin,n)
vect_y = np.zeros((n,2))
for i in range (0,n):
    vect_y[i,0]=vect_x[i]
    vect_y[i,1] = (1/math.sqrt(2*math.pi))*math.exp((-1/2)*((i-x0)**2)/(s**2))

print(vect_y)
 

# S09_6 Sea temperature statistics  
temp_mar = [13.8,13.3,13.9,15.0,16.4,20.0,21.9,22.3,22.0,21.2,18.8,16.0]
months = ["January","February","March","April","May","June","July","August","September","October","Novemeber","December"]
np.array(temp_mar)
print(temp_mar)
print("The average sea surface temperature in 2014 is {} °C".format(np.mean(temp_mar)))
print("The month in which the sea surface has been the coldest was {} and its temperature was {} °C".format(months[np.argmin(temp_mar)],np.min(temp_mar)))    
print("The month in which the sea surface has been the warmest was {} and its temperature was {} °C".format(months[np.argmax(temp_mar)],np.max(temp_mar)))   


# S09_10 Exam grades
def exam_grades():
    num = int(input("Number of student? "))
    vect_40 = np.zeros(num)
    vect_60 = np.zeros(num)
    table = np.zeros((num,4))
    q_3 = np.zeros(num)
    for i in range (0,num):
        vect_40[i]=(int(input("Enter the first grade: ")))
        vect_60[i]=(int(input("Enter the first grade: ")))
        table[i,0]= i+1
        table[i,1]= vect_40[i]
        table[i,2]= vect_60[i]
        table[i,3]= vect_40[i]*0.4 + vect_60[i]*0.6
        q_3[i] = vect_40[i]*0.4 + vect_60[i]*0.6
        
    print("The maximum total grade is {} and belong to index {}".format(np.max(q_3),np.argmax(q_3)+1))
    print("The minimum total grade is {}".format(np.min(q_3)))
    print("The average total grade is {}".format(np.mean(q_3)))
    return table

    


import matplotlib.pyplot as plt
import numpy as np
import math as mat
x = np.linspace(-2,2,101)
plt.xlabel("x")
plt.ylabel("f(x)")
y2= x**2
plt.plot(x,y2,'g',label="x**2")
y3 =x**3
plt.plot(x,y3,'ro',label="x**3")

y4=x**4
plt.plot(x,y4,'b',label="x**4")
plt.legend()
plt.show()

n = int(input("nb"))
x = np.linspace(-1,1,n)
y = np.cos(2*np.pi*x)
y2 = np.sin(2*np.pi*x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y,'g',label="2pix")
plt.plot(x,y2,'bo',label="3pix")
plt.legend()
plt.savefig("cos2pix.png")
plt.show()


n = int(input("nb"))
x = np.linspace(0,4,n)
y = np.sin(np.pi*x)*np.sin(20*np.pi*x)*np.exp(-x)
plt.plot(x,y,'g',label = 'fonc')
plt.legend()
plt.show()

n = int(input("nb"))
T = int(input("T"))
x = np.linspace(2,10,n)

y = 0.08206*T/x
plt.plot(x,y,'g',label = 'fonc')
plt.legend()
plt.show()


x = np.linspace(-1,1,100)
s = float(input("s"))
x0 = float(input("x0"))
y = (1/np.sqrt(2*np.pi))*np.exp((-1/2)*((x-x0)**2)/(s**2))
plt.plot(x,y,'g',label = 'fonc')
plt.title("Gaus")
plt.legend()
plt.show()



import matplotlib.pyplot as plt
n = int(input("nb"))
x = np.linspace(0,3,n)
plt.xlabel("x")
plt.ylabel("f(x)")
y2= np.exp(-x)
y=np.sin(3*np.pi*x)*np.exp(-x)
plt.plot(x,y,'g',label="e-x")
plt.plot(x,y2,'bo',label="sin(3pix)e-x")
plt.legend()
plt.show()


n = int(input("How many?"))
for i in range (0,n):
    x = np.linspace(-1,1,100)
    s = float(input("s"))
    x0 = float(input("x0"))
    y = (1/np.sqrt(2*np.pi))*np.exp((-1/2)*((x-x0)**2)/(s**2))
    a = input("line : ")
    nom = input("name ? ")
    plt.plot(x,y,a,label = nom)

plt.title("Gaus")
plt.legend()
plt.show()