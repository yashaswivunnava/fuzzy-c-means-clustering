import math
import numpy as np
import xlrd as xl
import matplotlib.pyplot as plt
def initU(num_clusters,n):

    c=0
    U=[]
    while c<num_clusters:
        temp=[]
        for N in range(n):
            temp.append(np.random.uniform(0,1))
        U.append(temp)
        c=c+1
    colsum=np.sum(U,axis=0)
    U=np.array(U)
    return np.divide(U,colsum)
def distance_points(Z,initial_data,m,number_cluster):

    v=[[0 for i in range(2)]for j in range(number_cluster)]
    initial_data=np.power(initial_data,m)
    for i in range(number_cluster):
        v[i][0]=(np.dot(initial_data[i],Z[0]))/(initial_data[i].sum())
        v[i][1]=(np.dot(initial_data[i],Z[1]))/(initial_data[i].sum())
    return v
def distance(Z,V,num_clusters):    Z=Z.T
    Dika=[]
    for i in range(num_clusters):
        temp=[]
        for k in range(len(Z)):
            x=(Z[k][0]-V[i][0])**2
            y=(Z[k][1]-V[i][1])**2
            temp.append(x+y)
        Dika.append(temp)
    return Dika

def partition_update(dataset,distance,num_clusters,m):
     updated_u=[[0 for a in range(len(dataset[0]))]for b in range(num_clusters)]
     for i in range(num_clusters):
         for k in range(len(dataset[0])):
             if (math.sqrt(distance[i][k])>0):
                 temp=0
                 for j in range(num_clusters):
                     temp+=((distance[i][k])/(distance[j][k]))**(2.0/(m-1))
                 updated_u[i][k]=1.0/temp
             elif (math.sqrt(distance[i][k])==0):
                 for e in range(num_clusters):
                     if(e!=i):
                         updated_u[i][k]=0

                     else:

                         updated_u[i][k]=1

             else:

                 continue
     return updated_u

def objective_function(dataset,distance,num_clusters,m):
    J=0
    for i in range(num_clusters):
        for k in range(len(dataset[0])):
            J=J+((dataset[i][k]**m)*distance[i][k])
    return J
read_f=xl.open_workbook("Data Sets.xlsx")
need_sheet=read_f.sheet_by_name("Data Set 5")
row_data = need_sheet.nrows
column_data=need_sheet.ncols
data_list=[]
i=2
while(i<row_data):
    j=0
    temp_data=[]
    while(j<column_data):
        data = need_sheet.cell_value(i,j)
        temp_data.append(data)
        j=j+1
    data_list.append(temp_data)
    i=i+1
np.random.shuffle(data_list)
sort_1=[]
sort_4=[]
for idx in range(len(data_list)):
    if idx%5==0:
        sort_4.append(data_list[idx])
    else:
        sort_1.append(data_list[idx])
cluster=range(2,11)
n=len(sort_1)
Z=np.array(sort_1)
Z=Z.T
m=input("Enter the value of m(Fuzziness):")
All_V_vec=[]
All_U_vec=[]
All_U_new=[]
dvector=[]
countervector=[]
for a in range(len(cluster)):
    U_prev=initU(cluster[a],n)
    count=1

    while 1:

        v=distance_points(Z,U_prev,m,cluster[a])
        d=distance(Z,v,cluster[a])
        updated_u=np.array(partition_update(U_prev,d,cluster[a],m))
        diff=[]
        for i in range(cluster[a]):
            temp1=[]
            for k in range(len(Z[0])):
                temp1.append(abs(updated_u[i][k]-U_prev[i][k]))
            diff.append(max(temp1))
        if(max(diff)<0.0001):
            break
        
else:

            U_prev=updated_u
        count=count+1
    All_U_vec.append(U_prev)
    All_U_new.append(updated_u)
    All_V_vec.append(v)
    dvector.append(d)
    countervector.append(count)

All_J=[]
for i in range(len(countervector)):
    tba=objective_function(All_U_new[i],dvector[i],cluster[i],m)
    All_J.append(tba)
ratio=[0 for i in range(7)]
for t in range(1,8):
    temp=(All_J[t]-All_J[t+1])/(All_J[t-1]-All_J[t])
    ratio[t-1]=abs(temp)
optimal_cluster = ratio.index(min(ratio))+3
optimal_U = All_U_new[optimal_cluster-2]
optimal_V=All_V_vec[optimal_cluster-2]
clusternum=optimal_U.argmax(axis=0)
for i in range(len(clusternum)):
    sort_1[i].append(clusternum[i])
sort_1=np.array(sort_1).T
cluster_area=(10)
area_distance_points=(15)
plt.title('sort_1')
plt.scatter(list(sort_1[0]),list(sort_1[1]), s=cluster_area, c=list(sort_1[2]))
optimal_center=np.array(optimal_V).T
plt.scatter(list(optimal_center[0]),list(optimal_center[1]), s=area_distance_points, c='k')
plt.show()
C=np.array(sort_4).T
initU_c=initU(optimal_cluster,len(C[0]))
count_1=0

while 1:

     max_v=optimal_V
     distance_class=distance(C,initU_c,optimal_cluster)
     new_U=np.array(partition_update(initU_c,distance_class,optimal_cluster,m))
     diff_c=[]
     for i in range(optimal_cluster):
        temp1_c=[]
        for k in range(len(C[0])):
             temp1_c.append(abs(new_U[i][k]-initU_c[i][k]))
        diff_c.append(max(temp1_c))
     if(max(diff_c)<0.0001):
         break

     else:

         initU_c=new_U
     count_1=count_1+1
clusternum=new_U.argmax(axis=0)
for i in range(len(clusternum)):
    sort_4[i].append(clusternum[i])
classif=np.array(sort_4).T
plt.title('sort_4')
plt.scatter(list(classif[0]),list(classif[1]), s=cluster_area, c=list(classif[2]))
optimal_center=np.array(optimal_V).T
plt.scatter(list(optimal_center[0]),list(optimal_center[1]), s=area_distance_points, c='k')
plt.show()
