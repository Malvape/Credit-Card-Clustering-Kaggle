# Credit-Card-Clustering-Kaggle
My solution to the credit card clustering problem of kaggle.
I did use principal component analysis to separate different types of users of credi card accordin to the [provided dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

the Ipynb. file is [here](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/blob/3D/Solucion.ipynb) there is the detailed step by step solution. And the following are the graphs i made.



```python
n_clusters=30
cost=[]
for i in range(1,n_clusters):
    kmean= KMeans(i)
    kmean.fit(X)
    cost.append(kmean.inertia_)  

plt.plot(cost, 'bx-')

```

![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/6d46b394-ef5f-45a3-a0e7-5e29d4decc44)

from there i decided to take 10 clusters, and plot them.

```python
for c in clusters:
    grid= sns.FacetGrid(clusters, col='cluster')
    grid.map(plt.hist, c)
```


![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/ec6ac654-62af-4608-8be7-9182ffe5fe9a)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/eba1c0b0-a3c4-491e-bd0d-801fadd7bb2e)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/be1bd873-6466-498c-a43b-09a3fb7e6676)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/a1fce305-62d3-417b-8ee9-0af84e8b524d)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/68953d76-e2d5-4ad1-be60-b47483639bd4)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/f154f175-d1d5-484b-b45c-56eb68e2dab0)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/804d6c4a-b457-4cbd-b1d6-6675680c87ed)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/529b0307-638a-4d2b-9e01-6fa286b4bce0)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/2a4b6df5-79ef-446a-a7b6-fef0b97b8001)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/0cce9677-e3a8-4f5c-905c-43d76145a5b6)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/53677a29-f3a7-4454-8329-c8a1b5afb7ea)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/bc7a5194-15e6-4578-a301-262000e5291b)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/f3f0d20a-ea9d-4e1c-b670-61ed850d8edc)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/9a6dc515-7364-4fdd-a1dd-7c3df2f19ad4)
![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/5759b8e0-48a9-4b88-a3a3-643f1dc27d90)

Finally the PCA plot with 3 axis.
```python
dist = 1 - cosine_similarity(X)

pca = PCA(3)
pca.fit(dist)
X_PCA = pca.transform(dist)
X_PCA.shape

x, y, z = X_PCA[:, 0], X_PCA[:, 1], X_PCA[:, 2]

# colors = {0: 'red',
#           1: 'blue',
#           2: 'green', 
#           3: 'yellow', 
#           4: 'orange',  
#           5:'purple'}

names = {0: 'who make all type of purchases', 
         1: 'Very frequent user', 
         2: 'almost no balance left', 
         3: 'frequent user low ticket', 
         4: 'Very infrequent user',
         5:'Wide purchases frequency',
         6:'Pays cash in davance',
         7:'no use of the credit card',
         8:'Low balance, high balance frequency',
         9:'no activity client'
         
         }
  
df = pd.DataFrame({'x': x, 'y':y, 'z':z, 'label':labels}) 
groups = df.groupby('label')

fig = plt.figure(figsize=(20, 13))
ax = fig.add_subplot(111, projection='3d')

for name, group in groups:
    #ax.plot(group.x, group.y, group.z, marker='o', linestyle='', ms=5,
    #        #color=colors[name],
    #        label=names[name], mec='none')
    ax.scatter(group.x, group.y, group.z, marker='o', s=20, label=names[name])

    ax.set_aspect('auto')
    #ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    #ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    #ax.tick_params(axis= 'z',which='both',left='off',top='off',labelleft='off')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")
plt.show()

```

![image](https://github.com/Malvape/Credit-Card-Clustering-Kaggle/assets/41355722/c92597ef-afb6-464c-8f08-aad1f1b4b3fa)



