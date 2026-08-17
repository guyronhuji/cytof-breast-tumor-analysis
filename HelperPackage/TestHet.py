import matplotlib.pyplot as plt
#Local,Global,Area
def TestHeteroMult(DB,UMAPMRK,min_dist=0.001, n_neighbors=60,fname=None,rstate=42):
        from scipy.spatial import distance
        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        import pandas as pd
        import umap
        import numpy as np
        import matplotlib.pyplot as plt

        Out=[]
        fit = umap.UMAP(
               n_neighbors=n_neighbors,
               min_dist=min_dist,
               n_components=2,
               metric='euclidean', random_state=rstate, verbose=False
        )
        X_2d = fit.fit_transform(DB[UMAPMRK]);
        L=DB.Line.unique()
#        print(L)
        for L1 in L:
            m=DB['Line']==L1


            
            xmax=X_2d[:,0].max()
            xmin=X_2d[:,0].min()
            ymax=X_2d[:,1].max()
            ymin=X_2d[:,1].min()
            mx=np.max([xmax,ymax])
            mn=np.min([xmin,ymin])

            m=DB.Line==L1
            b=np.linspace(round(mn,0)-1,round(mx,0)+1,50)
            A,_,_=np.histogram2d(X_2d[m,0],X_2d[m,1],bins=b)
            DD=distance.cdist(X_2d[m],X_2d[m]).flatten()
            hull1 = ConvexHull(X_2d[m])

            Out.append((L1,(A>0).sum(),np.round(np.quantile(DD,0.95),2),hull1.volume))
        return Out


def TestHetero4DB(DB1,DB2,DB3,DB4,UMAPMRK,CNum=1000,min_dist=0.001, n_neighbors=60,fname=None,rstate=42):
        from scipy.spatial import distance
        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        import pandas as pd
        import umap
        import numpy as np
        import matplotlib.pyplot as plt
        DB=pd.concat([
                DB1.sample(CNum,random_state=42),
                DB2.sample(CNum,random_state=42),
                DB3.sample(CNum,random_state=42),
                DB4.sample(CNum,random_state=42)
        ]).copy()
        Out=[]
        fit = umap.UMAP(
               n_neighbors=n_neighbors,
               min_dist=min_dist,
               n_components=2,
               metric='euclidean', random_state=rstate, verbose=False
        )
        X_2d = fit.fit_transform(DB[UMAPMRK]);
        L=DB.Line.unique()
        print(L)
        for L1 in L:
            m=CAll['Line']==L1

            # plt.scatter(X_2d[m,0],X_2d[m,1],c='b',s=1,label=L1);
            # plt.scatter(X_2d[~m,0],X_2d[~m,1],c='r',s=1,label=L2);
            # plt.legend(markerscale=10)
            # if fname is not None:
            #     plt.savefig(fname)
            
            xmax=X_2d[:,0].max()
            xmin=X_2d[:,0].min()
            ymax=X_2d[:,1].max()
            ymin=X_2d[:,1].min()
            mx=np.max([xmax,ymax])
            mn=np.min([xmin,ymin])

            m=DB.Line==L1
            b=np.linspace(round(mn,0)-1,round(mx,0)+1,50)
            A,_,_=np.histogram2d(X_2d[m,0],X_2d[m,1],bins=b)
            # plt.figure(figsize=(5,5))
            # plt.imshow(A>0)
            DD=distance.cdist(X_2d[m],X_2d[m]).flatten()
            hull1 = ConvexHull(X_2d[m])

            print(L1," Local: ",(A>0).sum()," Global: ",np.round(np.quantile(DD,0.95),2)," Area: ",hull1.volume)
            Out.append((L1,(A>0).sum(),np.round(np.quantile(DD,0.95),2),hull1.volume))
        return Out

def TestHeteroMult_2DB(DB1,DB2,UMAPMRK,CNum=1000,min_dist=0.001, n_neighbors=60,fname=None,rstate=42):
        from scipy.spatial import distance
        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        import pandas as pd
        import umap
        import numpy as np
        import matplotlib.pyplot as plt
        DB=pd.concat([
                DB1.sample(CNum,random_state=42),
                DB2.sample(CNum,random_state=42),
        ]).copy()
        Out=[]
        fit = umap.UMAP(
               n_neighbors=n_neighbors,
               min_dist=min_dist,
               n_components=2,
               metric='euclidean', random_state=rstate, verbose=False
        )
        X_2d = fit.fit_transform(DB[UMAPMRK]);
        L=DB.Line.unique()
        print(L)
        for L1 in L:
            m=CAll['Line']==L1

            # plt.scatter(X_2d[m,0],X_2d[m,1],c='b',s=1,label=L1);
            # plt.scatter(X_2d[~m,0],X_2d[~m,1],c='r',s=1,label=L2);
            # plt.legend(markerscale=10)
            # if fname is not None:
            #     plt.savefig(fname)
            
            xmax=X_2d[:,0].max()
            xmin=X_2d[:,0].min()
            ymax=X_2d[:,1].max()
            ymin=X_2d[:,1].min()
            mx=np.max([xmax,ymax])
            mn=np.min([xmin,ymin])

            m=DB.Line==L1
            b=np.linspace(round(mn,0)-1,round(mx,0)+1,50)
            A,_,_=np.histogram2d(X_2d[m,0],X_2d[m,1],bins=b)
            # plt.figure(figsize=(5,5))
            # plt.imshow(A>0)
            DD=distance.cdist(X_2d[m],X_2d[m]).flatten()
            hull1 = ConvexHull(X_2d[m])

            print(L1," Local: ",(A>0).sum()," Global: ",np.round(np.quantile(DD,0.95),2)," Area: ",hull1.volume)
            Out.append((L1,(A>0).sum(),np.round(np.quantile(DD,0.95),2),hull1.volume))
        return Out



import torch
import numpy as np
import matplotlib.pyplot as plt
def average_knn_distance(x, k: int, batch_size: int = None):
    """
    Compute for each point in x the average distance to its k nearest neighbors.
    
    Args:
      x           : Tensor of shape (N, 2)
      k           : number of nearest neighbors
      batch_size  : if None, does full N×N; otherwise splits into batches of rows
    Returns:
      avg_dists   : Tensor of shape (N,) with the mean distance to the k neighbors
    """
    # 1) pick device
    x=torch.tensor(x,dtype=torch.float32)
    if torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    x = x.to(device)

    N = x.size(0)
    # We'll collect averages in this list if using batching
    out = []

    # If no batch_size is given, do the whole matrix at once
    if batch_size is None:
        # full NxN distance matrix
        D = torch.cdist(x, x)                           # (N, N)
        # find k+1 smallest (including zero self-distance)
        vals, _ = torch.topk(D, k=k+1, largest=False)   # (N, k+1)
        # drop the first column (self) and average
        avg = vals[:, 1:].mean(dim=1)                   # (N,)
        return avg.cpu().numpy()

    # Otherwise, do row-wise in batches
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Xi = x[start:end]                               # (b, 2)
        Di = torch.cdist(Xi, x)                         # (b, N)
        vals, _ = torch.topk(Di, k=k+1, largest=False)  # (b, k+1)
        avg_i = vals[:, 1:].mean(dim=1)                 # (b,)
        out.append(avg_i.cpu())

    out=torch.cat(out, dim=0)
    return out.numpy()
