#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn import metrics
from s_dbw import SD
from itertools import product

class Metrics: 
    def __init__(self):
        pass
    
    
    def divide(self, data, labels):
        """
        Support function prepare labeled cluster_data for the folloving metrics:
        * Root-mean-square standard deviation
        * R-squared
        * Xie-Beni index
        """
        clusters = set(labels)
        clusters_data = []
        for cluster in clusters:
            clusters_data.append(data[labels == cluster, :])
        return clusters_data
    
    
    def cohesion(self, data, labels):
        """
        Cohesion calculates the sum of squared distances from each data point to its respective centroid.
        """
        clusters = sorted(set(labels))
        sse = 0
        
        for cluster in clusters:
            cluster_data = data[labels == cluster, :]
            centroid = cluster_data.mean(axis = 0)
            sse += ((cluster_data - centroid)**2).sum()
            
        return sse
    
    
    def sse_2(self, clusters, centroids):
        result = 0
        for cluster, centroid in zip(clusters, centroids):
            result += ((cluster - centroid) ** 2).sum()
        return result
    
    
    def separation(self, data, labels, cohesion_score):
        """
        Separation calculation leverages the fact that the following equation always holds true: TSS = WSS + BSS
        where TSS is the total sum of squared distances from each data point to the overall centroid. WSS is cohesion and BSS is separation.
        """
        # calculate separation as SST - SSE
        return self.sst(data) - cohesion_score
    
    
    def sst(self, data):
        return self.cohesion(data, np.zeros(data.shape[0]))
    
    
    def sst_2(self, data, centroids):
        c = self.get_centroids([data])
        #c = centroids
        return ((data - c) ** 2).sum()
    
    def get_centroids(self, clusters):
        centroids = []
        for cluster_data in clusters:
            centroids.append(cluster_data.mean(axis=0))
        return centroids
    
    def rmsstd_index(self, data, clusters, centroids):
        """
        RMSSTD first computes the the sum of squared distances from each data point to its respective centroid, which is SSE or cohesion. 
        Then, it divides the value by the product of the number of attributes and the degree of freedom, which is calculated as the number 
        of data points minus the number of clusters. Lastly, the we take the square root of the value to obtain RMSSTD.
        """
        df = data.shape[0] - len(clusters)
        attribute_num = data.shape[1]
        
        #return (self.cohesion(clusters, centroids) / (attribute_num * df)) ** .5
        return (self.sse_2(clusters, centroids) / (attribute_num * df)) ** .5
    
    
    def rs_index(self, data, clusters, centroids):
        """
        R-squared can be expressed in terms of separation and cohesion as follows: R-squared = separation / (cohesion + separation)
        """
        #sst = self.sst(data)
        #sse = self.cohesion(clusters, centroids)
        sst = self.sst_2(data, centroids)
        sse = self.sse_2(clusters, centroids)
        
        return (sst - sse) / sst
    
    
    def make_clusters(self, labels):
        """
        Support function
        Preparing labels for metrics with * markers
        """
        clusters = dict()

        for index, cluster in enumerate(labels):
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(index)

        result = []
        for index, cluster in clusters.items():
            result.append(cluster)

        return result
    
    
    def ch_index(self, X, labels):
        '''
        Calinski-Harabasz score is calculated using Scikit Learn's implementation.
        
        The Calinski-Harabasz index (CH) evaluates the cluster validity based on the average between- and within- cluster sum of squares. 
        It is a ratio of cohesion and separation adjusted by the respective degrees of freedom.
        
        Higher values of CH index indicate better clustering performance.
        
        labels = kmeans_model.labels_ format
        '''
        return metrics.calinski_harabasz_score(X, labels)
    
    
    def dunn_index(self, clusters, dist, objects, min_dist=1000000, max_diameter=0):
        """
        The Dunn Index (DI) (introduced by J. C. Dunn in 1974), a metric for evaluating clustering algorithms, is an internal scoring scheme, 
        in which the result is based on the aggregated data itself. Like all other indices of this type, 
        the aim of this Dunn index is to identify sets of compact clusters, with low variance between cluster members and well separated, 
        in which the means of the different clusters are sufficiently far from inside the variance of the cluster.

        The higher the value of the Dunn index, the better the aggregation. 
        The number of clusters which maximizes the Dunn index is considered to be the optimal number of clusters k. 
        It also has some drawbacks. As the number of clusters and the dimensionality of the data increase, so does the cost of computation.
        """

        for i in range(len(clusters) - 1):
            if len(clusters[1]) == 1:
                continue
            max_diameter = max(max_diameter, max_dist_between_2_clusters(clusters[i], clusters[i], dist, objects))
            
            for j in range(i + 1, len(clusters)):
                if len(clusters[j]) == 1:
                    continue
                min_dist = min(min_dist, min_dist_between_2_clusters(clusters[i], clusters[j], dist, objects))

        return min_dist / (max_diameter + eps)
    
    
    def silhouette_index(self, clusters, dist, objects):
        
        objects_vals = [None] * len(objects)
        for my_index, my_cluster in enumerate(clusters):
            for i in my_cluster:
                objects_vals[i] = [None, np.inf]

                for not_my_index, not_my_cluster in enumerate(clusters):
                    dist_sum = np.sum([dist[obj_index, i] for obj_index in not_my_cluster])

                    if not_my_index == my_index:
                        objects_vals[i][0] = dist_sum / (len(my_cluster) - 1)
                    else:
                        dist_sum /= len(not_my_cluster)
                        objects_vals[i][1] = min(dist_sum, objects_vals[i][1])

                objects_vals[i] = (objects_vals[i][1] - objects_vals[i][0]) / max(objects_vals[i][1], objects_vals[i][0])

        return sum(objects_vals) / len(objects)
    
    
    def silhouette_sklearn_index(self, vectors, labels, metric='euclidean'):
        """
        This silhouette score is calculated using sklearn.metrics
        """
        return metrics.silhouette_score(vectors, labels, metric)
    
    
    def db_sklearn_index(self, X, labels):
        """
        Computes the Davies-Bouldin score. The score is calculated using sklearn.metrics

        The score is defined as the average similarity measure of each cluster with its most similar cluster, 
        where similarity is the ratio of within-cluster distances to between-cluster distances. 
        Thus, clusters which are farther apart and less dispersed will result in a better score.

        The minimum score is zero, with lower values indicating better clustering.
        """
        return metrics.davies_bouldin_score(X, labels)
    
    
    def db_find_max_j(self, clusters, centroids, i):
        """
        Support function for computting the Davies-Bouldin score in db_index function.
        """
        max_val = 0
        max_j = 0
        for j in range(len(clusters)):
            if j == i:
                continue
            cluster_i_stat = within_cluster_dist_sum(clusters[i], centroids[i], i) / clusters[i].shape[0]
            cluster_j_stat = within_cluster_dist_sum(clusters[j], centroids[j], j) / clusters[j].shape[0]
            val = (cluster_i_stat + cluster_j_stat) / (((centroids[i] - centroids[j]) ** 2).sum() ** .5)
            if val > max_val:
                max_val = val
                max_j = j
        return max_val

    
    def db_index(self, data, clusters, centroids):
        """
        Computes the Davies-Bouldin score. 
        The score is defined as the average similarity measure of each cluster with its most similar cluster, 
        where similarity is the ratio of within-cluster distances to between-cluster distances. 
        Thus, clusters which are farther apart and less dispersed will result in a better score.

        The minimum score is zero, with lower values indicating better clustering.
        """
        result = 0
        for i in range(len(clusters)):
            result += db_find_max_j(clusters, centroids, i)
            
        return result / len(clusters)
    
    
    def xb_index(self, data, clusters, centroids):
        """
        Xie-Beni index first calculates SSE or cohesion by taking the sum of the squared distances from each data point to its centroid. 
        We denote this by A. Then, it finds the minimum pairwise squared distances between cluster centroids. 
        We denote this by B. We denote the number of data points as n. Xie-Beni index is calculated as A / (n*B).

        The Xie-Beni index defines the inter-cluster separation as the minimum square distance between cluster centers, 
        and the intra-cluster compactness as the mean square distance between each data object and its cluster center. 
        The optimal cluster number is reached when the minimum of Xie-Beni index is found.
        """
        sse = self.sse_2(clusters, centroids)
        min_dist = ((centroids[0] - centroids[1]) ** 2).sum()

        for centroid_i, centroid_j in list(product(centroids, centroids)):
            if (centroid_i - centroid_j).sum() == 0:
                continue
            dist = ((centroid_i - centroid_j) ** 2).sum()
            if dist < min_dist:
                min_dist = dist

        return sse / (data.shape[0] * min_dist)
    
    
    def sd_sdbw_index(X, labels, k=1.0, centers_id=None,  alg_noise='bind',centr='mean', nearest_centr=True, metric='euclidean'):
        """
        The score is calculated using s_dbw.SD
        SD = K*SCATT + DISTANCE
        where distance - distances between cluster centers, k - weighting coefficient equal to distance(Cmax).
        Scatt - means average scattering for clusters
        
        Lower value -> better clustering.
        """
        return SD(X, labels, k=1.0, centers_id=None,  alg_noise='bind',centr='mean', nearest_centr=True, metric='euclidean')
    
    
    def sd_index(clusters, means, dist, objects, k=1):
        """
        SD = K*SCATT + DISTANCE
        where distance - distances between cluster centers, k - weighting coefficient equal to distance(Cmax).
        Scatt - means average scattering for clusters
        
        Lower value -> better clustering.
        """
        stds = np.empty(shape = (len(clusters), objects.shape[1]))
        
        for i in range(len(clusters)):
            stds[i] = std_of_set(objects[clusters[i]], means[i])
            
        std_of_all = std_of_set(objects, np.mean(objects, axis = 0))

        scatt = np.mean(np.linalg.norm(stds, axis = 1))
        scatt /= np.linalg.norm(std_of_all)

        mmax = 0
        mmin = np.inf
        dists = 0
        for index, center in enumerate(means):
            my_dist = dist_point(dist, center)
            diff = np.apply_along_axis(my_dist, axis = -1, arr = means)
            dists += np.power(np.sum(diff), -1)
            mmax = max(mmax, np.max(diff))
            buf = np.delete(diff, index)
            if len(buf) == 0:
                buf = [0]
            mmin = min(mmin, np.min(buf))
            
        dists /= (mmin + eps)
        dists *= mmax

        return k * scatt + dists
    
    
    # vnnd_index(make_clusters(clusterer7.labels_), distances_our, vectors_w2v_sent)
    def vnnd_index(self, clusters, dist):
        """
        Can be time consuming

        This validity index measures the homogeneity of the clusters. Lower index value means more homogenous clustering. 
        This validity index does not use global representative points thus it can measure arbitrary shaped clusters, as well. 
        This validity index can evaluate results of any clustering algorithm but in some cases the calculation of the index can be time consuming. 
        But in case of density based algorithms this validity index can be calculated during the clustering process.
        Say hi to gen_min_span_tree=True
        """
        if len(clusters) <= 1:
            return 100000

        vnnd = 0

        for cluster in clusters:
            means = np.empty(len(cluster))
            for i, x in enumerate(cluster):
                buf = np.delete([dist[obj_index, i] for obj_index in cluster], i)
                if len(buf) == 0:
                    buf = [0]
                means[i] = np.min(buf)
            vnnd += np.power(np.std(means, ddof = 1), 2)

        return vnnd
    
   

    #CVNN 
    
    def _sep(self, labels, k, dk, dist):
        """
        CVNN supporting function
        """
        clusters = sorted(set(labels))
        max_sep = None
        for cluster in clusters:
            cluster_data = dist[labels == cluster]
            cluster_data = cluster_data[:, labels != cluster]
            cluster_dk = dk[labels == cluster]
            sep = len(cluster_data[cluster_data <= np.c_[cluster_dk]]) / (k * cluster_data.shape[0])
            if max_sep is None or max_sep < sep:
                max_sep = sep
        return max_sep


    def _com(self, labels, dist):
        """
        CVNN supporting function
        """
        clusters = sorted(set(labels))
        com = 0
        max_com = 0
        for cluster in clusters:
            cluster_data = dist[labels == cluster]
            cluster_data = cluster_data[:, labels == cluster]
            n_i = cluster_data.shape[0]
            #        print(n_i, cluster_data.sum())
            if n_i > 1:
                cur_sum = 2 * cluster_data.sum() / (n_i * (n_i - 1))
                com += cur_sum
                if max_com < cur_sum:
                    max_com = cur_sum
        return com, max_com


    def cvnn(self, labels, k, dk, dist):
        """
        Clustering Validation index based on Nearest Neighbors
        """
        com, max_com = self._com(labels, dist)
        return self._sep(labels, k, dk, dist) + com / max_com