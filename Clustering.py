
#   Author: Sean Saathoff
#   Goal: Using data about traffic patterns of various restaurants, use k-means clustering to group the restaurants into k different clusters.  Then the Purity and Normalized Mutual Information (NMI) scores are calculated for the clusters.

import random
import numpy as np

# The class to implement Clustering algorithm 
class Clustering(object):
    def __init__(self):
        self._daily_patterns = {}
        self._weekly_patterns = {}
        self._combined_patterns = {}
        self._pattern = {}
    
    #   This function takes the filename of the traffic patterns, and  
    #   populates the dictionaries created in the class
    def create_dicts(self, traffic_pat_filename):
        f = open(traffic_pat_filename, 'rU')
        for line in f:
            line = line.replace('\n', '')
            line = line.split('\t')
            key = line.pop(0)
            #if it is the daily patter, add to daily dict and combined dict
            if len(line) == 24:
                self._daily_patterns[key] = [float(s) for s in line]
                self._combined_patterns[key] = [float(s) for s in line]
            #if weekly pattern, add to weekly dict and combined dict
            elif len(line) == 70:
                self._weekly_patterns[key] = [float(s) for s in line]
                self._combined_patterns[key].extend([float(s) for s in line])
            
    # purpose:
    #   This function performs cluster algorithm for daily traffic patterns
    #   Takes as a parameter k, which indicates number of clusters
    #   Pattern determines which dictionary to use, 0 = daily, 1 = weekly, 2 = combined
    def cal_daily_cluster(self, k, pattern):
        
        if pattern == 0:
            self._pattern = self._daily_patterns
            print len(self._pattern['KFC'])
        elif pattern == 1:
            self._pattern = self._weekly_patterns
            print len(self._pattern['KFC'])
        elif pattern == 2:
            self._pattern = self._combined_patterns
            print len(self._pattern['KFC'])
        #get k initial random centroids
        centroids =  [self._pattern[key] for key in random.sample(self._pattern.keys(), k)]
        prev_clusters = [[]for i in range(k)]
        
        #assign each location to a cluster by computer temporal correlation
        #set max repeats to 100
        count = 0
        for x in range(100):
            count += 1
            clusters = [[] for i in range(k)]
            for loc in self._pattern.keys():
                data = self._pattern[loc]
                highest_corr = (0,0)
                
                #compute correlation with each centriod and highest highest correlation
                for i in range(k):
                    corr = self.calc_temp_corr(data, centroids[i])
                    if corr >= highest_corr[1]: highest_corr = (i,corr)
                clusters[highest_corr[0]].append(loc)
        
            if clusters == prev_clusters: break
            else: prev_clusters = clusters
    
            #calculate new centroids based on center of gravity/mean
            centroids = self.cal_centroid(clusters, k)
        return clusters


    # purpose:
    #   This function computes the Temporal Correlation between two locations
    #
    #   returns: a float, higher int means higher correlation
    #   
    # parameters:
    #   loc_data - list of data for location
    #   centroid_data - list of data for centroid
    def calc_temp_corr(self, loc_data, centroid_data):
        
        #get average and stand. dev. for location frequency pattern
        local_av = np.average(loc_data)
        local_std_dev = np.std(loc_data)
    
        #get average and stand. dev. for centroid data
        centroid_av = np.average(centroid_data)
        centroid_std_dev = np.std(centroid_data)

        n = len(loc_data)

        summation = 0.0;
        
        #calculate temporal correlation between location and centroid
        for p, q in map(None, loc_data, centroid_data):
            first_product = (p - local_av)/local_std_dev
            second_product = (q - centroid_av)/centroid_std_dev
            product = first_product * second_product
            summation += product

        t_corr = (1.0/n)*summation

        return t_corr
                                                    
                                                    
    # purpose:
    #   This function computes the new centroids based on current clusters
    #
    #   returns: a new list of centroids
    #   
    # parameters:
    #   clusters - list of clusters
    #   k - number of clusters to compute
    def cal_centroid(self, clusters, k):
        
        #initialize new centroid lists to be all 0
        length = len(self._pattern["KFC"])
        new_centroids = [[0]*length for i in range(k)]
        
        #for each centroid, get the group of locations in that cluster
        for i in range(k):
            group = clusters[i]
            num_locs = len(group)
            #sum up the frequency from each location
            for location in group:
                temp_list = [sum(pair) for pair in zip(new_centroids[i], self._pattern[location])]
                new_centroids[i] = temp_list
            
            #multiply through by 1/(num of locations in cluster)
            new_centroids[i][:] = [x/float(num_locs) for x in new_centroids[i]]
                                  
        return new_centroids

    # purpose:
    #   This function computes purity of our created clusters
    #   based on the provided golden standard clusters
    #
    #   returns: a float between 0 and 1 which is the purity score
    #   
    # parameters:
    #   clusters - list of clusters
    #   goldstandard_filename - filename that contains gold standard clusters
    def cal_purity(self, clusters, goldstandard_filename):
        
        purity_score = 0.0
        total_locs = len(self._pattern)
        truth_clusters = []  
        
        #put ground truth lists into a list
        f = open(goldstandard_filename, 'rU')
        for line in f:
            line = line.replace('\n', '')
            line = line.split('\t')
            truth_clusters.append(line)
       
        matches = []
        for cluster in clusters:
            temp_list = []
            for truth_clust in truth_clusters:
                num_matches = len(list(set(cluster)& set(truth_clust)))
                temp_list.append(num_matches)
            matches.append(temp_list)
   
        summation = 0.0
        for matches_list in matches:
            summation += max(matches_list)
        
        purity_score = (1.0/total_locs)*summation
        
        return purity_score

    # purpose:
    #   This function computes NMI of our created clusters
    #   based on the provided golden standard clusters
    #
    #   returns: a float between 0 and 1 which is the NMI score
    #   
    # parameters:
    #   clusters - list of clusters
    #   goldstandard_filename - filename that contains gold standard clusters
    def cal_nmi(self, clusters, goldstandard_filename):
        
        nmi_score = 0.0
        total_locs = len(self._pattern)
        truth_clusters = []  
        
        #put ground truth lists into a list
        f = open(goldstandard_filename, 'rU')
        for line in f:
            line = line.replace('\n', '')
            line = line.split('\t')
            truth_clusters.append(line)
        
        mi_score = 0.0
        #first calculate mutual information
        for cluster in clusters:
            for truth_clust in truth_clusters:
                intersect_len = len(list(set(cluster) & set(truth_clust)))
                clust_len = len(cluster)
                truth_len = len(truth_clust)
                coef = float(intersect_len)/float(total_locs)
                if coef != 0.0:
                    mi_score += coef*np.log((float(total_locs)*float(intersect_len))/(float(clust_len)*float(truth_len)))
        
        #now calculate entropy (H) for clusters and ground truth clusters
        h_cluster = 0.0
        h_truth = 0.0
        for cluster in clusters:
            coef = float(len(cluster))/float(total_locs)
            if coef != 0.0:
                h_cluster += coef*np.log(coef)
        h_cluster = h_cluster*-1.0
        for truth_clust in truth_clusters:
            coef = float(len(truth_clust))/float(total_locs)
            if coef != 0.0:
                h_truth += coef*np.log(coef)
        h_truth = h_truth*-1.0
            
        nmi_score = mi_score/((h_cluster+h_truth)/2)
                    
        return nmi_score
        
def main(args):
    cluster = Clustering()
    cluster.create_dicts("traffic_patterns.txt")
    f = open("data.txt", "w")
    for pattern in range(0,3):  #patterns: 0 - daily, 1 - weekly, 2 - combined
        for k in range(2,21):   #num of clusters: 2-21
            print "k =", k
            f.write("%d\n" % (k))
            for i in range (0,50):  #repeat 50 times for each setup
                print "i = ", i
                final_clusters = cluster.cal_daily_cluster(k, pattern)
                purity = cluster.cal_purity(final_clusters, "ground_truth_clusters.txt")
                f.write("%f\t" % (purity))
                nmi = cluster.cal_nmi(final_clusters, "ground_truth_clusters.txt")
                f.write("%f\n" % (nmi))

    f.close()
    
# this little helper will call main() if this file is executed from the command
# line but not call main() if this file is included as a module
if __name__ == "__main__":
    import sys
    main(sys.argv)

