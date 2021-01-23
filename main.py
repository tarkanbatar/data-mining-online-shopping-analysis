# Preprocess

import kmeans1d
import numpy as np

Data_A = Data_F_M_P.Administrative
Data_AD = Data_F_M_P.Administrative_Duration
Data_I = Data_F_M_P.Informational
Data_ID = Data_F_M_P.Informational_Duration
Data_PR = Data_F_M_P.ProductRelated
Data_PRD = Data_F_M_P.ProductRelated_Duration
Data_BR = Data_F_M_P.BounceRates
Data_ER = Data_F_M_P.ExitRates
Data_PV = Data_F_M_P.PageValues
Data_SD = Data_F_M_P.SpecialDay
Data_OS = Data_F_M_P.OperatingSystems
Data_B = Data_F_M_P.Browser
Data_R = Data_F_M_P.Region
Data_TT = Data_F_M_P.TrafficType
Data_W = Data_F_M_P.Weekend
Data_Label = Data_F_M_P.Revenue

Data_A_Arr = pd.Series(Data_A).array
Data_AD_Arr = pd.Series(Data_AD).array
Data_I_Arr = pd.Series(Data_I).array
Data_ID_Arr = pd.Series(Data_ID).array
Data_PR_Arr = pd.Series(Data_PR).array
Data_PRD_Arr = pd.Series(Data_PRD).array
Data_BR_Arr = pd.Series(Data_BR).array
Data_ER_Arr = pd.Series(Data_ER).array
Data_PV_Arr = pd.Series(Data_PV).array
Data_SD_Arr = pd.Series(Data_SD).array
Data_OS_Arr = pd.Series(Data_OS).array
Data_B_Arr = pd.Series(Data_B).array
Data_R_Arr = pd.Series(Data_R).array
Data_TT_Arr = pd.Series(Data_TT).array
Data_W_Arr = pd.Series(Data_W).array
Data_Label_Arr = pd.Series(Data_Label).array

clusters, centroids_A = kmeans1d.cluster(Data_A_Arr, 5)
clusters, centroids_AD = kmeans1d.cluster(Data_AD_Arr, 5)
clusters, centroids_I = kmeans1d.cluster(Data_I_Arr, 5)
clusters, centroids_ID = kmeans1d.cluster(Data_ID_Arr, 5)
clusters, centroids_PR = kmeans1d.cluster(Data_PR_Arr, 5)
clusters, centroids_PRD = kmeans1d.cluster(Data_PRD_Arr, 5)
clusters, centroids_BR = kmeans1d.cluster(Data_BR_Arr, 5)
clusters, centroids_ER = kmeans1d.cluster(Data_ER_Arr, 5)
clusters, centroids_PV = kmeans1d.cluster(Data_PV_Arr, 5)
clusters, centroids_SD = kmeans1d.cluster(Data_SD_Arr, 5)


def find_Centroid(arr):
  Out = np.array([0.0, 0.0, 0.0, 0.0])
  Out[0] = (arr[0] + arr[1]) / 2
  Out[1] = (arr[1] + arr[2]) / 2
  Out[2] = (arr[2] + arr[3]) / 2
  Out[3] = (arr[3] + arr[4]) / 2
  return Out

Cat_A = find_Centroid(centroids_A)
Cat_AD = find_Centroid(centroids_AD)
Cat_I = find_Centroid(centroids_I)
Cat_ID = find_Centroid(centroids_ID)
Cat_PR = find_Centroid(centroids_PR)
Cat_PRD = find_Centroid(centroids_PRD)
Cat_BR = find_Centroid(centroids_BR)
Cat_ER = find_Centroid(centroids_ER)
Cat_PV = find_Centroid(centroids_PV)
Cat_SD = find_Centroid(centroids_SD)


def find_Category(data, cat, ab):
  cat_data = []
  for every_data in data:
    if( float(every_data) <= cat[0]):
      string = ab+"_1"
      cat_data.append(string)
    elif( float(every_data) <= cat[1]):
      string = ab+"_2"
      cat_data.append(string)
    elif( float(every_data) <= cat[2]):
      string = ab+"_3"
      cat_data.append(string)
    elif( float(every_data) <= cat[3]):
      string = ab+"_4"
      cat_data.append(string)
    else:
      string = ab+"_5"
      cat_data.append(string)
  return cat_data

Data_A_Cat = find_Category(Data_A, Cat_A, "A")
Data_AD_Cat = find_Category(Data_AD, Cat_AD, "AD")
Data_I_Cat = find_Category(Data_I, Cat_I, "I")
Data_ID_Cat = find_Category(Data_ID, Cat_ID, "ID")
Data_PR_Cat = find_Category(Data_PR, Cat_PR, "PR")
Data_PRD_Cat = find_Category(Data_PRD, Cat_PRD, "PRD")
Data_BR_Cat = find_Category(Data_BR, Cat_BR, "BR")
Data_ER_Cat = find_Category(Data_ER, Cat_ER, "ER")
Data_PV_Cat = find_Category(Data_PV, Cat_PV, "PV")
Data_SD_Cat = find_Category(Data_SD, Cat_SD, "SD")


def change_Data(arr, ab):
  data = []
  for i in arr:
    string = str(ab) + "_" + str(i)
    data.append(string)
  return data

df_Data_A_Cat = pd.get_dummies(Data_A_Cat)
df_Data_AD_Cat = pd.get_dummies(Data_AD_Cat)
df_Data_I_Cat = pd.get_dummies(Data_I_Cat)
df_Data_ID_Cat = pd.get_dummies(Data_ID_Cat)
df_Data_PR_Cat = pd.get_dummies(Data_PR_Cat)
df_Data_PRD_Cat = pd.get_dummies(Data_PRD_Cat)
df_Data_BR_Cat = pd.get_dummies(Data_BR_Cat)
df_Data_ER_Cat = pd.get_dummies(Data_ER_Cat)
df_Data_PV_Cat = pd.get_dummies(Data_PV_Cat)
df_Data_SD_Cat = pd.get_dummies(Data_SD_Cat)
df_Data_Month = pd.get_dummies(Data_F_M_P.Month)

Cat_OS = change_Data(Data_OS_Arr, "OS")
df_Data_OS_Cat = pd.get_dummies(Cat_OS)

Cat_B = change_Data(Data_B_Arr, "B")
df_Data_B_Cat = pd.get_dummies(Cat_B)

Cat_R = change_Data(Data_R_Arr, "R")
df_Data_R_Cat = pd.get_dummies(Cat_R)

Cat_TT = change_Data(Data_TT_Arr, "TT")
df_Data_TT_Cat = pd.get_dummies(Cat_TT)

df_Data_VT_Cat = pd.get_dummies(Data_F_M_P.VisitorType)

Cat_W = change_Data(Data_W_Arr, "W")
df_Data_W_Cat = pd.get_dummies(Cat_W)

Cat_Label = change_Data(Data_Label_Arr, "Revenue")
df_Data_Label_Cat = pd.get_dummies(Cat_Label)




print("Since we have change numerical data to categorical data, the new category is looking like this: \n")

print("For Administrative : A_1 < "+ str(Cat_A[0]) + " < A_2 < "+ str(Cat_A[1]) + " < A_3 < "+ str(Cat_A[2]) + " < A_4 < "+ str(Cat_A[3]) + " < A_5")
print("For Administrative_Duration : AD_1 < "+ str(Cat_AD[0]) + " < AD_2 < "+ str(Cat_AD[1]) + " < AD_3 < "+ str(Cat_AD[2]) + " < AD_4 < "+ str(Cat_AD[3]) + " < AD_5")
print("For Informational : I_1 < "+ str(Cat_I[0]) + " < I_2 < "+ str(Cat_I[1]) + " < I_3 < "+ str(Cat_I[2]) + " < I_4 < "+ str(Cat_I[3]) + " < I_5")
print("For Informational_Duration : ID_1 < "+ str(Cat_ID[0]) + " < ID_2 < "+ str(Cat_ID[1]) + " < ID_3 < "+ str(Cat_ID[2]) + " < ID_4 < "+ str(Cat_ID[3]) + " < ID_5")
print("For ProductRelated : PR_1 < "+ str(Cat_PR[0]) + " < PR_2 < "+ str(Cat_PR[1]) + " < PR_3 < "+ str(Cat_PR[2]) + " < PR_4 < "+ str(Cat_PR[3]) + " < PR_5")
print("For ProductRelated_Duration : PRD_1 < "+ str(Cat_PRD[0]) + " < PRD_2 < "+ str(Cat_PRD[1]) + " < PRD_3 < "+ str(Cat_PRD[2]) + " < PRD_4 < "+ str(Cat_PRD[3]) + " < PRD_5")
print("For BounceRates : BR_1 < "+ str(Cat_BR[0]) + " < BR_2 < "+ str(Cat_BR[1]) + " < BR_3 < "+ str(Cat_BR[2]) + " < BR_4 < "+ str(Cat_BR[3]) + " < BR_5")
print("For ExitRates : ER_1 < "+ str(Cat_ER[0]) + " < ER_2 < "+ str(Cat_ER[1]) + " < ER_3 < "+ str(Cat_ER[2]) + " < ER_4 < "+ str(Cat_ER[3]) + " < ER_5")
print("For PageValues : PV_1 < "+ str(Cat_PV[0]) + " < PV_2 < "+ str(Cat_PV[1]) + " < PV_3 < "+ str(Cat_PV[2]) + " < PV_4 < "+ str(Cat_PV[3]) + " < PV_5")
print("For SpecialDay : SD_1 < "+ str(Cat_SD[0]) + " < SD_2 < "+ str(Cat_SD[1]) + " < SD_3 < "+ str(Cat_SD[2]) + " < SD_4 < "+ str(Cat_SD[3]) + " < SD_5")

DF_F_P_M = pd.concat((df_Data_A_Cat, df_Data_AD_Cat, df_Data_I_Cat, df_Data_ID_Cat, df_Data_PR_Cat, df_Data_PRD_Cat), axis=1)
DF_F_P_M = pd.concat((DF_F_P_M, df_Data_BR_Cat, df_Data_ER_Cat, df_Data_PV_Cat, df_Data_SD_Cat, df_Data_Month, df_Data_B_Cat), axis=1)
DF_F_P_M = pd.concat((DF_F_P_M, df_Data_OS_Cat, df_Data_R_Cat, df_Data_TT_Cat, df_Data_VT_Cat, df_Data_W_Cat, df_Data_Label_Cat), axis=1)
print("Our final Data Frame is looking like: \n")
print(DF_F_P_M)

# APRIORI Frequent Pattern Mining

import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
start_time = time.time()
DF_Apriori = DF_F_P_M

# Not taking frequent_itemsets that have a support less than 0.7
frequent_itemsets = apriori(DF_Apriori, min_support=0.7, use_colnames=True)

print('\nFrequent Itemsets: ')
print(frequent_itemsets)

pd.set_option
#pd.set_option('max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('max_colwidth', 300)

association_r = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print('\nAssociation Rules: ')
print(association_r)

# resetting the values to default
#pd.reset_option("display.max_rows")
pd.reset_option("display.max_colwidth")
pd.reset_option("display.max_columns")

antecendents_arr = association_r.antecedents
consequents_arr = association_r.consequents
support_arr = association_r.support

for i in range(len(support_arr)):
  a_ls = list(antecendents_arr[i])
  c_ls = list(consequents_arr[i])

  revenue_true = "Revenue_True"
  revenue_false = "Revenue_False"
  if revenue_true in str(a_ls):
    message = "If Revenue True then it contains " + str(c_ls) + "  with support " + str(support_arr[i])
    print(message)
  elif revenue_false in str(a_ls):
    message = "If Revenue False then it contains " + str(c_ls) + "  with support " + str(support_arr[i])
    print(message)
print("--- Apriori: %s seconds ---" % (time.time() - start_time))


#FP-GROWTH Frequent Pattern Mining

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

start_time = time.time()

DF_FP_Growth = DF_F_P_M

fp_frequent_itemset = fpgrowth(DF_FP_Growth, min_support=0.7, use_colnames=True)
print("Frequent item set has been listed below...")
print(fp_frequent_itemset)

pd.set_option
#pd.set_option('max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('max_colwidth', 100)

asc_rules = association_rules(fp_frequent_itemset, metric="lift",min_threshold=1)
print('\nAssociation Rules: ')
print(asc_rules)

# resetting the values to default
#pd.reset_option("display.max_rows")
pd.reset_option("display.max_colwidth")
pd.reset_option("display.max_columns")

antecendents_arr = asc_rules.antecedents
consequents_arr = asc_rules.consequents
support_arr = asc_rules.support

for i in range(len(support_arr)):
  a_ls = list(antecendents_arr[i])
  c_ls = list(consequents_arr[i])

  revenue_true = "Revenue_True"
  revenue_false = "Revenue_False"
  if revenue_true in str(a_ls):
    message = "If Revenue True then it contains " + str(c_ls) + "  with support " + str(support_arr[i])
    print(message)
  elif revenue_false in str(a_ls):
    message = "If Revenue False then it contains " + str(c_ls) + "  with support " + str(support_arr[i])
    print(message)
print("--- FP Growth: %s seconds ---" % (time.time() - start_time))


# ECLAT Frequent Pattern Mining

DF_Eclat = DF_F_P_M

start_time = time.time()

frequent_itemsets = apriori(DF_Eclat, min_support=0.7, use_colnames=True)
#bulid association rules using support metric
rules = association_rules(frequent_itemsets, metric="support", support_only=True, min_threshold=0.1 )

#use only support metric in Eclat algo using apriori

rules = rules[['antecedents', 'consequents', 'support']]
ruless = rules.sort_values('support', ascending=False)

print(ruless)

antecendents_arr = rules.antecedents
consequents_arr = rules.consequents
support_arr = rules.support

for i in range(len(support_arr)):
  a_ls = list(antecendents_arr[i])
  c_ls = list(consequents_arr[i])

  revenue_true = "Revenue_True"
  revenue_false = "Revenue_False"
  if revenue_true in str(a_ls):
    message = "If Revenue True then it contains " + str(c_ls) + "  with support " + str(support_arr[i])
    print(message)
  elif revenue_false in str(a_ls):
    message = "If Revenue False then it contains " + str(c_ls) + "  with support " + str(support_arr[i])
    print(message)
print("--- ECLAT: %s seconds ---" % (time.time() - start_time))

# Data Preprocess For Clustering

col_names = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType','Weekend', 'Revenue']
DF_C = pd.read_csv("online_shoppers_intention.csv", header=0, names=col_names)

DF_C = DF_C.replace('New_Visitor', 0)
DF_C = DF_C.replace('Returning_Visitor', 1)
DF_C = DF_C.replace('Other', 2)
DF_C = DF_C.replace('Feb', 2)
DF_C = DF_C.replace('Mar', 3)
DF_C = DF_C.replace('May', 5)
DF_C = DF_C.replace('June', 6)
DF_C = DF_C.replace('Jul', 7)
DF_C = DF_C.replace('Aug', 8)
DF_C = DF_C.replace('Sep', 9)
DF_C = DF_C.replace('Oct', 10)
DF_C = DF_C.replace('Nov', 11)
DF_C = DF_C.replace('Dec', 12)
DF_C = DF_C.replace(False, 0)
DF_C = DF_C.replace(True, 1)

print("After some preprocess for clustering, information of our dataset can be seen below...")
print(DF_C.info())

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

True_Label = DF_C.Revenue
DF_C_temp = DF_C
DF_C_temp = DF_C_temp.drop('Revenue', axis=1)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(DF_C_temp)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.scatter(pca_data[True_Label==0, 0], pca_data[True_Label==0, 1], s=0.5, c='blue', label ='Revenue_False')
plt.scatter(pca_data[True_Label==1, 0], pca_data[True_Label==1, 1], s=0.5, c='red', label ='Revenue_True')
plt.title('True Labeled Cluster Plot')
plt.xlabel('Information_1')
plt.ylabel('Information_2')
plt.legend();
plt.show()

# K-Means Clustering Analysis

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

DF_K_Means = DF_C

start_time = time.time()

Y_K_Means = np.array(DF_K_Means['Revenue'])
DF_K_Means = DF_K_Means.drop('Revenue', axis=1)
print('Normalizing data and separating label to test the clustering application')
X_K_Means = preprocessing.normalize(DF_K_Means)

scaler = MinMaxScaler()
X_Scaled_K_Means = scaler.fit_transform(X_K_Means) # Increase of accuracy with 1% (.78)

print('Since we have two values for labels we cluster data into two clusters\n')

kmeans =  KMeans(n_clusters=2) # .77 Normally
kmeans.fit(X_Scaled_K_Means)

correct = 0 # To keep the count of the correctly clustered entities

for i in range(len(X_K_Means)):
    predict_me = np.array(X_K_Means[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    # print(int(prediction[0]), int(Y_K_Means[i]))
    if int(prediction[0]) == int(Y_K_Means[i]):
        correct += 1

print('Accuracy: ', correct/len(X_K_Means)) # Number of correctly clustered entities according to the lables
print('Inertia: ', kmeans.inertia_) # Sum of squared distances of samples to their closest cluster center.
print('Number of iterations: ', kmeans.n_iter_) # Number of iterations run.
print('Centers of the two clusters: ')
for i in kmeans.cluster_centers_:
  print(i)

Label_K_Means = kmeans.labels_
print("\nHomogeneity: %0.3f" %metrics.homogeneity_score(True_Label, Label_K_Means))
print("Completeness: %0.3f" %metrics.completeness_score(True_Label, Label_K_Means))
print("V-measure: %0.3f" %metrics.v_measure_score(True_Label, Label_K_Means))
print("Adjusted Rand Index: %0.3f" %metrics.adjusted_rand_score(True_Label, Label_K_Means))
print("Adjusted Mutual Information: %0.3f" %metrics.adjusted_mutual_info_score(True_Label, Label_K_Means))
print("Silhouette Coefficient: %0.3f" %metrics.silhouette_score(X_Scaled_K_Means, Label_K_Means))

plt.scatter(pca_data[Label_K_Means==0, 0], pca_data[Label_K_Means==0, 1], s=0.5, c='blue', label ='Revenue_False')
plt.scatter(pca_data[Label_K_Means==1, 0], pca_data[Label_K_Means==1, 1], s=0.5, c='red', label ='Revenue_True')
plt.title('K_Means Cluster Plot')
plt.xlabel('')
plt.ylabel('')
plt.legend();
plt.show()
print("--- K-Means: %s seconds ---" % (time.time() - start_time))


# AGNES Clustering Analysis

DF_AGNES = DF_C

start_time = time.time()

DF_AGNES = DF_AGNES.drop('Revenue', axis=1)
X_AGNES = preprocessing.normalize(DF_AGNES)

Cluster_Agnes = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
predict_label = Cluster_Agnes.fit_predict(X_AGNES)
Label_AGNES = Cluster_Agnes.labels_

print("Homogeneity: %0.3f" %metrics.homogeneity_score(True_Label, Label_AGNES))
print("Completeness: %0.3f" %metrics.completeness_score(True_Label, Label_AGNES))
print("V-measure: %0.3f" %metrics.v_measure_score(True_Label, Label_AGNES))
print("Adjusted Rand Index: %0.3f" %metrics.adjusted_rand_score(True_Label, Label_AGNES))
print("Adjusted Mutual Information: %0.3f" %metrics.adjusted_mutual_info_score(True_Label, Label_AGNES))
print("Silhouette Coefficient: %0.3f" %metrics.silhouette_score(X_AGNES, Label_AGNES))


plt.scatter(pca_data[Label_AGNES==0, 0], pca_data[Label_AGNES==0, 1], s=0.5, c='blue', label ='Revenue_False')
plt.scatter(pca_data[Label_AGNES==1, 0], pca_data[Label_AGNES==1, 1], s=0.5, c='red', label ='Revenue_True')
plt.title('Agnes Cluster Plot')
plt.xlabel('')
plt.ylabel('')
plt.legend();
plt.show()
print("--- AGNES: %s seconds ---" % (time.time() - start_time))

# DBSCAN Clustering Analysis

DF_DBSCAN = DF_C

start_time = time.time()


#y = np.array(DF_AGNES['Revenue'])
DF_DBSCAN = DF_DBSCAN.drop('Revenue', axis=1)
X_DBSCAN = preprocessing.normalize(DF_DBSCAN)

Cluster_DBSCAN = DBSCAN(eps=0.25, min_samples=20).fit(X_DBSCAN)
Label_DBSCAN = Cluster_DBSCAN.labels_

core_samples_mask = np.zeros_like(Label_DBSCAN, dtype=bool)
core_samples_mask[Cluster_DBSCAN.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(Label_DBSCAN)) - (1 if -1 in Label_DBSCAN else 0)
n_noise_ = list(Label_DBSCAN).count(-1)


print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(True_Label, Label_DBSCAN))
print("Completeness: %0.3f" % metrics.completeness_score(True_Label, Label_DBSCAN))
print("V-measure: %0.3f" % metrics.v_measure_score(True_Label, Label_DBSCAN))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(True_Label, Label_DBSCAN))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(True_Label, Label_DBSCAN))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_DBSCAN, Label_DBSCAN))

