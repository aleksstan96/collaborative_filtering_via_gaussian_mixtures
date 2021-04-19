import numpy as np
import kmeans
import common
import naive_em
import em
K_array = [1, 2, 3, 4]
seeds_array = [0, 1, 2, 3, 4]
X = np.loadtxt("toy_data.txt")
costs = np.array([len(K_array), len(seeds_array)])
k_dict = dict()
bic_dict = dict()

# TODO: Your code here
# for k in K_array:
#     for seed in seeds_array:
#         gaussianMixture_init, arr_init = common.init(X, k, seed)
#         gaussianMixture, arr, likelihood = naive_em.run(X, gaussianMixture_init, arr_init)
#         # gaussianMixture, arr, cost = kmeans.run(X, gaussianMixture_init, arr_init)
#         k_dict.update({(k, seed): likelihood})
#         bic = common.bic(X, gaussianMixture, likelihood)
#         bic_dict.update({(k, seed): bic})
#         title = 'EM_naive' + str(k) + str(seed) + ': k= ' + str(k) + ', seed= ' + str(seed)
#         img_name = 'EM' + str(k) + str(seed) + '.png'
        #common.plot(X, gaussianMixture, arr, title, img_name)


# for k in K_array:
#     for seed in seeds_array:
#         gaussianMixture_init, arr_init = common.init(X, k, seed)
#         gaussianMixture, arr, cost = kmeans.run(X, gaussianMixture_init, arr_init)
#         k_dict.update({(k, seed): cost})
#         title = 'Kmeans' + str(k) + str(seed) + ': k= ' + str(k) + ', seed= ' + str(seed)
#         img_name = 'Kmeans' + str(k) + str(seed) + '.png'
#         common.plot(X, gaussianMixture, arr, title, img_name)


# common.plot(X, gaussianMixture, arr, "plot1")

X_gold = np.loadtxt('netflix_complete.txt')
X_netflix = np.loadtxt("netflix_incomplete.txt")
# K_array_netflix = [1, 12]
# for k in K_array_netflix:
#     for seed in seeds_array:
#         gaussianMixture_init, arr_init = common.init(X_netflix, k, seed)
#         gaussianMixture, arr, likelihood = em.run(X_netflix, gaussianMixture_init, arr_init)
#         # gaussianMixture, arr, cost = kmeans.run(X, gaussianMixture_init, arr_init)
#         k_dict.update({(k, seed): likelihood})
#         bic = common.bic(X_netflix, gaussianMixture, likelihood)
#         bic_dict.update({(k, seed): bic})
#         title = 'EM' + str(k) + str(seed) + ': k= ' + str(k) + ', seed= ' + str(seed)

# print(k_dict)
# print(max(bic_dict, key=bic_dict.get))
# print(bic_dict)

#prediction
gaussianMixture_init_pred, arr_init_pred = common.init(X_netflix, K=12, seed=1) #best result for K=12 was with seed = 1
mixture_pred = em.run(X_netflix, gaussianMixture_init_pred, arr_init_pred)
X_pred = em.fill_matrix(X_netflix, mixture_pred)

rmse = common.rmse(X_gold, X_pred)
print(rmse)
