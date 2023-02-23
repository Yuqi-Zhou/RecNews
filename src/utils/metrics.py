from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}

def calculate_single_user_metric(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    mrr = mrr_score(y_true, y_score)
    ndcg5 = ndcg_score(y_true, y_score, 5)
    ndcg10 = ndcg_score(y_true, y_score, 10)
    return [auc, mrr, ndcg5, ndcg10]


def sigmoid(x):
    return 1/(1+np.exp(np.clip(-x, a_min=-1e50, a_max=1e20)))

def cal_auc(label, pos_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, pos_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def stable_log1pex(x):
    res = -np.minimum(x, 0) + np.log(1+np.exp(-np.abs(x)))
    return res

def cal_llloss_with_logits(label, logits):
    ll = -np.mean(label*(-stable_log1pex(logits)) + (1-label)*(-logits - stable_log1pex(logits)))
    return ll

def prob_clip(x):
    return np.clip(x, a_min=1e-20, a_max=1)

def cal_llloss_with_neg_log_prob(label, neg_log_prob):
    ll = -np.mean((1-label)*neg_log_prob + label*(np.log(prob_clip(1 - prob_clip(np.exp(neg_log_prob))))))
    return ll

def cal_llloss_with_prob(label, prob):
    ll = -np.mean(label*np.log(prob_clip(prob)) + (1-label)*(np.log(prob_clip(1-prob))))
    return ll

def cal_prauc(label, pos_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(label, pos_prob)
    area = metrics.auc(recall, precision)
    return area

def cal_acc(label, prob):
    label = np.reshape(label, (-1,))
    prob = np.reshape(label, (-1,))
    prob_acc = np.mean(label*prob)
    return prob_acc

def stable_softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

# def gauc_score(y_true_list, y_pred_list, weights=0):
#     """compute GAUC
#     Args: 
#         y_true (list): list[narray dim(N,), ], all true labels of the data
#         y_pred (array):list[narray dim(N,), ], the predicted score
#         weight (dict): {userid: weight_value}, it contains weights for each group. 
#                     if it is 0, the weight is equal to the number
#                     of times the user is recommended
#                     if it is 1, the weight is equal to the number
#                     of times the user clicked
#     Return:
#         score: float, GAUC
#     """
#     for group_y_true, group_y_pred in zip(y_true_list, y_pred_list):
#         assert len(group_y_true) == len(group_y_pred)
    
#     # check positive and negative samples
#     user_len_list = np.array([len(group) for group in y_true_list])
#     pos_len_list = np.array([np.sum(group) for group in y_true_list])
#     neg_len_list = user_len_list - pos_len_list
#     any_without_pos = np.any(pos_len_list == 0)
#     any_without_neg = np.any(neg_len_list == 0)
#     non_zero_idx = np.full(len(user_len_list), True, dtype=np.bool)

#     if any_without_pos:
#         non_zero_idx *= (pos_len_list != 0)
#     if any_without_neg:
#         non_zero_idx *= (neg_len_list != 0)
#     if any_without_pos or any_without_neg:
#         item_list = user_len_list, neg_len_list, pos_len_list
#         user_len_list, neg_len_list, pos_len_list = map(lambda x: x[non_zero_idx], item_list)
    
#     score = 0
#     num = 0
#     for i, (group_y_true, group_y_pred) in enumerate(zip(y_true_list, y_pred_list)):
#         if non_zero_idx[i] == False:
#             continue
#         auc = cal_auc(group_y_true, group_y_pred)
#         if weights == 0:
#             user_weight = len(group_y_true)
#         elif weights == 1:
#             user_weight = np.sum(group_y_true)
#         else:
#             raise NotImplemented()
#         auc *= user_weight
#         num += user_weight
#         score += auc
#     if num == 0:
#         return 0
#     else:
#         return score / num

if __name__ == "__main__":
    pass
    
    