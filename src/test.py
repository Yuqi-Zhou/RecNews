import os
import importlib
from os import path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from config import model_name
from dataset.Dataset import BaseDataset, NewsDataset, UserDataset, BehaviorsDataset
from utils.metrics import calculate_single_user_metric

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
except AttributeError:
    print(f"{model_name} not included!")
    exit()

class Tester:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = self.config.batch_size
        self.test_dir = "data/small_dev"
        
    def get_news2vector(self):
        # get news to vector dic
        news2vector = {}
        news_dataset = NewsDataset(path.join(self.test_dir, 'news_parsed.tsv'))
        news_dataloader = news_dataset.get_dataloader(batch_size=self.batch_size, \
            shuffle=False, num_workers=self.config.num_workers)
        for batch in tqdm(news_dataloader, desc="Calculating vectors for news"):
            news_ids =  batch["id"] # batch, 1
            if any(id not in news2vector for id in news_ids):
                news_vector = self.model.get_news_vector(batch["news"])
                for id, vector in zip(news_ids, news_vector):
                    if id not in news2vector:
                        news2vector[id] = vector.to('cpu')
        news2vector['PADDED_NEWS'] = torch.zeros(
            list(news2vector.values())[0].size())
        return news2vector
        
    def get_user2vector(self, news2vector):
        # get news to vector dic
        user2vector = {}
        user_dataset = UserDataset(path.join(self.test_dir, 'behaviors.tsv'), \
            'data/small_train/user2int.tsv', news2vector)
        user_dataloader = user_dataset.get_dataloader(batch_size=self.batch_size, \
            shuffle=False, num_workers=self.config.num_workers)
        for batch in tqdm(user_dataloader,
                            desc="Calculating vectors for users"):
            user_strings = batch["clicked_news_string"]
            if any(user_string not in user2vector for user_string in user_strings):
                clicked_news_vector = batch["clicked_news"]
                user_vector = self.model.get_user_vector(clicked_news_vector)
                for user, vector in zip(user_strings, user_vector):
                    if user not in user2vector:
                        user2vector[user] = vector.to('cpu')
        return user2vector
    
    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        self.news2vector = self.get_news2vector()
        self.user2vector = self.get_user2vector(self.news2vector)
            
        behaviors_dataset = BehaviorsDataset(path.join(self.test_dir, 'behaviors.tsv'))
        behaviors_dataloader = behaviors_dataset.get_dataloader(batch_size=1, \
            shuffle=False, num_workers=self.config.num_workers)
        y_trues, y_preds = [], []
        for batch in tqdm(behaviors_dataloader,
                            desc="Calculating probabilities"):
            candidate_news = batch["candidate_news"]
            candidate_news_vector = torch.stack([self.news2vector[new] for new in candidate_news[0]])
            user_vector = self.user2vector[batch["clicked_news_string"][0]]
            click_probability = self.model.get_prediction(candidate_news_vector,
                                                    user_vector)
            y_pred = click_probability.tolist()
            y_true = batch["y_true"][0]
            y_trues.extend(y_true)
            y_preds.extend(y_pred)
        results = calculate_single_user_metric(y_trues, y_preds)
        aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
        print("AUC:{:.5f}\t MRR:{:.5f}\t NDCG@5:{:.5f}\t NDCG@10:{:.5f}".format(aucs, mrrs, ndcg5s, ndcg10s))
        return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s)    
        
def test_run():
    state = torch.load("./save/{}/query-epoch1-loss1.17590-score0.62236.pt".format(model_name))
    pretrained_word_embedding = torch.from_numpy(
            np.load('data/small_train/pretrained_word_embedding.npy')).float()
    config = state["config"]
    model = Model(config, pretrained_word_embedding)
    model.load_state_dict(state["model"])
    tester = Tester(model, config)
    tester.test()
    return tester

if __name__ == "__main__":
    pass