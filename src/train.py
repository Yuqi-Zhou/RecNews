import os
import importlib
from os import path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from config import model_name, add_query
from dataset.Dataset import BaseDataset, NewsDataset, UserDataset, BehaviorsDataset
from utils.metrics import calculate_single_user_metric

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
except AttributeError:
    print(f"{model_name} not included!")
    exit()

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.epoch = self.config.num_epochs
        self.batch_size = self.config.batch_size

        self.train_set = BaseDataset("data/small_train/behaviors_parsed.tsv", "data/small_train/news_parsed.tsv")
        self.dev_dir = "data/small_dev"
        self.train_dataloader = self.train_set.get_NRMS_dataloader(batch_size=config.batch_size, \
            shuffle=True, num_workers=config.num_workers)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), \
            lr=config.lr, weight_decay=config.weight_decay)
        
        if not os.path.exists("./save/{}".format(model_name)):
            os.mkdir("save/{}".format(model_name))

    def save_state_dict(self, filename):
        save_path = os.path.join("./save/{}".format(model_name), filename)
        torch.save({
            "model": self.model.state_dict(),
            "config": self.config
        }, save_path)

    def get_news2vector(self):
        # get news to vector dic
        news2vector = {}
        news_dataset = NewsDataset(path.join(self.dev_dir, 'news_parsed.tsv'))
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
        user_dataset = UserDataset(path.join(self.dev_dir, 'behaviors.tsv'), \
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
    def eval_epoch(self, epoch):
        # TODO
        self.model.eval()
        self.model.to(self.device)
        if epoch == 1:
            self.news2vector = self.get_news2vector()
            self.user2vector = self.get_user2vector(self.news2vector)
            
        behaviors_dataset = BehaviorsDataset(path.join(self.dev_dir, 'behaviors.tsv'))
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
        return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s)

    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)
        iterator_bar = tqdm(self.train_dataloader)
        step_sum = len(iterator_bar)
        loss_sum = 0.0
        for batch in iterator_bar:
            y_pred = self.model(batch["candidate_news"], batch["clicked_news"])
            y = torch.zeros(len(y_pred)).long().to(self.device)
            loss = self.loss_fn(y_pred, y)

            bar_description = "EPOCH[{}] LOSS[{:.5f}] ".format(epoch, loss.item())
            iterator_bar.set_description(bar_description)

            loss_sum += loss.item()

            loss.backward()
            self.optimizer.zero_grad()

        avg_loss = loss_sum / step_sum
        return avg_loss


    def train(self):
        min_loss = float('inf')
        max_score = float('-inf')
        print("Start Training")
        count = 0
        for epoch in range(1, self.epoch+1):
            avg_loss = self.train_epoch(epoch)
            self.last_epoch_avg_loss = avg_loss
            print("--- EPOCH[{}] AVG_LOSS[{:.5f}]".format(epoch, avg_loss))
            avg_score, _, _, _ = self.eval_epoch(epoch)
            if avg_loss < min_loss:
                min_loss = avg_loss
            if avg_score > max_score:
                count = 0
                max_score = avg_score
                if add_query is True:
                    self.save_state_dict(filename="query-epoch{}-loss{:.5f}-score{:.5f}.pt".format(epoch, avg_loss, avg_score))
                else:
                    self.save_state_dict(filename="epoch{}-loss{:.5f}-score{:.5f}.pt".format(epoch, avg_loss, avg_score))
            else:
                count+=1
                if count >= 2:
                    print("Early stop!")
                    exit()
        self.optimizer.zero_grad()
    
    
        
def train_run(config):
    pretrained_word_embedding = torch.from_numpy(
            np.load('data/small_train/pretrained_word_embedding.npy')).float()
    model = Model(config, pretrained_word_embedding)
    trainer = Trainer(model, config)
    trainer.train()
    return trainer

if __name__ == "__main__":
    pass