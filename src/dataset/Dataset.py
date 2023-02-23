import torch
from torch.utils.data import Dataset
from torch.utils import data
import pandas as pd
from ast import literal_eval
from os import path
import numpy as np
from config import model_name
import importlib
import torch

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(BaseDataset, self).__init__()
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities'
        ] for attribute in config.dataset_attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in config.dataset_attributes['record'])

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(
                    self.news2dict[key1][key2])
        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [0] * config.num_words_title,
            'abstract': [0] * config.num_words_abstract,
            'title_entities': [0] * config.num_words_title,
            'abstract_entities': [0] * config.num_words_abstract
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in config.dataset_attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [
            self.news2dict[x] for x in row.candidate_news.split()
        ]
        # TODO: 
        item["clicked_news"] = [
            self.news2dict[x]
            # for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
            for x in row.clicked_news.split()[-config.num_clicked_news_a_user:]
        ]
        if 'clicked_news_length' in config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]

        return item
    
    def get_NRMS_dataloader(self, batch_size, shuffle, num_workers):
        
        def collate_fn(batch):
            candidate_news = torch.stack([torch.stack([n["title"] for n in i["candidate_news"]]) for i in batch]) # batch_size, K+1, num_words
            clicked_news = torch.stack([torch.stack([n["title"] for n in i["clicked_news"]]) for i in batch])
            candidate_news = torch.permute(candidate_news, (1, 0, 2)) # K+1, batch_size, num_words
            clicked_news = torch.permute(clicked_news, (1, 0, 2))
            candidate_news = torch.split(candidate_news, 1, dim=0)
            clicked_news = torch.split(clicked_news, 1, dim=0)
            return {
                "candidate_news": [{"title": i} for i in candidate_news],
                "clicked_news": [{"title": i} for i in clicked_news]
            }
            
        def collate_fn_NAML(batch):
            return {
                "candidate_news": [
                                    {str(key) : torch.stack([b["candidate_news"][i][key] \
                                        for b in batch]) for key in config.dataset_attributes['news'] }
                                        for i in range(config.negative_sampling_ratio + 1) ],
                "clicked_news": [
                                    {str(key) : torch.stack([b["clicked_news"][i][key] \
                                        for b in batch]) for key in config.dataset_attributes['news'] }
                                        for i in range(config.num_clicked_news_a_user) ]
            }
        
        if model_name == "NAML":
            fn = collate_fn_NAML
        else:
            fn = collate_fn
        
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=fn)
    

class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities', 'category', 'subcategory'
                ])
            })
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item

    def get_dataloader(self, batch_size, shuffle, num_workers):
        
        def collate_fn(batch):

            return {
                "id": [i["id"] for i in batch], #batch,
                "title": torch.stack([i["title"] for i in batch]) # batch, num_words
            }
            
        def collate_fn_NAML(batch):
            return{
                'id': [i["id"] for i in batch],
                'news': {str(key) : torch.stack([b[key] \
                            for b in batch]) for key in config.dataset_attributes['news'] }
            }
        if model_name == "NAML":
            fn = collate_fn_NAML
        else:
            fn = collate_fn
            
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=fn)
        
        

class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, user2int_path, news2vector):
        super(UserDataset, self).__init__()
        self.news2vector = news2vector
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[-config.num_clicked_news_a_user:]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item

    def get_dataloader(self, batch_size, shuffle, num_workers):
        
        def collate_fn(batch):
            return {
                "clicked_news_string": [i["clicked_news_string"] for i in batch], #batch,
                "clicked_news": torch.stack([torch.stack([self.news2vector[n] for n in i['clicked_news']]) for i in batch])
            }
            
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=collate_fn)
        
class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item

    def get_dataloader(self, batch_size, shuffle, num_workers):
        
        def collate_fn(batch):
            candidate_news = [[c.split('-')[0] for c in i["impressions"]] for i in batch]
            clicked_news_string = [i["clicked_news_string"] for i in batch]
            y_true = [[int(c.split('-')[1]) for c in i["impressions"]] for i in batch]
            return {
                "candidate_news": candidate_news, #batch,
                "clicked_news_string": clicked_news_string,
                "y_true": y_true
            }
            
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=collate_fn)

