import os
import random
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import math
import warnings

import torch
from torch import nn
import torch.optim as optim
import tokenizers
from transformers import RobertaModel, RobertaConfig
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


"""
Part 0: Hyper-parameters
"""
SEED = 229

MAXLEN = 64
BATCH_SIZE = 32
NUM_EPOCH = 8
SPLIT = 5
LR = 1e-5


"""
Part I: Raw Data
"""
class Raw_Dataset:
    def __init__(self, tweet_file_path, nasdaq_file_path):
        self.tweet = Raw_Dataset.__read_tweet_dataset(tweet_file_path)
        self.nasdaq = Raw_Dataset.__read_nasdaq_dataset(nasdaq_file_path)
        self.data_df = self.__combine_data()
        self.train_val_df, self.test_df = self.__split_train_test()
        self.reweighted_train_val_df = self.__re_weighting()

    @staticmethod
    def __read_tweet_dataset(tweet_file_path):
        tweet = pd.read_csv(tweet_file_path)[['date', 'content']]
        tweet['date'] = pd.to_datetime(tweet['date'])
        tweet = tweet[tweet['date'] >= '2016-11-09'].reset_index(drop=True)
        return tweet

    @staticmethod
    def __read_nasdaq_dataset(nasdaq_file_path):
        nasdaq = pd.read_csv(nasdaq_file_path)[['Date', 'Close']]
        nasdaq['Date'] = pd.to_datetime(nasdaq['Date']).dt.strftime('%Y-%m-%d')

        nasdaq['Tweet_Start_Date'], nasdaq['Tweet_End_Date'] = np.NaN, np.NaN
        nasdaq['Return'], nasdaq['Return_Bucket'] = np.NaN, np.NaN
        for i, row in nasdaq.iterrows():
            if i - 1 < 0:
                continue
            nasdaq.loc[i, 'Tweet_Start_Date'] = nasdaq.iloc[i - 1]['Date']
            nasdaq.loc[i, 'Tweet_End_Date'] = nasdaq.iloc[i]['Date']
            ret = math.log(nasdaq.iloc[i]['Close'] / nasdaq.iloc[i - 1]['Close'])
            nasdaq.loc[i, 'Return'] = ret
            nasdaq.loc[i, 'Return_Bucket'] = 1 if ret < 0 else 0

        return nasdaq.iloc[1:]

    def __combine_data(self):
        tweet, nasdaq = self.tweet, self.nasdaq
        result = pd.DataFrame(columns=['date', 'content', 'nasdaq_date', 'return', 'return_bucket'])
        for i, row in nasdaq.iterrows():
            tweet_filtered = tweet[(tweet['date'] >= row['Tweet_Start_Date']) & \
                                   (tweet['date'] < row['Tweet_End_Date'])]
            tweet_filtered = tweet_filtered[tweet_filtered['content'].apply(lambda x: len(x.split(' '))) >= 5]
            tweet_filtered['nasdaq_date'] = row['Tweet_End_Date']
            tweet_filtered['return'] = row['Return']
            tweet_filtered['return_bucket'] = row['Return_Bucket']
            tweet_filtered = tweet_filtered[['date', 'content', 'nasdaq_date', 'return', 'return_bucket']]
            result = result.append(tweet_filtered, ignore_index = True)
        return result

    def __split_train_test(self):
        data_df = self.data_df
        train_val_df, test_df, _, _ = train_test_split(data_df, data_df['return_bucket'], test_size=0.20,
                                                       random_state=SEED)
        print(f"Dataset Shape --- data_df {data_df.shape}, train_val_df {train_val_df.shape}, test_df {test_df.shape}")
        return train_val_df.reset_index(), test_df.reset_index()

    def __re_weighting(self):
        train_val_df = self.train_val_df
        ratio = len(train_val_df[train_val_df['return_bucket'] == 0]) // len(
            train_val_df[train_val_df['return_bucket'] == 1])
        if ratio < 1:
            return

        reweighted_train_val_df = train_val_df[train_val_df['return_bucket'] == 0]
        for i in range(ratio):
            reweighted_train_val_df = reweighted_train_val_df.append(train_val_df[train_val_df['return_bucket'] == 1])
        print(
            f'Reweighted Dataset Shape --- reweighted_ratio: {ratio}, reweighted_train_df: {reweighted_train_val_df.shape}')
        return reweighted_train_val_df


"""
Part II: Encoding
"""
class BERT_Encoded_Dataset(torch.utils.data.Dataset):
    # Description:
    #    Inherit from torch.utils.data.Dataset,
    #    which is an abstract class representing a dataset. All subclasses should
    #    overwrite __getitem__ to map index into each sample and optionally overwrite
    #    __len__ to get number of samples.
    #    (Ref. "https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset")

    def __init__(self, df, max_len=MAXLEN):
        self.df = df
        self.max_len = max_len
        self.labeled = 'return_bucket' in df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='./input/roberta-base/vocab.json',
            merges_file='./input/roberta-base/merges.txt',
            lowercase=True,
            add_prefix_space=True)

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        data['tweet'], data['ids'], data['masks'] = self._get_input_data_with_encoding(row)
        if self.labeled:
            data['output'] = row['return_bucket']

        return data

    def __len__(self):
        # Calculate Number of Samples via len()
        return len(self.df)

    def _get_input_data_with_encoding(self, row):
        # Calculate ids: ids = <s> + encoding_ids + </s>
        tweet = " " + " ".join(row['content'].lower().split())
        encoding = self.tokenizer.encode(tweet)
        ids = [0] + encoding.ids + [2]

        # Pad
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len  # Pad with 1
        else:
            ids = ids[0:MAXLEN]

        # Translate ids, masks, offsets into Tensors
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))

        return tweet, ids, masks


def get_train_val_dataloaders(train_val_df, train_idx, val_idx, batch_size=8):
    train_df, val_df = train_val_df.iloc[train_idx], train_val_df.iloc[val_idx]
    train_dataloader = torch.utils.data.DataLoader(
        BERT_Encoded_Dataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        BERT_Encoded_Dataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    return {"train": train_dataloader, "val": val_dataloader}


def get_test_dataloaders(test_df, batch_size=8):
    test_dataloader = torch.utils.data.DataLoader(
        BERT_Encoded_Dataset(test_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)
    return test_dataloader


"""
Part III: BERT + NN Model
"""
class BERT_NN(nn.Module):
    def __init__(self, max_len=MAXLEN, batch_size=BATCH_SIZE):
        super(BERT_NN, self).__init__()

        self.max_len = max_len
        self.batch_size = BATCH_SIZE

        # roBERTa
        with open('./input/roberta-base/config.json') as f:
            config_display = json.load(f)
            print('BERT Config --- ' + str(config_display))
        config = RobertaConfig.from_pretrained(
            './input/roberta-base/config.json', output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(
            './input/roberta-base/pytorch_model.bin', config=config)
        self.hidden_size = config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # CNN
        self.cnn_output_size = 2
        self.conv1 = torch.nn.Conv1d(self.hidden_size, 256, 3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                                     padding_mode='zeros')
        self.conv2 = torch.nn.Conv1d(256, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                                     padding_mode='zeros')
        self.conv3 = torch.nn.Conv1d(16, self.cnn_output_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                                     padding_mode='zeros')

        # Linear
        self.fc = nn.Linear(self.max_len * self.cnn_output_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        num_sample = input_ids.shape[0]

        # roBERTa
        # Dimension Description:
        #    input_ids/attention_mask dimensions - [-1, sequence_length=128, hidden_size=768]
        #    os dimensions -  [-1, sequence_length=128, hidden_size=768]
        #    ms dimensions - [-1, hidden_size=768]
        #    hs tuple dimensions - 13*[-1, sequence_length=128, hidden_size=768]
        os, ms, hs = self.roberta(input_ids, attention_mask)

        # Mean
        x = torch.stack([hs[-1], hs[-2], os])  # [-1, 3, sequence_length=128, hidden_size=768]
        x = torch.mean(x, 0)  # [-1, sequence_length=128, hidden_size=768]

        # CNN
        x = x.transpose(1, 2)  # [-1, hidden_size=768, sequence_length=128]
        x = self.conv1(x)  # [-1, 256, sequence_length=128]
        x = self.conv2(x)  # [-1, 16, sequence_length=128]
        x = self.conv3(x)  # [-1, cnn_output_size=2, sequence_length=128]

        # Dropout
        x = self.dropout(x)

        # Linear
        x = x.reshape(num_sample, self.max_len * self.cnn_output_size)
        x = self.fc(x)

        return x


# Evaluation
class Confusion_Matrix:
    def __init__(self):
        self.tp, self.fp, self.fn, self.tn = 0.0, 0.0, 0.0, 0.0
        self.precision, self.recall, self.F1_score = None, None, None

    def update(self, batch_pred, batch_true, pred_is_probs=True):
        batch_tp, batch_fp, batch_fn, batch_tn = Confusion_Matrix.__calculate(batch_pred, batch_true, pred_is_probs)
        self.tp += batch_tp
        self.fp += batch_fp
        self.fn += batch_fn
        self.tn += batch_tn
        self.__evaluate()

    @staticmethod
    def __calculate(pred, true, pred_is_probs):  # Evaluation Function
        if pred_is_probs:
            label_pred = np.argmax(pred, axis=1)
        else:
            label_pred = pred
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(true)):
            tp = tp + 1 if true[i] == 1 and label_pred[i] == 1 else tp
            fp = fp + 1 if true[i] == 0 and label_pred[i] == 1 else fp
            fn = fn + 1 if true[i] == 1 and label_pred[i] == 0 else fn
            tn = tn + 1 if true[i] == 0 and label_pred[i] == 0 else tn
        return tp, fp, fn, tn

    def __evaluate(self):  # Evaluation Function
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        self.precision_positive = tp / (tp + fp) if tp != 0 else 0.000001
        self.recall_positive = tp / (tp + fn) if tp != 0 else 0.000001
        self.F1_score_positive = 2 / (1 / self.precision_positive + 1 / self.recall_positive) if tp != 0 else 0.000001
        self.precision_negative = tn / (tn + fn) if tn != 0 else 0.000001
        self.recall_negative = tn / (tn + fp) if tn != 0 else 0.000001
        self.F1_score_negative = 2 / (1 / self.precision_negative + 1 / self.recall_negative) if tp != 0 else 0.000001
        self.macro_F1_Score = 0.5 * (self.F1_score_positive + self.F1_score_negative)
        self.accuracy = (tp + tn) / (tp + fp + fn + tn)


"""
Part IV: Training
"""
class BERT_NN_Training:
    @staticmethod
    def run_k_fold_training(train_val_df, test_df, num_epoch=NUM_EPOCH):
        skf = StratifiedKFold(n_splits=SPLIT, shuffle=True, random_state=SEED)
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['return_bucket']), start=1):
            print(f'Fold: {fold}')

            # Model
            model = BERT_NN()
            if torch.cuda.is_available():
                model.cuda()
            else:
                model.cpu()

            # Optimizer
            optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1.2e-2)

            # Data Loaders
            loaders_dict = get_train_val_dataloaders(train_val_df, train_idx, val_idx, BATCH_SIZE)
            loaders_dict['test'] = get_test_dataloaders(test_df)

            # Train and Predict
            BERT_NN_Training.__predict(model, loaders_dict, -1, 0)
            for epoch in range(num_epoch):
                BERT_NN_Training.__train(model, loaders_dict, optimizer, epoch, num_epoch)
                BERT_NN_Training.__predict(model, loaders_dict, epoch, num_epoch)

    @staticmethod
    def __train(model, loaders_dict, optimizer, epoch, num_epoch):
        for phase in ['train', 'val']:
            start_time = time.time()

            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss, num_samples = 0.0, len(loaders_dict[phase].dataset)
            epoch_conf_mtr = Confusion_Matrix()

            for batch_data in tqdm(loaders_dict[phase]):
                tweet, ids, masks, output_true = BERT_NN_Training.__unpack_data(batch_data)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Propagate Forward
                    output_pred = model(ids, masks)

                    # Calculate Loss and Propagate Backward
                    batch_loss = BERT_NN_Training.__loss_fn(output_pred, output_true)
                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()
                    epoch_loss += batch_loss.item() * len(ids)

                    # Evaluate a Batch
                    output_pred = torch.softmax(output_pred, dim=1).cpu().detach().numpy()
                    output_true = output_true.cpu().detach().numpy()
                    epoch_conf_mtr.update(output_pred, output_true)

            # Time
            end_time = time.time()
            time_cost = end_time - start_time

            # Print Summary for Each Epoch
            epoch_loss = epoch_loss / num_samples
            print(f'Epoch {epoch + 1}/{num_epoch} --- Time {time_cost:.2f} Second | ' +
                  f'{phase} | Number Samples: {num_samples} | Loss: {epoch_loss:.2f} | ' +
                  f'TP, FP, FN, TN: {epoch_conf_mtr.tp:.0f}, {epoch_conf_mtr.fp:.0f}, {epoch_conf_mtr.fn:.0f}, {epoch_conf_mtr.tn:.0f} | ' +
                  f'(Positive) P, R, F1: {epoch_conf_mtr.precision_positive:.2f}, {epoch_conf_mtr.recall_positive:.2f}, {epoch_conf_mtr.F1_score_positive:.2f} | ' +
                  f'Macro F1, A: {epoch_conf_mtr.macro_F1_Score:.2f}, {epoch_conf_mtr.accuracy:.2f}')

    @staticmethod
    def __predict(model, loaders_dict, epoch, num_epoch):
        conf_mtr = Confusion_Matrix()
        num_samples = len(loaders_dict['test'].dataset)
        for batch_data in tqdm(loaders_dict['test']):
            tweet, ids, masks, output_true = BERT_NN_Training.__unpack_data(batch_data)
            output_pred = model(ids, masks)

            # Evaluate a Batch
            output_pred = torch.softmax(output_pred, dim=1).cpu().detach().numpy()
            output_true = output_true.cpu().detach().numpy()
            conf_mtr.update(output_pred, output_true)
        print(f'Epoch {epoch + 1}/{num_epoch} --- ' +
              f'Test | Number Samples: {num_samples} | ' +
              f'TP, FP, FN, TN: {conf_mtr.tp:.0f}, {conf_mtr.fp:.0f}, {conf_mtr.fn:.0f}, {conf_mtr.tn:.0f} | ' +
              f'(Positive) P, R, F1: {conf_mtr.precision_positive:.2f}, {conf_mtr.recall_positive:.2f}, {conf_mtr.F1_score_positive:.2f} | ' +
              f'Macro F1, A: {conf_mtr.macro_F1_Score:.2f}, {conf_mtr.accuracy:.2f}')

    @staticmethod
    def __loss_fn(pred, true):  # Loss Function
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(pred, true)

    @staticmethod
    def __unpack_data(batch_data):
        if torch.cuda.is_available():
            tweet, ids, masks = batch_data['tweet'], batch_data['ids'].cuda(), batch_data['masks'].cuda()
            output_true = batch_data['output'].long().cuda()
        else:
            tweet, ids, masks = batch_data['tweet'], batch_data['ids'].cpu(), batch_data['masks'].cpu()
            output_true = batch_data['output'].long().cpu()
        return tweet, ids, masks, output_true


"""
Main Functions
"""
if __name__ == "__main__":
    seed_everything(seed_value=SEED)
    raw_dataset = Raw_Dataset(tweet_file_path='realdonaldtrump.csv', nasdaq_file_path='^IXIC_short.csv')
    BERT_NN_Training.run_k_fold_training(raw_dataset.reweighted_train_val_df, raw_dataset.test_df)
