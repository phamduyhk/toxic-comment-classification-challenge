# coding: utf-8
import glob
import os
import io
import string
import re
import random
import spacy
import torchtext
from torchtext.vocab import Vectors
import pandas as pd
import torch


class Preprocessing():
    def __init__(self):
        pass

    def get_data(self, path, train_file, test_file, vectors, max_length, batch_size):
        """
        :param path (str)               : path to train and test data
        :param train_file (str)         : train data file name (Except csv)
        :param test_file (str)          : test data file name (Except csv)
        :param vector_list (str)        : vector list file path
        :param max_length (int)         : max length of output text
        :param batch_size (int)         : batch size

        :return:
            train_dl
            val_dl
            test_dl
            TEXT: "comment_text"

        :detail:
            LABEL<1,2,3,4,5,6>: "toxic","severe_toxic","obscene","threat","insult","identity_hate"
        """
        # データを読み込んだときに、読み込んだ内容に対して行う処理を定義します
        TEXT = torchtext.data.Field(sequential=True, tokenize=self.tokenizer_with_preprocessing, use_vocab=True,
                                    lower=True, include_lengths=True, batch_first=True, fix_length=max_length,
                                    init_token="<cls>", eos_token="<eos>")
        LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

        temp_path = self.reformat_csv_header(
            path=path, train_file=train_file, test_file=test_file)

        train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
            path=temp_path, train=train_file,
            test=test_file, format='csv',
            fields=[('Text', TEXT), ('toxic', LABEL), ('severe_toxic', LABEL), ('obscene', LABEL),
                    ('threat', LABEL), ('insult', LABEL), ('identity_hate', LABEL)])

        train_ds, val_ds = train_val_ds.split(
            split_ratio=0.7, random_state=random.seed(2395))

        # torchtextで単語ベクトルとして英語学習済みモデルを読み込みます
        english_fasttext_vectors = Vectors(name=vectors)

        # ベクトル化したバージョンのボキャブラリーを作成します
        TEXT.build_vocab(
            train_ds, vectors=english_fasttext_vectors, min_freq=10)

        # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
        train_dl = torchtext.data.Iterator(
            train_ds, batch_size=batch_size, train=True)

        val_dl = torchtext.data.Iterator(
            val_ds, batch_size=batch_size, train=False, sort=False)

        test_dl = torchtext.data.Iterator(
            test_ds, batch_size=batch_size, train=False, sort=False)

        return train_dl, val_dl, test_dl, TEXT

    def reformat_csv_header(self, path, train_file, test_file):
        """
        remove index col in csv file
        :arg
            :param path (str)               : path to train and test data
            :param train_file (str)         : train data file name (Except csv)
            :param test_file (str)          : test data file name (Except csv)

        Return:
            temp path
        """

        """
        "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
        """

        train = pd.read_csv(os.path.join(path, train_file))
        test = pd.read_csv(os.path.join(path, test_file))
        train = train.drop('id', axis=1)
        test = test.drop('id', axis=1)
        # for label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
        #     test[label] = pd.Series(0, index=test.index)
        temp_path = os.path.join(path, "temp")
        if not os.path.isdir(temp_path):
            os.mkdir(temp_path)
        train.to_csv(os.path.join(temp_path, train_file),
                     index=False, header=False)
        test.to_csv(os.path.join(temp_path, test_file),
                    index=False, header=False)
        return temp_path

    @staticmethod
    def preprocessing_text(text):
        # 改行コードを消去
        text = re.sub('¥n', '', text)

        # 　数字を消去
        for num in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            text = re.sub(num, '', text)

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")

        # ピリオドなどの前後にはスペースを入れておく
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        return text

    # 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）
    @staticmethod
    def tokenizer_punctuation(text):
        return text.strip().split()

    # 前処理と分かち書きをまとめた関数を定義
    def tokenizer_with_preprocessing(self, text):
        text = self.preprocessing_text(text)
        ret = self.tokenizer_punctuation(text)
        return ret


if __name__ == '__main__':
    path = "../data/"
    train_file = "train.csv"
    test_file = "test.csv"
    vector_list = "../data/wiki-news-300d-1M.vec"
    instance = Preprocessing()
    train_dl, val_dl, test_dl, TEXT = instance.get_data(path=path, train_file=train_file, test_file=test_file,
                                                        vectors=vector_list, max_length=256,
                                                        batch_size=1280)
