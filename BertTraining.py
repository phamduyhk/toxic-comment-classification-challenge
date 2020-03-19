# coding: utf-8
import json
import os
import random
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
import torchtext
import pandas as pd
import datetime
import re
import string

# local import
from utils.bert import BertModel
from utils.dataloader import Preprocessing
from utils.bert import BertTokenizer, load_vocab
from utils.bert import get_config, BertModel, set_learned_params
from utils.EarlyStopping import EarlyStopping

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

preprocessing = Preprocessing()


class BertTraining(nn.Module):
    '''BERTモデルにIMDbのポジ・ネガを判定する部分をつなげたモデル'''

    def __init__(self, net_bert):
        super(BertTraining, self).__init__()

        # BERTモジュール
        self.bert = net_bert  # BERTモデル

        # headにポジネガ予測を追加
        # 入力はBERTの出力特徴量の次元、出力はポジ・ネガの2つ
        self.cls = nn.Linear(in_features=768, out_features=2)

        # 重み初期化処理
        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False,
                attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # BERTの基本モデル部分の順伝搬
        # 順伝搬させる
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            encoded_layers, pooled_output, attention_probs = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers, pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)

        # 入力文章の1単語目[CLS]の特徴量を使用して、ポジ・ネガを分類します
        vec_0 = encoded_layers[:, 0, :]
        vec_0 = vec_0.view(-1, 768)  # sizeを[batch_size, hidden_sizeに変換
        out = self.cls(vec_0)

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return out, attention_probs
        elif attention_show_flg == False:
            return out

 # 単語分割用のTokenizerを用意
def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

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


# 単語分割用のTokenizerを用意
tokenizer_bert = BertTokenizer(
    vocab_file="./weights/bert-base-uncased-vocab.txt", do_lower_case=True)


# 前処理と単語分割をまとめた関数を定義
# 単語分割の関数を渡すので、tokenizer_bertではなく、tokenizer_bert.tokenizeを渡す点に注意
def tokenizer_with_preprocessing(text, tokenizer=tokenizer_bert.tokenize):
    text = preprocessing_text(text)
    ret = tokenizer(text)  # tokenizer_bert
    return ret


def main():
    # define output dataframe
    sample = pd.read_csv("./data/sample_submission.csv")
    # データを読み込んだときに、読み込んだ内容に対して行う処理を定義します
    max_length = 256

    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                                use_vocab=True,
                                lower=True, include_lengths=True, batch_first=True, fix_length=max_length,
                                init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[UNK]')

    LABEL1 = torchtext.data.Field(sequential=False, use_vocab=False)
    LABEL2 = torchtext.data.Field(sequential=False, use_vocab=False)
    LABEL3 = torchtext.data.Field(sequential=False, use_vocab=False)
    LABEL4 = torchtext.data.Field(sequential=False, use_vocab=False)
    LABEL5 = torchtext.data.Field(sequential=False, use_vocab=False)
    LABEL6 = torchtext.data.Field(sequential=False, use_vocab=False)

    # (注釈)：各引数を再確認
    # sequential: データの長さが可変か？文章は長さがいろいろなのでTrue.ラベルはFalse
    # tokenize: 文章を読み込んだときに、前処理や単語分割をするための関数を定義
    # use_vocab：単語をボキャブラリーに追加するかどうか
    # lower：アルファベットがあったときに小文字に変換するかどうか
    # include_length: 文章の単語数のデータを保持するか
    # batch_first：ミニバッチの次元を先頭に用意するかどうか
    # fix_length：全部の文章を指定した長さと同じになるように、paddingします
    # init_token, eos_token, pad_token, unk_token：文頭、文末、padding、未知語に対して、どんな単語を与えるかを指定

    # フォルダ「data」から各tsvファイルを読み込みます
    # BERT用で処理するので、10分弱時間がかかります
    temp_path = preprocessing.reformat_csv_header(
            path="./data", train_file="train.csv", test_file="test.csv")

    print("temp path {}".format(temp_path))
    print("text {}".format(vars(TEXT)))
            
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=temp_path, train='train.csv',
        test='test.csv', format='csv',
        fields=[('Text', TEXT), ('toxic', LABEL1), ('severe_toxic', LABEL2), ('obscene', LABEL3),
                ('threat', LABEL4), ('insult', LABEL5), ('identity_hate', LABEL6)])


    # torchtext.data.Datasetのsplit関数で訓練データとvalidationデータを分ける
    train_ds, val_ds = train_val_ds.split(
        split_ratio=0.8, random_state=random.seed(1234))

    # BERTはBERTが持つ全単語でBertEmbeddingモジュールを作成しているので、ボキャブラリーとしては全単語を使用します
    # そのため訓練データからボキャブラリーは作成しません

    vocab_bert, ids_to_tokens_bert = load_vocab(
        vocab_file="./weights/bert-base-uncased-vocab.txt")

    # このまま、TEXT.vocab.stoi= vocab_bert (stoiはstring_to_IDで、単語からIDへの辞書)としたいですが、
    # 一度bulild_vocabを実行しないとTEXTオブジェクトがvocabのメンバ変数をもってくれないです。
    # （'Field' object has no attribute 'vocab' というエラーをはきます）

    # 1度適当にbuild_vocabでボキャブラリーを作成してから、BERTのボキャブラリーを上書きします
    TEXT.build_vocab(train_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_bert

    # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
    batch_size = 32  # BERTでは16、32あたりを使用する

    train_dl = torchtext.data.Iterator(
        train_ds, batch_size=batch_size, train=True)

    val_dl = torchtext.data.Iterator(
        val_ds, batch_size=batch_size, train=False, sort=False)

    test_dl = torchtext.data.Iterator(
        test_ds, batch_size=batch_size, train=False, sort=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dl, "val": val_dl}

    print(vars(train_ds[0]))
    print(vars(test_ds[0]))

    # モデル設定のJOSNファイルをオブジェクト変数として読み込みます
    config = get_config(file_path="./weights/bert_config.json")

    # BERTモデルを作成します
    net_bert = BertModel(config)

    # BERTモデルに学習済みパラメータセットします
    net_bert = set_learned_params(
        net_bert, weights_path="./weights/pytorch_model.bin")

    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        # モデル構築
        net = BertTraining(net_bert)

        # 訓練モードに設定
        net.train()

        print('done setup network')

        # 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

        # 1. まず全部を、勾配計算Falseにしてしまう
        for name, param in net.named_parameters():
            param.requires_grad = False

        # 2. 最後のBertLayerモジュールを勾配計算ありに変更
        for name, param in net.bert.encoder.layer[-1].named_parameters():
            param.requires_grad = True

        # 3. 識別器を勾配計算ありに変更
        for name, param in net.cls.named_parameters():
            param.requires_grad = True

        # 最適化手法の設定

        # BERTの元の部分はファインチューニング
        optimizer = optim.Adam([
            {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
            {'params': net.cls.parameters(), 'lr': 5e-5}
        ], betas=(0.9, 0.999))

        # 損失関数の設定
        criterion = nn.CrossEntropyLoss()
        # nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 学習・検証を実行する。1epochに20分ほどかかります
        num_epochs = 2
        net_trained = train_model(net, dataloaders_dict,
                                  criterion, optimizer, num_epochs=num_epochs, label=label, device=device)

        # 学習したネットワークパラメータを保存します
        save_path = './weights/bert_fine_tuning_weights.pth'
        torch.save(net_trained.state_dict(), save_path)

        # テストデータ
        net_trained.eval()  # モデルを検証モードに
        net_trained.to(device)  # GPUが使えるならGPUへ送る

        predicts = []

        for batch in tqdm(test_dl):  # testデータのDataLoader
            # batchはTextとLableの辞書オブジェクト
            # GPUが使えるならGPUにデータを送る
            inputs = batch.Text[0].to(device)  # 文章

            # 順伝搬（forward）計算
            with torch.set_grad_enabled(False):
                # BertForIMDbに入力
                outputs = net_trained(inputs, token_type_ids=None, attention_mask=None,
                                      output_all_encoded_layers=False, attention_show_flg=False)

                _, preds = torch.max(outputs, 1)  # ラベルを予測

                preds = preds.cpu()
                preds = preds.numpy().tolist()

                predicts += preds

        sample[label] = predicts

    # save predictions
    if not os.path.exists("./submission"):
        os.mkdir("./submission")
    sample.to_csv("./submission/submission_Bert_{}_{}ep.csv".format(
        datetime.datetime.now().date(), num_epochs), index=False)


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, label, device="cpu"):
    # GPUが使えるかを確認
    print("using device：", device)
    print('-----start-------')

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = dataloaders_dict["train"].batch_size

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_metrics = 0  # epochの正解数
            iteration = 1

            # 開始時刻を保存
            t_epoch_start = time.time()
            t_iter_start = time.time()

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書型変数

                # GPUが使えるならGPUにデータを送る
                inputs = batch.Text[0].to(device)  # 文章
                labels = getattr(batch, label)
                labels = labels.to(device)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # BertForIMDbに入力
                    outputs = net(inputs, token_type_ids=None, attention_mask=None,
                                  output_all_encoded_layers=False, attention_show_flg=False)

                    loss = criterion(outputs, labels)  # 損失を計算

                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            acc = (torch.sum(preds == labels.data)
                                   ).double() / batch_size
                            print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec. || 本イテレーションの正解率：{}'.format(
                                iteration, loss.item(), duration, acc))
                            t_iter_start = time.time()

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    y_true = labels.data.cpu()
                    preds = preds.cpu()
                    epoch_metrics += roc_auc_score(y_true, preds)

            # epochごとのlossと正解率
            t_epoch_finish = time.time()
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_eval = epoch_metrics / len(dataloaders_dict[phase])

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs,
                                                                           phase, epoch_loss, epoch_eval))
            t_epoch_start = time.time()

    return net


if __name__ == "__main__":
    main()
