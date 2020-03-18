# coding: utf-8

import math
import numpy as np

import torch
from torch import nn


# BERT用にLayerNormalization層を定義します。
# 実装の細かな点をTensorFlowに合わせています。


class BertLayerNorm(nn.Module):
    """LayerNormalization層 """

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


# BERTのEmbeddingsモジュールです


class BertEmbeddings(nn.Module):
    """文章の単語ID列と、1文目か2文目かの情報を、埋め込みベクトルに変換する
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        # 3つのベクトル表現の埋め込み

        # Token Embedding：単語IDを単語ベクトルに変換、
        # vocab_size = 30522でBERTの学習済みモデルで使用したボキャブラリーの量
        # hidden_size = 768 で特徴量ベクトルの長さは768
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        # （注釈）padding_idx=0はidx=0の単語のベクトルは0にする。BERTのボキャブラリーのidx=0が[PAD]である。

        # Transformer Positional Embedding：位置情報テンソルをベクトルに変換
        # Transformerの場合はsin、cosからなる固定値だったが、BERTは学習させる
        # max_position_embeddings = 512　で文の長さは512単語
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        # Sentence Embedding：文章の1文目、2文目の情報をベクトルに変換
        # type_vocab_size = 2
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # 作成したLayerNormalization層
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Dropout　'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        '''
        input_ids： [batch_size, seq_len]の文章の単語IDの羅列
        token_type_ids：[batch_size, seq_len]の各単語が1文目なのか、2文目なのかを示すid
        '''

        # 1. Token Embeddings
        # 単語IDを単語ベクトルに変換
        words_embeddings = self.word_embeddings(input_ids)

        # 2. Sentence Embedding
        # token_type_idsがない場合は文章の全単語を1文目として、0にする
        # そこで、input_idsと同じサイズのゼロテンソルを作成
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 3. Transformer Positional Embedding：
        # [0, 1, 2 ・・・]と文章の長さだけ、数字が1つずつ昇順に入った
        # [batch_size, seq_len]のテンソルposition_idsを作成
        # position_idsを入力して、position_embeddings層から768次元のテンソルを取り出す
        seq_length = input_ids.size(1)  # 文章の長さ
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # 3つの埋め込みテンソルを足し合わせる [batch_size, seq_len, hidden_size]
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # LayerNormalizationとDropoutを実行
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayer(nn.Module):
    '''BERTのBertLayerモジュールです。Transformerになります'''

    def __init__(self, config):
        super(BertLayer, self).__init__()

        # Self-Attention部分
        self.attention = BertAttention(config)

        # Self-Attentionの出力を処理する全結合層
        self.intermediate = BertIntermediate(config)

        # Self-Attentionによる特徴量とBertLayerへの元の入力を足し算する層
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embedderモジュールの出力テンソル[batch_size, seq_len, hidden_size]
        attention_mask：Transformerのマスクと同じ働きのマスキング
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output  # [batch_size, seq_length, hidden_size]


class BertAttention(nn.Module):
    '''BertLayerモジュールのSelf-Attention部分です'''

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        '''
        input_tensor：Embeddingsモジュールもしくは前段のBertLayerからの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''
            self_output, attention_probs = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output


class BertSelfAttention(nn.Module):
    '''BertAttentionのSelf-Attentionです'''

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        # num_attention_heads': 12

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * \
                             self.attention_head_size  # = 'hidden_size': 768

        # Self-Attentionの特徴量を作成する全結合層
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        '''multi-head Attention用にテンソルの形を変換する
        [batch_size, seq_len, hidden] → [batch_size, 12, seq_len, hidden/12] 
        '''
        new_x_shape = x.size()[
                      :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールもしくは前段のBertLayerからの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        # 入力を全結合層で特徴量変換（注意、multi-head Attentionの全部をまとめて変換しています）
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # multi-head Attention用にテンソルの形を変換
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 特徴量同士を掛け算して似ている度合をAttention_scoresとして求める
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
                           math.sqrt(self.attention_head_size)

        # マスクがある部分にはマスクをかけます
        attention_scores = attention_scores + attention_mask
        # （備考）
        # マスクが掛け算でなく足し算なのが直感的でないですが、このあとSoftmaxで正規化するので、
        # マスクされた部分は-infにしたいです。 attention_maskには、0か-infが
        # もともと入っているので足し算にしています。

        # Attentionを正規化する
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # ドロップアウトします
        attention_probs = self.dropout(attention_probs)

        # Attention Mapを掛け算します
        context_layer = torch.matmul(attention_probs, value_layer)

        # multi-head Attentionのテンソルの形をもとに戻す
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # attention_showのときは、attention_probsもリターンする
        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer


class BertSelfOutput(nn.Module):
    '''BertSelfAttentionの出力を処理する全結合層です'''

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 'hidden_dropout_prob': 0.1

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states：BertSelfAttentionの出力テンソル
        input_tensor：Embeddingsモジュールもしくは前段のBertLayerからの出力
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    '''Gaussian Error Linear Unitという活性化関数です。
    LeLUが0でカクっと不連続なので、そこを連続になるように滑らかにした形のLeLUです。
    '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    '''BERTのTransformerBlockモジュールのFeedForwardです'''

    def __init__(self, config):
        super(BertIntermediate, self).__init__()

        # 全結合層：'hidden_size': 768、'intermediate_size': 3072
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        # 活性化関数gelu
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        '''
        hidden_states： BertAttentionの出力テンソル
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)  # GELUによる活性化
        return hidden_states


class BertOutput(nn.Module):
    '''BERTのTransformerBlockモジュールのFeedForwardです'''

    def __init__(self, config):
        super(BertOutput, self).__init__()

        # 全結合層：'intermediate_size': 3072、'hidden_size': 768
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # 'hidden_dropout_prob': 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states： BertIntermediateの出力テンソル
        input_tensor：BertAttentionの出力テンソル
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# BertLayerモジュールの繰り返し部分モジュールの繰り返し部分です


class BertEncoder(nn.Module):
    def __init__(self, config):
        '''BertLayerモジュールの繰り返し部分モジュールの繰り返し部分です'''
        super(BertEncoder, self).__init__()

        # config.num_hidden_layers の値、すなわち12 個のBertLayerモジュールを作ります
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：返り値を全TransformerBlockモジュールの出力にするか、
        それとも、最終層だけにするかのフラグ。
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # 返り値として使うリスト
        all_encoder_layers = []

        # BertLayerモジュールの処理を繰り返す
        for layer_module in self.layer:

            if attention_show_flg == True:
                '''attention_showのときは、attention_probsもリターンする'''
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
            elif attention_show_flg == False:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg)

            # 返り値にBertLayerから出力された特徴量を12層分、すべて使用する場合の処理
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # 返り値に最後のBertLayerから出力された特徴量だけを使う場合の処理
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        # attention_showのときは、attention_probs（最後の12段目）もリターンする
        if attention_show_flg == True:
            return all_encoder_layers, attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers


class BertPooler(nn.Module):
    '''入力文章の1単語目[cls]の特徴量を変換して保持するためのモジュール'''

    def __init__(self, config):
        super(BertPooler, self).__init__()

        # 全結合層、'hidden_size': 768
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 1単語目の特徴量を取得
        first_token_tensor = hidden_states[:, 0]

        # 全結合層で特徴量変換
        pooled_output = self.dense(first_token_tensor)

        # 活性化関数Tanhを計算
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertModel(nn.Module):
    '''モジュールを全部つなげたBERTモデル'''

    def __init__(self, config):
        super(BertModel, self).__init__()

        # 3つのモジュールを作成
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # Attentionのマスクと文の1文目、2文目のidが無ければ作成する
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # マスクの変形　[minibatch, 1, 1, seq_length]にする
        # 後ほどmulti-head Attentionで使用できる形にしたいので
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # マスクは0、1だがソフトマックスを計算したときにマスクになるように、0と-infにする
        # -infの代わりに-10000にしておく
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 順伝搬させる
        # BertEmbeddinsモジュール
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # BertLayerモジュール（Transformer）を繰り返すBertEncoderモジュール
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''

            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        # BertPoolerモジュール
        # encoderの一番最後のBertLayerから出力された特徴量を使う
        pooled_output = self.pooler(encoded_layers[-1])

        # output_all_encoded_layersがFalseの場合はリストではなく、テンソルを返す
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output
