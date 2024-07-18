import re
import random
import time
from statistics import mode
import os
import math
from PIL import Image
import numpy as np
import pandas
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from transformers import CLIPTokenizer, CLIPModel
from transformers import ViltProcessor, ViltForQuestionAnswering
from torchvision import transforms, datasets, models
from torch.nn.utils.rnn import pad_sequence
import torchtext
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import gc

start_time = time.time()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#========================BERT・CLIP分散表現==================================================
def process_text(text):
    # lowercase
    text = text.lower()

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

#CLIP
model_clip, preprocess = clip.load("ViT-B/32", device)


#========================BERT・CLIP分散表現==================================================



# ===================================1. データローダーの作成===================================
# 通常のデータローダー
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_csv_path, df_json_path,
                 image_dir, transform=None, answer=True,
                 cm_path='../class_mapping.csv'):
        self.transform = transform # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_json_path)
        if df_csv_path is not None: # Check if df_csv_path is provided
            self.df_csv = pandas.read_csv(df_csv_path)  # 画像ファイルのパス，question, answerを持つDataFrame
            def string_to_tensor(embedding_str):
                # カンマ区切りの文字列をリストに変換し、各要素をfloatに変換
                embedding_list = list(map(float, embedding_str.split(',')))
                # リストをテンソルに変換
                embedding_tensor = torch.tensor(embedding_list)
                return embedding_tensor

            # データフレームの 'embeddings' 列をテンソルに変換
            self.df_csv['embedding'] = self.df_csv['embedding'].apply(string_to_tensor)
            self.df_csv['clip_question'] = self.df_csv['clip_question'].apply(string_to_tensor)
        else:
            self.df_csv = None # Or any other appropriate handling
        self.cm_df = pandas.read_csv(cm_path)
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    answer = process_text(answer["answer"])
                    if answer not in self.answer2idx:
                        self.answer2idx[answer] = len(self.answer2idx)
            for answer in self.cm_df["answer"]:
                answer = process_text(answer)
                if answer not in self.answer2idx:
                    self.answer2idx[answer] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        question = self.df_csv['embedding'][idx]

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            answers = torch.Tensor(answers).long()
            target = torch.zeros(len(self.answer2idx))
            for i in answers:
                target[i] += 1
            target = target / len(answers)
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, torch.Tensor(question), torch.Tensor(answers), torch.Tensor(target), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)

#CLIP用データローダー
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df_csv_path, df_json_path,
                 image_dir, transform=None, answer=True,
                 cm_path='../class_mapping.csv'):
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_json_path)
        if df_csv_path is not None: # Check if df_csv_path is provided
            self.df_csv = pandas.read_csv(df_csv_path)  # 画像ファイルのパス，question, answerを持つDataFrame
            def string_to_tensor(embedding_str):
                # カンマ区切りの文字列をリストに変換し、各要素をfloatに変換
                embedding_list = list(map(float, embedding_str.split(',')))
                # リストをテンソルに変換
                embedding_tensor = torch.tensor(embedding_list)
                return embedding_tensor

            # データフレームの 'embeddings' 列をテンソルに変換
            self.df_csv['clip_question'] = self.df_csv['clip_question'].apply(string_to_tensor)
            self.df_csv['clip_image'] = self.df_csv['clip_image'].apply(string_to_tensor)
        else:
            self.df_csv = None # Or any other appropriate handling
        self.cm_df = pandas.read_csv(cm_path)
        self.answer = answer


        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    answer = process_text(answer["answer"])
                    if answer not in self.answer2idx:
                        self.answer2idx[answer] = len(self.answer2idx)
            for answer in self.cm_df["answer"]:
                answer = process_text(answer)
                if answer not in self.answer2idx:
                    self.answer2idx[answer] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = self.df_csv['clip_image'][idx]
        question = self.df_csv['clip_question'][idx]

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            answers = torch.Tensor(answers).long()
            target = torch.zeros(len(self.answer2idx))
            for i in answers:
                target[i] += 1
            target = target / len(answers)
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, torch.Tensor(question), torch.Tensor(answers), torch.Tensor(target), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)

# BLIP用データローダー
class BLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df_csv_path, df_json_path,
                 image_dir, transform=None, answer=True,
                 cm_path='../class_mapping.csv'):
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_json_path)
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
        self.transform = transform

        if df_csv_path is not None: # Check if df_csv_path is provided
            self.df_csv = pandas.read_csv(df_csv_path)  # 画像ファイルのパス，question, answerを持つDataFrame
            def string_to_tensor(embedding_str):
                # カンマ区切りの文字列をリストに変換し、各要素をfloatに変換
                embedding_list = list(map(float, embedding_str.split(',')))
                # リストをテンソルに変換
                embedding_tensor = torch.tensor(embedding_list)
                return embedding_tensor

            # データフレームの 'embeddings' 列をテンソルに変換
            self.df_csv['clip_question'] = self.df_csv['clip_question'].apply(string_to_tensor)
            self.df_csv['clip_image'] = self.df_csv['clip_image'].apply(string_to_tensor)
            self.df_csv['blip_question'] = self.df_csv['blip_question'].apply(string_to_tensor)
            self.df_csv['blip_image'] = self.df_csv['blip_image'].apply(string_to_tensor)
        else:
            self.df_csv = None # Or any other appropriate handling
        self.cm_df = pandas.read_csv(cm_path)
        self.answer = answer


        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    answer = process_text(answer["answer"])
                    if answer not in self.answer2idx:
                        self.answer2idx[answer] = len(self.answer2idx)
            for answer in self.cm_df["answer"]:
                answer = process_text(answer)
                if answer not in self.answer2idx:
                    self.answer2idx[answer] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = self.df_csv['blip_image'][idx]
        question = self.df_csv['blip_question'][idx]

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            answers = torch.Tensor(answers).long()
            target = torch.zeros(len(self.answer2idx))
            for i in answers:
                target[i] += 1
            target = target / len(answers)
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, torch.Tensor(question), torch.Tensor(answers), torch.Tensor(target), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)


# ===================================1. データローダーの作成===================================




# ===================================2. 評価指標の実装=========================================
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

def optimized_VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        # 各答えに対して、一致する予測の数をカウント
        matches = (pred.unsqueeze(0) == answers.unsqueeze(1)).sum(dim=1) - 1
        acc = torch.clamp(matches.float() / 3, max=1).sum() / 10
        total_acc += acc
    return total_acc / len(batch_pred)
# ===================================2. 評価指標の実装=========================================




# ===================================3. モデルの実装===========================================


#<<<<<<<<<<<<<<<<<<<<<<<<<< Closs Attention >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class CrossAttention(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.d_model = d_model
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_features, text_features):
        # img_features: (batch_size, num_img_features, d_model)
        # text_features: (batch_size, seq_len, d_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Transpose for multihead attention: (seq_len, batch_size, d_model)
        text_features = text_features.transpose(0, 1)
        img_features = img_features.transpose(0, 1)
        # 特徴ベクトル次元に変換（例として線形変換を使用）
        # ここでの変換方法は具体的な用途に依存する
        text_features = text_features.unsqueeze(1).to(device)
        img_features = img_features.unsqueeze(1).to(device)

        #txt_projection_layer = nn.Linear(text_features.size(-1), self.d_model, device = device)
        #text_features = txt_projection_layer(text_features)  # (512, 128, d_model)
        #img_projection_layer = nn.Linear(img_features.size(-1), self.d_model, device = device)
        #img_features = img_projection_layer(img_features)  # (512, 128, d_model)

        # Apply multihead attention
        txt_attn_output, txt_attn_weights = self.multihead_attn(text_features, img_features, img_features)
        img_attn_output, img_attn_weights = self.multihead_attn(img_features, text_features, text_features)

        # Add & Norm
        text_features = self.dropout(txt_attn_output)
        text_features = self.layer_norm1(text_features)
        img_features = self.dropout(img_attn_output)
        img_features = self.layer_norm2(img_features)

        # Apply feed forward layer with dropout and normalization
        txt_output = F.relu(self.dropout(self.layer_norm2(text_features)))
        img_output = F.relu(self.dropout(self.layer_norm2(img_features)))

        return txt_output.transpose(0, 1), txt_attn_weights, img_output.transpose(0, 1), img_attn_weights
#<<<<<<<<<<<<<<<<<<<<<<<<<< Closs Attention >>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Self-Attention >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class CompactSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads=8, dropout=0.1):
        super(CompactSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # nn.MultiheadAttention expects inputs of shape (sequence_length, batch_size, embed_size)
        # So we need to add a batch dimension to the input tensor (sequence_length, embed_size) to (sequence_length, 1, embed_size)
        x = x.unsqueeze(1)  # Add batch dimension

        # MultiheadAttention layer
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)

        # Remove the added batch dimension
        attn_output = attn_output.squeeze(1)
        attn_output = self.layer_norm(attn_output)

        return attn_output
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Self-Attention >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.sketch1 = nn.Parameter(torch.randint(0, output_dim, (input_dim1,)).float())
        self.sketch2 = nn.Parameter(torch.randint(0, output_dim, (input_dim2,)).float())
        self.sign1 = nn.Parameter(2 * (torch.randint(0, 2, (input_dim1,)) - 0.5).float())
        self.sign2 = nn.Parameter(2 * (torch.randint(0, 2, (input_dim2,)) - 0.5).float())

    def forward(self, x, y):
        # Apply Count Sketch
        x_sketch = self.count_sketch(x, self.sketch1, self.sign1)
        y_sketch = self.count_sketch(y, self.sketch2, self.sign2)

        # Perform FFT
        x_fft = torch.fft.fft(x_sketch, n=self.output_dim)
        y_fft = torch.fft.fft(y_sketch, n=self.output_dim)

        # Element-wise product in FFT space
        z_fft = x_fft * y_fft

        # Inverse FFT
        z = torch.fft.ifft(z_fft, n=self.output_dim)
        z = z.real  # Only take the real part

        return z

    def count_sketch(self, x, sketch, sign):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.output_dim).to(x.device)
        for i in range(x.size(1)):
            output[:, sketch[i].long()] += sign[i] * x[:, i]
        return output


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VQAModel_1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 自前のVQAモデル
class VQAModel_1(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int, train_batch_size):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.mcb = CompactBilinearPooling(512, 512, 16000)

        self.fc = nn.Sequential(
            nn.Linear(16000, 768),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(768, n_answer)
        )

    def forward(self, image, question, max_len, len_seq):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #x = torch.cat([image, question], dim=1)
        x = self.mcb(image, question)
        x = self.fc(x)

        return x
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VQAModel_1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VQAModel_2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class VQAModel_2(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int, train_batch_size):
        super().__init__()
        self.train_batch_size = train_batch_size

        self.fc = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(768, n_answer)
        )

    def forward(self, image, question, max_len, len_seq):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = torch.cat([image, question], dim=1)
        x = self.fc(x)

        return x
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VQAModel_2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VQAModel_3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class VQAModel_3(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int, train_batch_size):
        super().__init__()
        self.train_batch_size = train_batch_size

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question, max_len, len_seq):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = torch.cat([image, question], dim=1)

        x = self.fc(x)

        return x
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VQAModel_3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# ===================================3. モデルの実装=========================================




# ===================================4. 学習の実装============================================
def train(model, dataloader, optimizer, criterion, device, train_batch_size):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, target, mode_answer in dataloader:
        image, question, answer, target, mode_answer = \
            image.to(device), question.to(device), answers.to(device), target.to(device), mode_answer.to(device)

        # Calculate the lengths of each question sequence
        len_seq = torch.tensor([question.size(1)] * question.size(0))
        max_len = question.size(1) # All questions have the same length after BERT embedding

        pred = model(image, question, max_len, len_seq) # Pass max_len and len_seq to the model
        pred = F.log_softmax(pred,dim=1).to(device)


        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def train_2(model, dataloader, optimizer, criterion, device, train_batch_size):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for inputs, answers, target, mode_answer in dataloader:
        answer, target, mode_answer = \
            answers.to(device), target.to(device), mode_answer.to(device)

        # Calculate the lengths of each question sequenc
        inputs = {k:v.to(device) for k,v in inputs.items()}
        outputs = model(**inputs)
        pred = outputs.logits
        pred = F.log_softmax(pred,dim=1).to(device)

        #target = F.one_hot(mode_answer, num_classes=pred.shape[1]).float()
        #target = F.softmax(target, dim=1).to(device)

        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start



def eval(model, dataloader, optimizer, criterion, device, train_batch_size):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        # Calculate the lengths of each question sequence
        len_seq = torch.tensor([question.size(1)] * question.size(0))
        print(len_seq.device)
        max_len = question.size(1) # All questions have the same length after BERT embedding

        pred = model(image, question, max_len, len_seq) # Pass max_len and len_seq to the model
        pred = F.log_softmax(pred,dim=1).to(device)

        target = F.one_hot(mode_answer, num_classes=pred.shape[1]).float()
        target = F.softmax(target, dim=1).to(device)

        loss = criterion(pred, target)

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start
# ===================================4. 学習の実装============================================


# ===================================4.1. learning =========================================

def learning(model, train_loader, train, optimizer, scheduler, criterion, device, num_epoch, train_batch_size):
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device,
                                                                    train_batch_size = train_batch_size)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

# ===================================4.1. learning =========================================

# deviceの設定
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(32, padding=(4, 4, 4, 4), padding_mode='constant'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])
train_batch_size = 128

train_dataset_1 = CLIPDataset(df_csv_path="../train_emb.csv",
                           df_json_path="../train.json",
                           image_dir="../train")
train_dataset_2 = CLIPDataset(df_csv_path="../train_emb.csv",
                           df_json_path="../train.json",
                           image_dir="../train")
train_dataset_3 = CLIPDataset(df_csv_path="../train_emb.csv",
                           df_json_path="../train.json",
                           image_dir="../train")
test_dataset_1 = CLIPDataset(df_csv_path="../valid_emb.csv",
                          df_json_path="../valid.json",
                          image_dir="../valid", answer=False)
test_dataset_2 = CLIPDataset(df_csv_path="../valid_emb.csv",
                          df_json_path="../valid.json",
                          image_dir="../valid", answer=False)
test_dataset_3 = CLIPDataset(df_csv_path="../valid_emb.csv",
                          df_json_path="../valid.json",
                          image_dir="../valid", answer=False)
test_dataset_1.update_dict(train_dataset_1)
test_dataset_2.update_dict(train_dataset_2)
test_dataset_3.update_dict(train_dataset_3)

# dataloader / model
train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=128, shuffle=True, num_workers=2)
train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=128, shuffle=True, num_workers=2)
train_loader_3 = torch.utils.data.DataLoader(train_dataset_3, batch_size=128, shuffle=True, num_workers=2)
test_loader_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=1, shuffle=False, num_workers=2)
test_loader_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=1, shuffle=False, num_workers=2)
test_loader_3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=1, shuffle=False, num_workers=2)


model_1 = VQAModel_1(vocab_size=len(train_dataset_1.question2idx)+1, n_answer=len(train_dataset_1.answer2idx),
                 train_batch_size = train_batch_size)
model_1 = model_1.to(device)
model_2 = VQAModel_2(vocab_size=len(train_dataset_2.question2idx)+1, n_answer=len(train_dataset_2.answer2idx),
                 train_batch_size = train_batch_size)
model_2 = model_2.to(device)
model_3 = VQAModel_3(vocab_size=len(train_dataset_3.question2idx)+1, n_answer=len(train_dataset_3.answer2idx),
                 train_batch_size = train_batch_size)
model_3 = model_3.to(device)


# optimizer / criterion
criterion = nn.KLDivLoss(reduction = 'batchmean')
num_epoch_1 = 30
num_epoch_2 = 25
num_epoch_3 = 25
optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=0.001, weight_decay=1e-5)
optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=0.001, weight_decay=1e-5)
optimizer_3 = torch.optim.AdamW(model_3.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=num_epoch_1)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=num_epoch_2)
scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=num_epoch_3)


learning(model_1, train_loader_1, train, optimizer_1, scheduler_1, criterion, device, num_epoch_1, train_batch_size)
torch.save(model_1.state_dict(), "model_1.pth")
learning(model_2, train_loader_2, train, optimizer_2, scheduler_2, criterion, device, num_epoch_2, train_batch_size)
torch.save(model_2.state_dict(), "model_2.pth")
learning(model_3, train_loader_3, train, optimizer_3, scheduler_3, criterion, device, num_epoch_3, train_batch_size)
torch.save(model_3.state_dict(), "model_3.pth")
gc.collect()
torch.cuda.empty_cache()

model_1 = VQAModel_1(vocab_size=len(train_dataset_1.question2idx)+1, n_answer=len(train_dataset_1.answer2idx),
                 train_batch_size = train_batch_size)
model_1.load_state_dict(torch.load('model_1.pth'))
model_1.to(device)
model_2 = VQAModel_2(vocab_size=len(train_dataset_1.question2idx)+1, n_answer=len(train_dataset_1.answer2idx),
                 train_batch_size = train_batch_size)
model_2.load_state_dict(torch.load('model_2.pth'))
model_2.to(device)
model_3 = VQAModel_3(vocab_size=len(train_dataset_3.question2idx)+1, n_answer=len(train_dataset_3.answer2idx),
                 train_batch_size = train_batch_size)
model_3.load_state_dict(torch.load('model_3.pth'))
model_3.to(device)

# 提出用ファイルの作成
model_1.eval()
submission = []
submission_average1 = []
for image, question in test_loader_1:
    image, question = image.to(device), question.to(device)
    len_seq = torch.tensor([question.size(1)] * question.size(0))
    max_len = question.size(1) # All questions have the same length after BERT embedding
    pred = model_1(image, question, max_len, len_seq)
    pred_ave = pred.clone().detach().cpu()
    pred_ave = F.softmax(pred_ave, dim=1)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)
    submission_average1.append(pred_ave)

submission = [train_dataset_1.idx2answer[id] for id in submission]
submission = np.array(submission)
np.save("submission_1.npy", submission)
print('finish:1')

model_2.eval()
submission = []
submission_average2 = []
for image, question in test_loader_2:
    image, question = image.to(device), question.to(device)
    len_seq = torch.tensor([question.size(1)] * question.size(0))
    max_len = question.size(1) # All questions have the same length after BERT embedding
    pred = model_2(image, question, max_len, len_seq)
    pred_ave = pred.clone().detach().cpu()
    pred_ave = F.softmax(pred_ave, dim=1)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)
    submission_average2.append(pred_ave)

submission = [train_dataset_2.idx2answer[id] for id in submission]
submission = np.array(submission)
np.save("submission_2.npy", submission)
print('finish:2')

model_3.eval()
submission = []
submission_average3 = []
for image, question in test_loader_3:
    image, question = image.to(device), question.to(device)
    len_seq = torch.tensor([question.size(1)] * question.size(0))
    max_len = question.size(1) # All questions have the same length after BERT embedding
    pred = model_3(image, question, max_len, len_seq)
    pred_ave = pred.clone().detach().cpu()
    pred_ave = F.softmax(pred_ave, dim=1)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)
    submission_average3.append(pred_ave)

submission = [train_dataset_3.idx2answer[id] for id in submission]
submission = np.array(submission)
np.save("submission_3.npy", submission)
print('finish:3')


submission_average = []
for i in range(len(submission_average2)):
    pred_average = (submission_average1[i] + submission_average2[i] + submission_average3[i]) / 3
    pred_average = pred_average.argmax(1).cpu().item()
    submission_average.append(pred_average)

submission_average = [train_dataset_1.idx2answer[id] for id in submission_average]
submission_average = np.array(submission_average)
np.save("submission_average.npy", submission_average)

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} [s]")

