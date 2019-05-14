import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import os

"""
将词向量编码作为输入到Lstm中,预测下一个api
"""


class ApiModel(nn.Module):
    def __init__(self, emb_dim, hidden_size, vector_weights, need_attention):
        super(ApiModel,self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.api_embedding = nn.Embedding.from_pretrained(vector_weights,)

        self.api_embedding.weight.requires_grad = False

        self.class_rnn = nn.LSTM(input_size=self.emb_dim,
                                 hidden_size=self.hidden_size)
        self.func_rnn = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size)
        self.linear = nn.Linear(in_features=self.hidden_size,
                                out_features=self.emb_dim,
                                bias=True)
        self.need_attention = need_attention
        # 对象层编码的损失
        self.class_loss = None

    def forward(self, input, freq_input, input_lengths=None):
        batch_size, func_len, class_len = input.size()

        """
        词嵌入层
        """
        class_input = input.view(-1, class_len)
        # shape= [batch_size*func_len, class_seq_len, emb_size]
        input_emb = self.api_embedding(class_input)

        """
        对象层次的rnn
        """
        # shape = [class_seq_len, batch_size*func_len , hidden_size]
        class_output, hidden = self.class_rnn(input_emb.permute([1, 0, 2]))
        class_output = F.dropout(class_output, 0.25, self.training)

        """
        方法层次的rnn
        """
        # 取最后一个隐藏状态作为输出
        func_input = class_output[-1].view(batch_size, func_len, -1)
        # shape = [func_len, batch_size, hidden_size]
        func_output, hidden = self.func_rnn(func_input.permute([1, 0, 2]))
        func_output = F.dropout(func_output, 0.25, self.training)
        # shape = [batch_size, func_len, hidden_size]
        func_output = func_output.permute([1, 0, 2])

        """
        使用待预测对象的共现频率向量做加权平均
        """
            # shape = [batch_size, func_len,  hidden_size]
        if self.need_attention == 1:
            func_output = torch.mul(func_output,freq_input.unsqueeze(2))

        # shape = [batch_size, hidden_size]
        output_pool = F.avg_pool1d(func_output.transpose(1, 2), func_len).squeeze(2)
        encoding = F.tanh(output_pool)

        """
        全联接层转化为emb_dim维度
        """
        encoding = self.linear(encoding)
        return encoding


class ApiRecommendationModel(nn.Module):
    def __init__(self, config, center_vec_path,context_vec_path, need_attention):
        super(ApiRecommendationModel, self).__init__()

        self.apiModel = ApiModel(emb_dim=config['emb_dim'],
                                 hidden_size=config['hidden_size'],
                                 vector_weights=util.read_vector_from_file(center_vec_path),
                                 need_attention= need_attention)
        # 编码层的词嵌入
        self.api_embedding = nn.Embedding.from_pretrained(util.read_vector_from_file(context_vec_path))
        self.api_embedding.weight.requires_grad = False

    """
    api_seq : [batch_size, func_len, class_len] api调用序列
    fre_input: [batch_size, func_len] 对象共现频率
    candidate_api_list : [batch_size, max_class_len] 候选api选用序列
    correct_api : [batch_size, max_class_len] 正确的api调用下标, one-hot编码
    """
    def forward(self, api_seq, freq_input, candidate_api_list):
        func_encoding = self.apiModel(api_seq, freq_input)

        """
        将方法的隐藏状态 * api编码表示转移概率
        """
        # shape = [batch_size, max_class_len, emb_dim]
        candidate_api_emb = self.api_embedding(candidate_api_list)
        # shape = [batch_size, max_class_len]
        score = torch.sum(torch.mul(func_encoding.unsqueeze(1), candidate_api_emb), dim=2)

        return score









