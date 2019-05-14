import torch
import numpy as np


def read_vector_from_file(vector_path):
    return torch.from_numpy(np.loadtxt(vector_path,dtype=np.float32))


def read_freq_from_file(freq_file_path):
    # 从文件读取类共现文件
    freq = {}
    with open(freq_file_path, 'r') as f:
        for line in f:
            center_class_id, context_class_id, count = line.split()
            center_class_id = int(center_class_id)
            context_class_id = int(context_class_id)
            count = float(count)

            if center_class_id not in freq:
                freq[center_class_id] = {}
            if context_class_id not in freq[center_class_id]:
                freq[center_class_id][context_class_id] = count
    return freq


# 读取字典表
def read_voc_from_file(voc_file_path):
    voc = {}
    id_to_class = {}
    with open(voc_file_path,'r') as f:
        for line in f:
            class_id, api_id, new_id = (int(x) for x in line.split())
            if class_id not in voc:
                voc[class_id] = {}
            if api_id not in voc[class_id]:
                voc[class_id][api_id] = new_id
            if new_id not in id_to_class:
                id_to_class[new_id] = class_id

    return voc, id_to_class


"""
评测系统指标
"""


def get_api_len(api_seq):
    api_len = []
    # batch_size
    for i in range(api_seq.shape[0]):
        api_len.append(len(np.flatnonzero(api_seq[i].cpu().data.numpy())))
    return api_len


def evaluate(scores, correct_api, api_len):
    acc_1 = acc_2 = acc_3 = acc_4 = 0
    acc_1_with_dis = acc_2_with_dis = acc_3_with_dis = acc_4_with_dis = 0
    total_num = len(scores)
    total_num_with_dis = 0
    top_loc = np.argsort(-scores)
    for i in range(len(top_loc)):
        current_correct_api = correct_api[i]
        #print(current_correct_api)
        if current_correct_api in top_loc[i][:1]:
            acc_1 += 1
            acc_1_with_dis += api_len[i]
        if current_correct_api in top_loc[i][:2]:
            acc_2 += 1
            acc_2_with_dis += api_len[i]
        if current_correct_api in top_loc[i][:3]:
            acc_3 += 1
            acc_3_with_dis += api_len[i]
        if current_correct_api in top_loc[i][:4]:
            acc_4 += 1
            acc_4_with_dis += api_len[i]
        total_num_with_dis += api_len[i]
    return acc_1, acc_2, acc_3, acc_4, total_num, acc_1_with_dis, acc_2_with_dis, acc_3_with_dis, acc_4_with_dis,total_num_with_dis


use_cuda = torch.cuda.is_available()

def gVar(data):
    tensor=data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    #if use_cuda:
        #tensor = tensor.cuda()
    return tensor
