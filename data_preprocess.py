"""
数据预处理
将api序列生成需要用的文本信息
"""
import config
import util
import numpy as np
import random


class ApiDataSet:
    def __init__(self, conf, total_num=11833148):
        self.data_dir = conf['data_dir']
        self.api_seq_path = conf['api_seq_path']
        self.voc, self.id_to_class = util.read_voc_from_file(self.data_dir + conf['voc_path'])
        self.freq = util.read_freq_from_file(self.data_dir + conf['freq_path'])

        # 方法中最多包含的对象调用长度
        self.max_func_len = conf['max_func_len']
        # 最长的类调用序列
        self.max_class_len = conf['max_class_len']
        # 最长的候选方法
        self.max_candidate_api_len = conf['max_candidate_api_len']
        # 文件长度
        self.total_num = total_num


    # 生成训练所需要的数据
    def __generate_data__(self, test_size=0.01, valid_size=0.01):
        """
        api_seq : [data_size, func_len, class_len] api调用序列
        fre_input: [data_size, func_len] 对象共现频率
        candidate_api_list : [data_size, max_class_len] 候选api选用序列
        correct_api : [data_size] 正确的api调用下标
        alpha : [data_size] 该条样本的权重
        center_class_attention : [data_size] 目标类
        context_class_attention : [data_size, func_len] 类别
        """
        api_seq_data = []
        fre_input_data = []
        candidate_api_data = []
        correct_api_data = []
        alpha_data = []
        center_class_attention_data = []
        context_class_attention_data = []

        test_count = int(self.total_num * test_size)
        valid_count = test_count + int(self.total_num * valid_size)

        count = 0
        with open(self.data_dir + self.api_seq_path, 'r') as f:
            for line in f:
                count += 1
                if count == self.total_num:
                    break
                api_list = [int(x) for x in line.split()]

                alpha_list = []
                sum_alpha = 0.0
                for i in range(1, len(api_list)):
                    api_seq, class_id_list = self.deal_with_api_list(api_list[:i])
                    predict_api = api_list[i]
                    predict_class = self.get_class_by_id(predict_api)

                    fre_input = [0.0] * self.max_func_len
                    context_class_attention = [0] * self.max_func_len

                    alpha = 0.0
                    for j in range(len(fre_input)):
                        if class_id_list[j] == 0:
                            continue
                        if predict_class in self.freq[class_id_list[j]]:
                            fre_input[j] = self.freq[class_id_list[j]][predict_class] / 100
                            context_class_attention[j] = class_id_list[j]
                            alpha += fre_input[j] * 1 / (i-j)
                    sum_alpha += alpha
                    if count > valid_count:
                        candidate_api = self.get_method_by_class(predict_class, is_train=True)
                    else:
                        candidate_api = self.get_method_by_class(predict_class, is_train=False)
                    # 如果预测的api不在candidate_api中
                    if predict_api not in candidate_api:
                        candidate_api[0] = predict_api
                    correct_api = candidate_api.index(predict_api)

                    api_seq_data.append(api_seq)
                    fre_input_data.append(fre_input)
                    candidate_api_data.append(candidate_api)
                    correct_api_data.append(correct_api)
                    center_class_attention_data.append(predict_class)
                    context_class_attention_data.append(context_class_attention)
                    alpha_list.append(alpha)
                # 对权重做归一化处理
                for alpha in alpha_list:
                    alpha_data.append(alpha / sum_alpha)

                if count == test_count:
                    test_data_set = self.package_data(api_seq_data,fre_input_data, candidate_api_data, correct_api_data, alpha_data, center_class_attention_data, context_class_attention_data)
                    self.save_to_file(test_data_set, 'test/')
                    print('save test data ... ')
                    api_seq_data = []
                    fre_input_data = []
                    candidate_api_data = []
                    correct_api_data = []
                    alpha_data = []
                    center_class_attention_data = []
                    context_class_attention_data = []
                    test_data_set = None

                if count == valid_count:
                    valid_data_set = self.package_data(api_seq_data,fre_input_data, candidate_api_data, correct_api_data, alpha_data, center_class_attention_data, context_class_attention_data)
                    self.save_to_file(valid_data_set, 'valid/')
                    print('save valid data...')
                    api_seq_data = []
                    fre_input_data = []
                    candidate_api_data = []
                    correct_api_data = []
                    center_class_attention_data = []
                    context_class_attention_data = []
                    alpha_data = []

            train_data_set = self.package_data(api_seq_data,fre_input_data, candidate_api_data, correct_api_data, alpha_data,center_class_attention_data,context_class_attention_data)
            self.save_to_file(train_data_set,'train/')
        print('data set process done...')

    def save_to_file(self, data_set, name):
        api_seq_data, fre_input_data, candidate_api_data, correct_api_data, alpha_data,center_class_attention_data,context_class_attention_data = data_set
        data_size = api_seq_data.shape[0]
        with open(self.data_dir + name + 'data_size' , 'w') as f:
            f.write('data_size = ' + str(api_seq_data.shape[0]) + '\n')
        api_seq_data = api_seq_data.reshape(data_size * self.max_func_len, self.max_class_len)
        np.savetxt(self.data_dir + name + 'api_seq_data.txt', api_seq_data, fmt="%d")
        np.savetxt(self.data_dir + name + 'fre_input_data.txt', fre_input_data, fmt="%f")
        np.savetxt(self.data_dir + name + 'candidate_api_data.txt', candidate_api_data, fmt="%d")
        np.savetxt(self.data_dir + name + 'correct_api_data.txt', correct_api_data, fmt="%d")
        np.savetxt(self.data_dir + name + 'alpha_data.txt', alpha_data, fmt="%f")
        np.savetxt(self.data_dir + name + 'center_class.txt', center_class_attention_data, fmt="%d")
        np.savetxt(self.data_dir + name + 'context_class.txt', context_class_attention_data,fmt="%d")

    def get_data_from_file(self, name, data_size):
        api_seq_data = np.loadtxt(self.data_dir + name + 'api_seq_data.txt', dtype=np.long)
        api_seq_data = api_seq_data.reshape(data_size, self.max_func_len, self.max_class_len)
        fre_input_data = np.loadtxt(self.data_dir + name + 'fre_input_data.txt', dtype=np.float32)
        candidate_api_data = np.loadtxt(self.data_dir + name + 'candidate_api_data.txt', dtype=np.long)
        correct_api_data = np.loadtxt(self.data_dir + name + 'correct_api_data.txt', dtype=np.int)
        alpha_data = np.loadtxt(self.data_dir + name + 'alpha_data.txt' ,dtype=np.float32)
        #center_class_attention_data = np.loadtxt(self.data_dir + name + 'center_class.txt', dtype=np.int)
        #context_class_attention_data = np.loadtxt(self.data_dir + name + 'context_class.txt', dtype=np.int)

        # numpy to tensor
        api_seq_data = util.gVar(api_seq_data)
        fre_input_data = util.gVar(fre_input_data)
        candidate_api_data = util.gVar(candidate_api_data)
        correct_api_data = util.gVar(correct_api_data)
        alpha_data = util.gVar(alpha_data)
        #center_class_attention_data = util.gVar(center_class_attention_data)
        #context_class_attention_data = util.gVar(context_class_attention_data)

        return api_seq_data, fre_input_data, candidate_api_data, correct_api_data, alpha_data  #, center_class_attention_data, context_class_attention_data

    def package_data(self, api_seq_data,fre_input_data, candidate_api_data, correct_api_data, alpha_data,center_class_attention_data,context_class_attention_data):
        print('data_set size is ', len(api_seq_data))
        return (np.array(api_seq_data, dtype=np.long), np.array(fre_input_data, dtype=np.float32),
                np.array(candidate_api_data, dtype=np.long), np.array(correct_api_data, dtype=np.int),
                np.array(alpha_data, dtype=np.float32),np.array(center_class_attention_data,dtype=np.int),np.array(context_class_attention_data,dtype=np.int))

    def deal_with_api_list(self, api_list):
        api_seq = np.zeros((self.max_func_len, self.max_class_len))
        result = {}
        classes = []
        for api in api_list:
            class_id = self.get_class_by_id(api)
            if class_id not in result:
                classes.append(class_id)
                result[class_id] = []
            result[class_id].append(api)
        if len(classes) > self.max_func_len:
            classes = classes[-1*self.max_func_len:]
        for i in range(len(classes)):
            apis = result[class_id][-1*self.max_class_len:]
            for j in range(len(apis)):
                api_seq[i][j] = apis[j]

        return api_seq.tolist(), self.pad_seq(classes, self.max_func_len)

    def get_method_by_class(self, class_id, is_train):
        method_list = [x for x in self.voc[class_id]]
        random.shuffle(method_list)
        if is_train:
            return self.pad_seq(method_list, 50)
        else:
            return self.pad_seq(method_list, self.max_candidate_api_len)

    def pad_seq(self, seq, maxlen):
        if len(seq)<maxlen:
            seq=np.append(seq, [0]*maxlen)
            seq=seq[:maxlen]
            seq = seq.tolist()
        else:
            seq=seq[:maxlen]
        return seq

    def get_class_by_id(self, api_id):
        return self.id_to_class[api_id]


if __name__ == '__main__':
    api = ApiDataSet(conf=config.get_config(),total_num=1000000)
    api.__generate_data__()
