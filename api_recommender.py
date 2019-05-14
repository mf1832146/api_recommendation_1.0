import util
import data_preprocess as preprocess
import torch
import torch.utils.data as Data
from torch import optim
import model as api_model
import numpy as np
import torch.nn as nn
import config
import os
import sys


class ApiRecommender():
    def __init__(self, conf, model_path , need_loss_weight):
        self.conf = conf
        self.model = None
        self.optimizer = None
        self.path = conf['workdir'] + model_path
        self.need_loss_weight = need_loss_weight

    def log(self, msg):
        if not os.path.exists(self.path + 'log/'):
            os.makedirs(self.path + 'log/')
        with open(self.path +'log/log.txt', 'a+') as f:
            f.write(msg + '\n')

    def save_model(self, model, epoch):
        if not os.path.exists(self.path + 'models/'):
            os.makedirs(self.path + 'models/')
        torch.save(model.state_dict(), self.path + 'models/epo%d.h5' % epoch)

    def load_model(self, model, epoch):
        assert os.path.exists(self.path + 'models/epo%d.h5' % epoch), 'Weights at epoch %d not found' % epoch
        model.load_state_dict(torch.load(self.path + 'models/epo%d.h5' % epoch))

    # 训练
    def train(self):
        log_every = self.conf['log_every']
        save_every = self.conf['save_every']
        nb_epoch = self.conf['nb_epoch']

        api_data = preprocess.ApiDataSet(self.conf)

        train_api_seq_data, train_fre_input_data, train_candidate_api_data, train_correct_api_data, train_alpha_data = api_data.get_data_from_file(name='train/', data_size=3769596)
        self.log('loading train_data from file..')
        valid_api_seq_data, valid_fre_input_data, valid_candidate_api_data, valid_correct_api_data, valid_alpha_data = api_data.get_data_from_file(name='valid/', data_size=37905)
        self.log('loading valid_data from file...')


        train_set = Data.TensorDataset(train_api_seq_data, train_fre_input_data, train_candidate_api_data, train_correct_api_data, train_alpha_data)
        valid_set = Data.TensorDataset(valid_api_seq_data, valid_fre_input_data, valid_candidate_api_data, valid_correct_api_data, valid_alpha_data)

        train_data_loader = Data.DataLoader(dataset=train_set, batch_size=self.conf['batch_size'],
                                                      shuffle=True, drop_last=True, num_workers=1)
        valid_data_loader = Data.DataLoader(dataset=valid_set, batch_size=self.conf['batch_size'],
                                                      shuffle=True, drop_last=True, num_workers=1)
        print('loading data finished')

        criterion = nn.CrossEntropyLoss(reduce=False)
        top_acc_3 = 0.0
        for epoch in range(self.conf['reload'] + 1, nb_epoch):
            itr = 1
            losses = []
            for api_seq_data, fre_input_data, candidate_api_data, correct_api_data,alpha_data in train_data_loader:
                # 指定训练模式
                self.model.train(mode=True)
                if torch.cuda.is_available():
                    api_seq_data = api_seq_data.cuda()
                    fre_input_data = fre_input_data.cuda()
                    candidate_api_data = candidate_api_data.cuda()
                    correct_api_data = correct_api_data.cuda()
                    alpha_data = alpha_data.cuda()

                score = self.model(api_seq_data, fre_input_data, candidate_api_data)

                batch_loss = criterion(score, correct_api_data)
                # print(batch_loss.shape)

                if self.need_loss_weight:
                    loss = torch.sum(torch.mul(batch_loss, alpha_data))
                    total_alpha = torch.sum(alpha_data)
                    loss = loss / total_alpha
                else:
                    loss = torch.mean(batch_loss)
                # 将损失乘上权重
                #loss = torch.mul(loss, alpha_data)
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if itr % log_every == 0:
                    self.log('epo:[%d/%d] itr:%d Loss=%.5f' % (
                    epoch, nb_epoch, itr, np.mean(losses)))
                    losses = []

                if itr % save_every == 0:
                    top_acc_3 = self.valid_model(valid_data_loader, top_acc_3, epoch)
                itr = itr + 1

        # 验证测试集上的准确率
        self.test_model(api_data)

    def test_model(self, api_data):
        test_api_seq_data, test_fre_input_data, test_candidate_api_data, test_correct_api_data, test_alpha_data = api_data.get_data_from_file(
            name='test/', data_size=39040)
        test_set = Data.TensorDataset(test_api_seq_data, test_fre_input_data, test_candidate_api_data,
                                      test_correct_api_data, test_alpha_data)
        test_data_loader = Data.DataLoader(dataset=test_set, batch_size=self.conf['batch_size'],
                                           shuffle=True, drop_last=True, num_workers=1)

        self.log('loading data finished...')
        acc_1 = acc_2 = acc_3 = acc_4 = total_num = 0
        acc_1_with_dis = acc_2_with_dis = acc_3_with_dis = acc_4_with_dis = total_num_with_dis = 0
        for api_seq_data, fre_input_data, candidate_api_data, correct_api_data, alpha_data in test_data_loader:
            # 指定训练模式
            self.model.train(mode=False)
            api_len = util.get_api_len(api_seq_data)

            if torch.cuda.is_available():
                api_seq_data = api_seq_data.cuda()
                fre_input_data = fre_input_data.cuda()
                candidate_api_data = candidate_api_data.cuda()
                correct_api_data = correct_api_data.cuda()
                alpha_data = alpha_data.cuda()

            score = self.model(api_seq_data, fre_input_data, candidate_api_data)

            acc_1_batch, acc_2_batch, acc_3_batch, acc_4_batch, total_num_batch, acc_1_with_dis_batch, acc_2_with_dis_batch, acc_3_with_dis_batch, acc_4_with_dis_batch,total_num_with_dis_batch = util.evaluate(score.cpu().data.numpy(), correct_api_data.cpu().data.numpy(), api_len)
            acc_1 += acc_1_batch
            acc_2 += acc_2_batch
            acc_3 += acc_3_batch
            acc_4 += acc_4_batch
            total_num += total_num_batch

            acc_1_with_dis += acc_1_with_dis_batch
            acc_2_with_dis += acc_2_with_dis_batch
            acc_3_with_dis += acc_3_with_dis_batch
            acc_4_with_dis += acc_4_with_dis_batch

            total_num_with_dis += total_num_with_dis_batch
        self.log('test acc_1=%.5f,acc_2=%.5f,acc_3=%.5f,acc_4=%.5f' % (
            acc_1 / total_num,
            acc_2 / total_num,
            acc_3 / total_num,
            acc_4 / total_num
        ))
        self.log('test acc_1_with_dis=%.5f,acc_2_with_dis=%.5f,acc_3_with_dis=%.5f,acc_4_with_dis=%.5f' % (
            acc_1_with_dis / total_num_with_dis,
            acc_2_with_dis / total_num_with_dis,
            acc_3_with_dis / total_num_with_dis,
            acc_4_with_dis / total_num_with_dis
        ))

    def valid_model(self, valid_data_loader, top_acc_3, epoch):
        self.log('valid...')
        acc_1 = acc_2 = acc_3 = acc_4 = total_num = 0
        acc_1_with_dis = acc_2_with_dis = acc_3_with_dis = acc_4_with_dis = total_num_with_dis = 0

        for api_seq_data, fre_input_data, candidate_api_data, correct_api_data, alpha_data in valid_data_loader:
            # 指定训练模式
            self.model.train(mode=False)
            api_len = util.get_api_len(api_seq_data)

            if torch.cuda.is_available():
                api_seq_data = api_seq_data.cuda()
                fre_input_data = fre_input_data.cuda()
                candidate_api_data = candidate_api_data.cuda()
                correct_api_data = correct_api_data.cuda()
                alpha_data = alpha_data.cuda()

            score = self.model(api_seq_data, fre_input_data, candidate_api_data)

            acc_1_batch, acc_2_batch, acc_3_batch, acc_4_batch, total_num_batch, acc_1_with_dis_batch, acc_2_with_dis_batch, acc_3_with_dis_batch, acc_4_with_dis_batch, total_num_with_dis_batch = util.evaluate(
                score.cpu().data.numpy(), correct_api_data.cpu().data.numpy(), api_len)
            acc_1 += acc_1_batch
            acc_2 += acc_2_batch
            acc_3 += acc_3_batch
            acc_4 += acc_4_batch
            total_num += total_num_batch

            acc_1_with_dis += acc_1_with_dis_batch
            acc_2_with_dis += acc_2_with_dis_batch
            acc_3_with_dis += acc_3_with_dis_batch
            acc_4_with_dis += acc_4_with_dis_batch

            total_num_with_dis += total_num_with_dis_batch

        # 如果准确值比当前最高值高
        if top_acc_3 < acc_3_with_dis / total_num_with_dis:
            top_acc_3 = acc_3_with_dis / total_num_with_dis
            self.log('valid acc_1=%.5f,acc_2=%.5f,acc_3=%.5f,acc_4=%.5f' % (
                acc_1 / total_num,
                acc_2 / total_num,
                acc_3 / total_num,
                acc_4 / total_num
            ))
            self.log('valid acc_1_with_dis=%.5f,acc_2_with_dis=%.5f,acc_3_with_dis=%.5f,acc_4_with_dis=%.5f' % (
                acc_1_with_dis / total_num_with_dis,
                acc_2_with_dis / total_num_with_dis,
                acc_3_with_dis / total_num_with_dis,
                acc_4_with_dis / total_num_with_dis
            ))
            self.save_model(self.model, epoch)
            self.log('save model..')
        else:
            self.log('current model is not higher than acc_3=%.5f' % top_acc_3)

        return top_acc_3

if __name__ == '__main__':
    conf = config.get_config()

    need_different_vector = False
    need_alpha_loss = False
    need_attention = -1
    model_save_path = ''
    # 第一个参数表示是否使用频率做attention
    # 利用类共现矩阵做attention
    if int(sys.argv[1]) == 1:
        print('use freq attention')
        need_attention = 1
        model_save_path += 'freq_attention_'
    # 自己学习attention
    if int(sys.argv[1]) == 2:
        print('use self learn attention')
        need_attention = 2
        model_save_path += 'self_attention_'
    # 第二个参数表示是否使用不对称的向量表示
    if int(sys.argv[2]) == 1:
        print('use different vector')
        need_different_vector = True
        model_save_path += 'different_vec_'
    center_vec_path = sys.argv[3]
    context_vec_path = sys.argv[4]
    # 第三个参数表示是否使用alpha来改变不同长度的损失
    if int(sys.argv[5]) == 1:
        print('use alpha to change loss weight')
        need_alpha_loss = True
        model_save_path += 'alpha_loss_'
    if model_save_path != '':
        model_save_path += '/'

    recommender = ApiRecommender(conf, model_path=model_save_path, need_loss_weight=need_alpha_loss)
    if conf['reload'] == -1:
        recommender.model = api_model.ApiRecommendationModel(conf,
                                                             center_vec_path=center_vec_path,
                                                             context_vec_path=context_vec_path,
                                                             need_attention=need_attention)
    else:
        recommender.load_model(conf['reload'], recommender.model, conf['reload'])
    recommender.model = recommender.model.cuda() if torch.cuda.is_available() else recommender.model
    recommender.optimizer = optim.Adam(recommender.model.parameters(), lr=conf['lr'])
    recommender.train()


