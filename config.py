def get_config():
    conf = {
        # 数据集存放位置
        #'data_dir': '/home/jin/tz/data/',
        'workdir': './',
        'data_dir': './data/',
        'api_seq_path': 'api_seq.txt',
        'voc_path': 'voc.txt',
        'freq_path': 'freq.txt',

        'vector_path': './data/context_embeddings.txt',
        'max_func_len': 10,
        'max_class_len': 10,
        'max_candidate_api_len': 100,

        # 模型参数
        'emb_dim': 100,
        'hidden_size': 200,
        'max_alpha': 2,

        'lr': 0.001,

        'batch_size': 200,
        'nb_epoch': 200,

        'log_every': 20,
        'save_every': 100,
        'reload': -1
    }
    return conf