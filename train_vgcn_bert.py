from model_vgcn_bert import VGCN_Bert
import os
import time
import numpy as np
import pickle as pkl
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam  # , warmup_linear


from utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# STEP 1: CONFIGURATIONS FOR TRAINING

parser = argparse.ArgumentParser()
# ds = dataset, sw = stopwords, lr = learning rate, l2 = L2 regularization
parser.add_argument('--ds', type=str, default='pheme')
parser.add_argument('--load', type=int, default='0')
parser.add_argument('--sw', type=int, default='1')
parser.add_argument('--dim', type=int, default='16')
parser.add_argument('--lr', type=float, default=1e-5)  # 2e-5
parser.add_argument('--l2', type=float, default=0.01)  # 0.001
parser.add_argument('--model', type=str, default='VGCN_BERT')
args = parser.parse_args()


config_dataset = args.ds
config_load_model_from_checkpoint = True if args.load == 1 else False
config_use_stopwors = True if args.sw == 1 else False
config_gcn_embedding_dim = args.dim
config_learning_rate0 = args.lr
config_l2_decay = args.l2
config_model_type = args.model

config_warmup_proportion = 0.1
config_vocab_adj = 'all'  # pmi / tf / all
config_adj_npmi_threshold = 0.2
config_adj_tf_threshold = 0
config_loss_criterion = 'cross_entropy'

MAX_SEQ_LENGTH = 200 + config_gcn_embedding_dim
total_train_epochs = 9
batch_size = 16  # 12
gradient_accumulation_steps = 1
bert_model_scale = 'bert-base-uncased'
do_lower_case = True
perform_metrics_str = ['weighted avg', 'f1-score']
do_softmax_before_mse = True

data_dir = './prepared_data/'
output_dir = './model_output/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


model_file_save = config_model_type + str(config_gcn_embedding_dim) + '_model_' + \
    config_dataset + '_' + config_loss_criterion + '_' + \
    "sw" + str(int(config_use_stopwors)) + '.pt'

print('\n')
print('----------STEP 1: CONFIGURATIONS FOR TRAINING--------')
print('Dataset: ', config_dataset)
print('Will Delete Stop Words: ', config_use_stopwors)
print('Vocab GCN Hidden Dim: vocab_size -> 128 -> ' + str(config_gcn_embedding_dim))
print('Learning Rate0: ', config_learning_rate0)
print('Weight Decay: ', config_l2_decay)
print('Loss Criterion: ', config_loss_criterion)
print('Will Perform Softmax before MSE: ', do_softmax_before_mse)
print('Vocab Adjcent: ', config_vocab_adj)
print('MAX_SEQ_LENGTH: ', MAX_SEQ_LENGTH)
print('Perform Metrics: ', perform_metrics_str)
print('Saved Model File Name: ', model_file_save)


# STEP 2.1: PREPARE DATASET & LOAD VOCABULARY ADJACENT MATRIX

print('\n')
print('----------STEP 2: PREPARE DATASET & LOAD VOCABULARY ADJACENT MATRIX----------')
print(' Load and seperate', config_dataset, ' dataset, with vocabulary graph adjacent matrix')

objects = []
names = ['index_label', 'train_label', 'train_label_prob', 'test_label',
         'test_label_prob', 'clean_docs', 'vocab_adj_tf', 'vocab_adj_pmi', 'vocab_map']

for i in range(len(names)):
    datafile = data_dir + "/data_%s.%s" % (config_dataset, names[i])
    with open(datafile, 'rb') as f:
        objects.append(pkl.load(f, encoding='latin1'))

index_labels_list, train_label, train_label_prob, test_label, test_label_prob, shuffled_clean_docs, gcn_vocab_adj_tf, gcn_vocab_adj_pmi, gcn_vocab_map = tuple(
    objects)

label2idx = index_labels_list[0]
idx2label = index_labels_list[1]

all_labels = np.hstack((train_label, test_label))
all_labels_prob = np.vstack((train_label_prob, test_label_prob))

examples = []
for i, text in enumerate(shuffled_clean_docs):
    example = InputExample(i, text.strip(), confidence=all_labels_prob[i], label=all_labels[i])
    examples.append(example)

num_classes = len(label2idx)
gcn_vocab_size = len(gcn_vocab_map)
train_size = len(train_label)
test_size = len(test_label)

indexs = np.arange(0, len(examples))
train_examples = [examples[i] for i in indexs[:train_size]]
test_examples = [examples[i] for i in indexs[train_size:train_size + test_size]]

if config_adj_tf_threshold > 0:
    gcn_vocab_adj_tf.data *= (gcn_vocab_adj_tf.data > config_adj_tf_threshold)
    gcn_vocab_adj_tf.eliminate_zeros()
if config_adj_npmi_threshold > 0:
    gcn_vocab_adj_pmi.data *= (gcn_vocab_adj_pmi.data > config_adj_npmi_threshold)
    gcn_vocab_adj_pmi.eliminate_zeros()

if config_vocab_adj == 'pmi':
    gcn_vocab_adj_list = [gcn_vocab_adj_pmi]
elif config_vocab_adj == 'tf':
    gcn_vocab_adj_list = [gcn_vocab_adj_tf]
elif config_vocab_adj == 'all':
    gcn_vocab_adj_list = [gcn_vocab_adj_tf, gcn_vocab_adj_pmi]

norm_gcn_vocab_adj_list = []
for i in range(len(gcn_vocab_adj_list)):
    adj = gcn_vocab_adj_list[i]

    print('Zero ratio for vocab adj %dth: %.8f' %
          (i, 100 * (1 - adj.count_nonzero() / (adj.shape[0] * adj.shape[1]))))

    adj = normalize_adj(adj)
    norm_gcn_vocab_adj_list.append(sparse_scipy2torch(adj.tocoo()).to(device))

gcn_adj_list = norm_gcn_vocab_adj_list


train_classes_num, train_classes_weight = get_class_count_and_weight(train_label, len(label2idx))
loss_weight = torch.tensor(train_classes_weight).to(device)
loss_weight = torch.tensor(loss_weight, dtype=torch.float32).to(device)

tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)


# STEP 2.2: PREPARE PYTORCH DATALOADER

def get_pytorch_dataloader(examples, tokenizer, batch_size):
    dataset = CorpusDataset(examples, tokenizer, gcn_vocab_map, MAX_SEQ_LENGTH, config_gcn_embedding_dim)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.pad)


train_dataloader = get_pytorch_dataloader(train_examples, tokenizer, batch_size)
test_dataloader = get_pytorch_dataloader(test_examples, tokenizer, batch_size)

total_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * total_train_epochs)

print('Train Classes Count: ', train_classes_num)
print('Batch size: ', batch_size)
print('Num steps: ', total_train_steps)
print('Number of Examples for Training: ', len(train_examples))
print('Number of Examples for Training After Dataloader: ', len(train_dataloader) * batch_size)
print('Number of Examples for Validate: ', len(test_examples))




# STEP 3: START TRAINING VGCN_BERT MODEL

def predict(model, examples, tokenizer, batch_size):
    dataloader = get_pytorch_dataloader(examples, tokenizer, batch_size)
    predict_out = []
    confidence_out = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, _, _, gcn_swop_eye = batch

            _, score_out = model(gcn_adj_list, gcn_swop_eye,
                                 input_ids, segment_ids, input_mask)
            if config_loss_criterion == 'mse' and do_softmax_before_mse:
                score_out = torch.nn.functional.softmax(score_out, dim=-1)
            predict_out.extend(score_out.max(1)[1].tolist())
            confidence_out.extend(score_out.max(1)[0].tolist())

    return np.array(predict_out).reshape(-1), np.array(confidence_out).reshape(-1)


def evaluate(model, gcn_adj_list, predict_dataloader, epoch_th, dataset_name):
    model.eval()
    predict_out = []
    all_label_ids = []
    ev_loss = 0
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch
            _, logits = model(gcn_adj_list, gcn_swop_eye,input_ids,  segment_ids, input_mask)

            if config_loss_criterion == 'mse':
                if do_softmax_before_mse:
                    logits = F.softmax(logits, -1)
                loss = F.mse_loss(logits, y_prob)
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), label_ids)
            ev_loss += loss.item()

            _, predicted = torch.max(logits, -1)
            predict_out.extend(predicted.tolist())
            all_label_ids.extend(label_ids.tolist())
            eval_accuracy = predicted.eq(label_ids).sum().item()
            total += len(label_ids)
            correct += eval_accuracy

        f1_metrics = f1_score(np.array(all_label_ids).reshape(-1), np.array(predict_out).reshape(-1), average='weighted')
        print("Report:\n"+classification_report(np.array(all_label_ids).reshape(-1), np.array(predict_out).reshape(-1), digits=4))

    ev_acc = correct/total
    end = time.time()
    print('Epoch : %d, %s: %.3f Acc : %.3f on %s, Spend:%.3f minutes for evaluation'
          % (epoch_th, ' '.join(perform_metrics_str), 100 * f1_metrics, 100. * ev_acc, dataset_name, (end - start) / 60.0))
    print('*' * 50)
    return ev_loss, ev_acc, f1_metrics



print('\n')
print('----------STEP 3: START TRAINING VGCN_BERT MODEL----------')

if config_load_model_from_checkpoint and os.path.exists(os.path.join(output_dir, model_file_save)):
    checkpoint = torch.load(os.path.join(output_dir, model_file_save), map_location='cpu')
    if 'step' in checkpoint:
        prev_save_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
    else:
        prev_save_step = -1
        start_epoch = checkpoint['epoch'] + 1

    valid_acc_prev = checkpoint['valid_acc']
    perform_metrics_prev = checkpoint['perform_metrics']
    model = VGCN_Bert.from_pretrained(bert_model_scale, state_dict=checkpoint['model_state'], gcn_adj_dim=gcn_vocab_size, 
        gcn_adj_num=len(gcn_adj_list), config_gcn_embedding_dim=config_gcn_embedding_dim, num_labels=len(label2idx))

    pretrained_dict = checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {
        k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)

    print('Loaded the pretrain model:', model_file_save, ', epoch:', checkpoint['epoch'], 'step:', prev_save_step, 'valid acc:',
          checkpoint['valid_acc'], ' '.join(perform_metrics_str)+'_valid:', checkpoint['perform_metrics'])

else:
    start_epoch = 0
    valid_acc_prev = 0
    perform_metrics_prev = 0
    model = VGCN_Bert.from_pretrained(bert_model_scale, gcn_adj_dim=gcn_vocab_size, gcn_adj_num=len(
        gcn_adj_list), gcn_embedding_dim=config_gcn_embedding_dim, num_labels=len(label2idx))
    prev_save_step = -1

model.to(device)

optimizer = BertAdam(model.parameters(), lr=config_learning_rate0,
                     warmup=config_warmup_proportion, t_total=total_train_steps, weight_decay=config_l2_decay)

train_start = time.time()
global_step_th = int(len(train_examples) / batch_size /
                     gradient_accumulation_steps * start_epoch)

all_loss_list = {'train': [], 'test': []}
all_f1_list = {'train': [], 'test': []}
for epoch in range(start_epoch, total_train_epochs):
    train_loss = 0
    model.train()
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        if prev_save_step > -1:
            if step <= prev_save_step:
                continue
        if prev_save_step > -1:
            prev_save_step = -1
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, y_prob, label_ids, gcn_swop_eye = batch

        _, logits = model(gcn_adj_list, gcn_swop_eye,
                          input_ids, segment_ids, input_mask)

        if config_loss_criterion == 'mse':
            if do_softmax_before_mse:
                logits = F.softmax(logits, -1)
            loss = F.mse_loss(logits, y_prob)
        else:
            if loss_weight is None:
                loss = F.cross_entropy(logits, label_ids)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, num_classes), label_ids, loss_weight)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()

        train_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
        if step % 40 == 0:
            print("Epoch:{}-{}/{}, Train {} Loss: {}, Cumulated Time: {}m ".format(epoch, step,
                  len(train_dataloader), config_loss_criterion, loss.item(), (time.time() - train_start)/60.0))

    print('*' * 50)
    test_loss, test_acc, perform_metrics = evaluate(model, gcn_adj_list, test_dataloader, epoch, 'Test_set')
    all_loss_list['train'].append(train_loss)
    all_loss_list['test'].append(test_loss)
    all_f1_list['test'].append(perform_metrics)
    print("Epoch:{} Completed, Total Train Loss:{}, Test Loss:{}, Spend {}m ".format(
        epoch, train_loss, test_loss, (time.time() - train_start) / 60.0))

    if perform_metrics > perform_metrics_prev:
        to_save = {'epoch': epoch, 'model_state': model.state_dict(),
                   'valid_acc': test_acc, 'lower_case': do_lower_case,
                   'perform_metrics': perform_metrics}
        torch.save(to_save, os.path.join(output_dir, model_file_save))

        perform_metrics_prev = perform_metrics

        valid_f1_best_epoch = epoch

print('\n')
print('Optimization Finished! Total Spend Time:', (time.time() - train_start)/60.0)
print('Test Weighted F1: %.3f at %d Epoch.' % (100*perform_metrics_prev, valid_f1_best_epoch))
