import math
import torch
from torch import nn
from torch.nn import functional as F

import torchtext
from torchtext.data.utils import get_tokenizer
from pyitcast.transformer import TransformerModel

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),  # 一个分割器对象,按照文本为基础英文进行分割
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
print(test_txt.examples[0].text[:10])  # use examples[0].text to get text object (测试集文本前十个)

# use train_txt to build a vocab object
TEXT.build_vocab(train_txt)

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, batchsize):
    """
    :param data: the text data we got previously(train, val, test, txt)
    :param batchsize: batch_size
    :return:
    """
    # map each word to a certain integer
    data = TEXT.numericalize([data.examples[0].text])

    num_batches = data.size(0) // batchsize

    # narrow:对数据进行切割, 0切行, 1切列, 行切是闭开区间，列切是闭闭区间
    data = data.narrow(0, 0, num_batches * batchsize)
    data = data.view(batchsize, -1).t().contiguous()  # change the shape of data
    return data.to(device)


batch_size = 20
eval_batch_size = 10

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# set maximum length of a sentence
max_len = 35


def get_batch(source, i):
    """
    :param source: train/val/test data
    :param i: index of batch
    :return:
    """
    seq_len = min(max_len, len(source) - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len].view(-1)
    return data, target

# source = test_data
# i = 2
# x, y = get_batch(source, i)
# print(train_txt[:10])
# print(x)
# print(y)
# print(x.shape)
# print(y.shape)

# set hyperparameters
# get the number of unrepeated words
num_tokens = len(TEXT.vocab.stoi)

embed_size = 200

ffw = 200

num_layers = 2

num_heads = 2

dropout = 0.2

model = TransformerModel(num_tokens, embed_size, num_heads, ffw, num_layers, dropout).to(device)

loss_function = nn.CrossEntropyLoss()

lr = 3

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, 0.95)

import time


def train():
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch, i in enumerate(range(0, train_data.size(0) - 1, max_len)):
        data, targets = get_batch(train_data, i)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output.view(-1, num_tokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

        log_interval = 200
        if batch % log_interval == 0 and batch > 8:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} |'
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // max_len, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, max_len):
            data, targets = get_batch(data_source, i)

            output = eval_model(data)
            output_flat = output.view(-1, num_tokens)  # 每个单词都有一个概率,所以是num_tokens个
            total_loss += loss_function(output_flat, targets).item()
    return total_loss


best_val_loss = float("inf")
epochs = 15
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 90)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} |'
          .format(epoch, (time.time() - epoch_start_time),
                  val_loss))
    print('-' * 90)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    scheduler.step()

test_loss = evaluate(best_model, test_data)
print('-' * 90)
print('| end of training | test loss {:5.2f} '.format(test_loss))
print('-' * 90)
