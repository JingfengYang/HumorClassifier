from utilsCNN import readData,build_vocab,batchIter,idData
import torch
import torch.nn as nn
import time
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORD_EMBEDDING_DIM = 100
FILTER_SIZES = [1, 2, 3, 5]
PRINT_EVERY=10000
EVALUATE_EVERY_EPOCH=1
NUM_FILTERS=36
DROUPOUT_RATE=0.1
BATCH_SIZE=32
INIT_LEARNING_RATE=0.0
EPOCH=50
WARMUP=400
THERESHHOLD=0.5

class Encoder(nn.Module):
    def __init__(self,word_size,word_dim,num_filters,filter_sizes,dropout_p=0.0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(word_size,word_dim)
        #self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, word_dim)) for K in filter_sizes])
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self,input,input_len,train=True):
        x = self.embedding(input)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        if train:
            x = self.dropout(x)
        logit = self.fc1(x).squeeze(1)
        return logit

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

class SimpleLossCompute:

    def __init__(self,  opt=None):
        self.opt = opt

    def __call__(self, x, y, norm,train=True):
        loss = F.binary_cross_entropy(torch.sigmoid(x),y.float())
        if train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
        else:
            if self.opt is not None:
                self.opt.optimizer.zero_grad()

        return loss.item() * norm


def run_epoch(data_iter, model, loss_compute,train=True):

    start = time.time()
    total_sents = 0
    total_loss = 0
    sents = 0

    humorTrueTotal = 0
    humorPredTotal = 0
    humorCorrectTotal = 0
    for i, (sent_batch,tag_batch) in enumerate(data_iter):##
        out = model(sent_batch[0], sent_batch[1],train=train)
        loss = loss_compute(out, tag_batch, sent_batch[0].size()[0],train=train)
        total_loss += loss
        total_sents += sent_batch[0].size()[0]
        sents += sent_batch[0].size()[0]

        if i %  PRINT_EVERY== 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Sents per Sec: %f Loss: %f " %
                    (i, sents / elapsed , loss / sent_batch[0].size()[0] ))
            start = time.time()
            sents = 0
        if not train:
            results = (out>THERESHHOLD).long().detach().tolist()
            y = tag_batch.detach().tolist()

            for pred,gold in zip(results,y):
                if pred==1 and gold==1:
                    humorCorrectTotal += 1
                    humorTrueTotal += 1
                    humorPredTotal += 1
                else:
                    if pred==1:
                        humorPredTotal += 1
                    if gold==1:
                        humorTrueTotal += 1
    f=0.0
    if not train:
        if not humorPredTotal:
            humorPredTotal = 1
            humorCorrectTotal=1
        if not humorCorrectTotal:
            humorCorrectTotal=1
        p = humorCorrectTotal / humorPredTotal
        r = humorCorrectTotal / humorTrueTotal
        f=2 * p * r / (p + r)
        print("Humor classification precision: %f recall: %f fscore: %f" % (p, r, f))

    return total_loss / total_sents,f

def run(epoch,model,batch_size,trainData,valData,testData,word_vocab):
    valResult=[]
    testResult=[]
    model_opt = NoamOpt(WORD_EMBEDDING_DIM, 1, WARMUP,
                        torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE,betas=(0.9, 0.98), eps=1e-9))
    lc=SimpleLossCompute(model_opt)
    for i in range(epoch):
        model.train()
        run_epoch(batchIter(trainData,batch_size,word_vocab), model, lc,train=True)
        model.eval()
        print('Evaluation_val: epoch: %d' % (i))
        loss,f=run_epoch(batchIter(valData, batch_size, word_vocab), model, lc, train=False)
        print('Loss:', loss)
        valResult.append(f)
        print('Evaluation_test: epoch: %d' % (i))
        loss,f=run_epoch(batchIter(testData, batch_size, word_vocab), model, lc, train=False)
        print('Loss:', loss)
        testResult.append(f)
    valBest=max(valResult)
    print('ValBest epoch:', [i for i, j in enumerate(valResult) if j == valBest])
    testBest = max(testResult)
    print('TestBest epoch:', [i for i, j in enumerate(testResult) if j == testBest])

trainSents=readData('data/train_text.txt','data/train_label.txt')
valSents=readData('data/val_text.txt','data/val_label.txt')
testSents=readData('data/test_text.txt','data/test_label.txt')
vocDic=build_vocab(trainSents)
trainData=idData(trainSents,vocDic)
valData=idData(valSents,vocDic)
testData=idData(testSents,vocDic)
encoder=Encoder(len(vocDic),WORD_EMBEDDING_DIM,NUM_FILTERS,FILTER_SIZES,DROUPOUT_RATE).to(device)
run(EPOCH,encoder,BATCH_SIZE,trainData,valData,testData,vocDic)