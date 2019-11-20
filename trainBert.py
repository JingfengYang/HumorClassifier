from utilsBert import readData,batchIter,idData
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, WarmupLinearSchedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORD_EMBEDDING_DIM = 100
FILTER_SIZES = [1, 2, 3, 5]
PRINT_EVERY=10000
EVALUATE_EVERY_EPOCH=1
NUM_FILTERS=36
DROUPOUT_RATE=0.1
BATCH_SIZE=32
INIT_LEARNING_RATE=0.00002
EPOCH=50
WARMUP=0
THERESHHOLD=0.5
ADAM_EPSILON=1e-8

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        #self.fc = nn.Linear(768, 1)
        self.fc = nn.Linear(768, 2)

    def forward(self,input,input_len,train=True):
        encoded_layers, pooler_output = self.bert(input)
        #logit = self.fc(pooler_output).squeeze(1)
        logit = self.fc(pooler_output)
        return logit

class SimpleLossCompute:

    def __init__(self,  opt, scheduler):
        self.opt = opt
        self.scheduler = scheduler

    def __call__(self, x, y, norm,train=True):
        #loss = F.binary_cross_entropy(torch.sigmoid(x),y.float())
        loss=F.cross_entropy(x,y)
        if train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.scheduler.step()
                self.opt.zero_grad()
        else:
            if self.opt is not None:
                self.opt.zero_grad()

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
            #results = (out>THERESHHOLD).long().detach().tolist()
            y = tag_batch.detach().tolist()
            results = out.argmax(-1).detach().tolist()

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

def run(epoch,model,batch_size,trainData,valData,testData,tokenizer):
    valResult=[]
    testResult=[]
    t_total = (((len(trainData[0])-1)//batch_size)+1) * epoch
    optimizer = AdamW(model.parameters(), lr=INIT_LEARNING_RATE, eps=ADAM_EPSILON)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP, t_total=t_total)
    lc=SimpleLossCompute(optimizer,scheduler)
    for i in range(epoch):
        model.train()
        run_epoch(batchIter(trainData,batch_size,tokenizer), model, lc,train=True)
        model.eval()
        print('Evaluation_val: epoch: %d' % (i))
        loss,f=run_epoch(batchIter(valData, batch_size, tokenizer), model, lc, train=False)
        print('Loss:', loss)
        valResult.append(f)
        print('Evaluation_test: epoch: %d' % (i))
        loss,f=run_epoch(batchIter(testData, batch_size, tokenizer), model, lc, train=False)
        print('Loss:', loss)
        testResult.append(f)
    valBest=max(valResult)
    print('ValBest epoch:', [i for i, j in enumerate(valResult) if j == valBest])
    testBest = max(testResult)
    print('TestBest epoch:', [i for i, j in enumerate(testResult) if j == testBest])

trainSents=readData('data/train_text.txt','data/train_label.txt')
valSents=readData('data/val_text.txt','data/val_label.txt')
testSents=readData('data/test_text.txt','data/test_label.txt')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
trainData=idData(trainSents,tokenizer)
print('Val')
valData=idData(valSents,tokenizer)
print('Test')
testData=idData(testSents,tokenizer)
encoder=Encoder().to(device)
run(EPOCH,encoder,BATCH_SIZE,trainData,valData,testData,tokenizer)