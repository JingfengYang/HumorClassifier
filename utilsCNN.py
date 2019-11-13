from collections import Counter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def readData(text_file,label_file):
    data=[]
    with open(text_file) as reader1, open(label_file) as reader2:
        for line1,line2 in zip(reader1,reader2):
            data.append([line1.strip().split(),int(line2.strip())])
    return data

def build_vocab(data,min_count=0):
    word_count=Counter()
    word_voc={'PAD':0,'UNK':1}
    for sent,label in data:
        word_count.update(sent)
    for word,count in word_count.items():
        if count>0:
            word_voc[word]=len(word_voc)
    return word_voc

def getWordId(word,voc):
    if word in voc:
        return voc[word]
    else:
        return voc['UNK']

def idData(data,voc):
    data = sorted(data, key=lambda x: len(x), reverse=True)
    textData=[]
    labelData=[]
    for sent,label in data:
        idSent=[getWordId(word,voc) for word in sent]
        textData.append(idSent)
        labelData.append(label)
    return [textData,labelData]


def padding(batch,pad=0):
    lengths=[len(sent) for sent in batch]
    maxLen=max(lengths)
    outbatch = []
    for sent in batch:
        outbatch.append(sent + [pad] * (maxLen - len(sent)))
    return [torch.tensor(outbatch, dtype=torch.long, device=device),
            torch.tensor(lengths, dtype=torch.long, device=device)]

def batchIter(data,batch_size,voc):
    textData=data[0]
    labelData=data[1]
    batch_num=((len(textData)-1)//batch_size)+1
    for i in range(batch_num):
        yield(padding(textData[i*batch_size:(i+1)*batch_size],pad=voc['PAD']),
              torch.tensor(labelData[i*batch_size:(i+1)*batch_size], dtype=torch.long, device=device))

