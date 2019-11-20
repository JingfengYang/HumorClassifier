from collections import Counter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def readData(text_file,label_file):
    data=[]
    with open(text_file) as reader1, open(label_file) as reader2:
        for line1,line2 in zip(reader1,reader2):
            data.append([line1.strip().split(),int(line2.strip())])
    return data

def idData(data,tokenizer):
    data = sorted(data, key=lambda x: len(x), reverse=True)
    allHeads = []
    allSents = []
    allTags = []
    for sent,label in data:
        assert (len(sent) > 0)
        words = ["[CLS]"] + sent + ["[SEP]"]
        idSent = []
        idHead = []
        for w in words:
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0] * (len(tokens) - 1)
            idSent.extend(xx)
            idHead.extend(is_head)
        allSents.append(idSent)
        allTags.append(label)
        allHeads.append(idHead)
    return [allSents,allTags,allHeads]


def padding(batch,pad=0):
    lengths=[len(sent) for sent in batch]
    maxLen=max(lengths)
    outbatch = []
    for sent in batch:
        outbatch.append(sent + [pad] * (maxLen - len(sent)))
    return [torch.tensor(outbatch, dtype=torch.long, device=device),
            torch.tensor(lengths, dtype=torch.long, device=device)]

def batchIter(data,batch_size,tokenizer):
    textData=data[0]
    labelData=data[1]
    batch_num=((len(textData)-1)//batch_size)+1
    for i in range(batch_num):
        yield(padding(textData[i*batch_size:(i+1)*batch_size],pad=tokenizer._convert_token_to_id('[PAD]')),
              torch.tensor(labelData[i*batch_size:(i+1)*batch_size], dtype=torch.long, device=device))

