import random

allSent=[]

with open('non_humor.txt') as reader:
    for line in reader:
        allSent.append((line.strip(),'0'))

with open('humor.txt') as reader:
    for line in reader:
        allSent.append((line.strip(), '1'))

random.shuffle(allSent)

length=len(allSent)

with open('data/train_text.txt','w') as writer1, open('data/train_label.txt','w') as writer2:
    for sent, label in allSent[:length // 10 * 8]:
        writer1.write(sent+'\n')
        writer2.write(label+'\n')

with open('data/val_text.txt','w') as writer1, open('data/val_label.txt','w') as writer2:
    for sent, label in allSent[length // 10 * 8:length // 10 * 9]:
        writer1.write(sent+'\n')
        writer2.write(label+'\n')

with open('data/test_text.txt','w') as writer1, open('data/test_label.txt','w') as writer2:
    for sent, label in allSent[length // 10 * 9:]:
        writer1.write(sent+'\n')
        writer2.write(label+'\n')






