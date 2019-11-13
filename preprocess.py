import csv
from nltk.tokenize import word_tokenize

wset=set()
with open('shortjokes.csv') as csv_file, open('humor.txt','w') as writer:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count1 = 0
    fre=0
    for row in csv_reader:
        if line_count1 == 0:
            line_count1 += 1
        else:
            sent = word_tokenize(row[1].strip())
            assert(len(sent)>0)
            if len(sent) > 100:
                fre += 1
            for w in sent:
                assert not(' ' in w)
                wset.add(w)

            writer.write(' '.join(sent)+'\n')
            line_count1 += 1
        if line_count1%10000==0:
            print(line_count1)

    print('>100:',fre)

    print('humor_sents:',line_count1-1)

non_humor=0
with open('news.2016.en.shuffled') as reader,open('non_humor.txt','w') as writer:
    line_count = 0
    for line in reader:
        sent = word_tokenize(line.strip())
        flag=1
        for word in sent:
            if not word in wset:
                flag=0
                break
        line_count+=1
        if line_count%10000==0:
            print(line_count,non_humor)
        if len(sent)>100:
            continue
        if flag==0:
            continue
        non_humor+=1
        writer.write(' '.join(sent)+'\n')
        if non_humor>10*(line_count1-1):
            break
    print('non_humor_sents:', non_humor)