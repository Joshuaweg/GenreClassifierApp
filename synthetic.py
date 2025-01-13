#coding: cp1252
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
import sys
import string
from GenreClassifier.GenreClassifierNN import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
printable = set(string.printable)
print("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv('data.csv', encoding='ISO-8859-1')
stop_words = set(stopwords.words('english'))
X,y = data["summary"], data["genre"]
print(X[0])
print(y[0])
target_category=["fantasy","science","crime","history","horror","thriller","psychology","romance","sports","travel"]
label_pipeline = lambda x: target_category.index(x)
#before resampling we need to convert the text data in summary to numerical data by creating embeddings with each summary using pytorch
#while tokenizing we need to reomve non-ascii characters and stopwords, personal pronouns, and other words that do not contribute to the meaning of the text
tokenizer = get_tokenizer("basic_english")
data_pipe = dp.iter.IterableWrapper(['data.csv'])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter=',', as_tuple=True)
vocab = build_vocab_from_iterator(yield_tokens(data_pipe,target_category), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
with open('text_vocab10.pickle', 'wb') as handle:
    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
sys.exit()
X = [torch.LongTensor(vocab(tokenizer(i))) for i in X]
X = [torch.tensor([i for i in i if i not in stop_words], dtype=torch.long) for i in X]
print(X[0])
X = [torch.nn.functional.pad(i, (0, 471 - len(i))) for i in X]
print(X[0])
X_text = torch.cat(X)
y = [label_pipeline(i) for i in y]
print(y[0])
#need to prevent ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (4660,) + inhomogeneous part.
# error is caused by the fact that the summaries are of different lengths, so we need to pad them to the same length
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
X_resampled=torch.LongTensor(X_resampled)
y_resampled=torch.LongTensor(y_resampled)
x_offset =torch.LongTensor([0])
x_offset =[x_offset]+[len(i) for i in X_resampled]
x_offset = torch.tensor(x_offset[:-1]).cumsum(dim=0)
print(x_offset)
idx,c = np.unique(y_resampled,return_counts=True)
sns.barplot(x=idx,y=c)
plt.show()
X_resampled= list(zip(X_resampled,x_offset))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_SIZE=32
HIDDEN_SIZE=64
CHECKPOINT="GenreClassifier/models/NNGenreClassifier10.pth"
test_label ='Fantasy'
test_line =('He always knew he was different.First there were the dreams.Then the deaths began. When Matt Freeman gets into trouble with the police, hes sent to be fostered in Yorkshire. Its not long before he senses theres something wrong with his guardian; with the whole village. Then Matt learns about the Old Ones and begins to understand just how he is different. But no one will believe him; no one can help. There is no proof. There is no logic. There is just the Gate.')
EPOCHS = 200
LR = .0002
BATCH_SIZE=1
USE_PRETRAIN=False

model,criterion,optimizer = build_model(vocab,EMBEDDING_SIZE,HIDDEN_SIZE,target_category,LR)
scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=10, verbose=True)

model.to(device)
max_val = 0
if not(USE_PRETRAIN):
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        cum_loss = 0
        accu_train = 0
        for i in range(0, len(X_train)):
            model.train()
            data = X_train[i][0]
            data=data.to(device)
            target = torch.LongTensor([y_train[i]])
            target=target.to(device)
            optimizer.zero_grad()
            offset=torch.LongTensor([0])
            offset=offset.to(device)
            output = model(data,offset)
            loss = criterion(output, target)
            cum_loss+= loss.item()
            accu_train += (output.argmax(1) == target).sum()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | "
                    "average loss {:8.3f} | train accuracy: {:8.3f}| target: {} | prediction: {}".format(
                        epoch, i, len(X_train), cum_loss/100, accu_train/100,target_category[target],target_category[output.argmax(1)]
                    )
                )
                cum_loss = 0
                accu_train = 0
        #validation
        accu_val = 0
        valid_loss =0
        with torch.no_grad():
            model.eval()
            accu_val = 0
            for i in range(0, len(X_test)):
                data = X_test[i][0]
                data=data.to(device)
                target = torch.LongTensor([y_test[i]])
                target=target.to(device)
                offset = torch.LongTensor([0])
                offset=offset.to(device)
                output = model(data,offset)
                loss = criterion(output, target)
                valid_loss+=loss.item()
                accu_val += (output.argmax(1) == target).sum()
            if accu_val > max_val:
                max_val = accu_val
                torch.save(model.state_dict(),CHECKPOINT)
            scheduler.step(valid_loss)
        print("-" * 59)
        print(
         "| end of epoch {:3d} | time: {:5.2f}s | "
         "valid accuracy {:8.3f} ".format(
         epoch, time.time() - epoch_start_time, accu_val/len(X_test)
         )
         )
        print("-" * 59)
eb_model=TextClassificationModel(len(vocab),EMBEDDING_SIZE,HIDDEN_SIZE,10)
eb_model.load_state_dict(load_model("GenreClassifier/models/NNGenreClassifier10.pth"))
eb_model.eval()
