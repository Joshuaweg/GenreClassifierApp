#coding: cp1252
"""
Program to classify book genres with a feedforward neural network
Takes in data from good reads dataset
"""
import re
import json
import itertools
import random
import time
import torch
import shap
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import sys
import torchdata.datapipes as dp
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import shutil
import torch.nn.functional as F
import pickle
import spacy
from captum.attr._core.lrp import PropagationRule
from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase, ShapleyValueSampling,LRP, LayerLRP
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from collections import Counter
from IPython.core.display import HTML, display
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
tokenizer = get_tokenizer("basic_english")
target_category=["fantasy","fiction","nonfiction"]
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: target_category.index(x)
EPOCHS = 20
LR = .0002
BATCH_SIZE=200
#device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device =  torch.device('cpu')
#custom LRP methods
def create_random_masks(token_length, num_samples=100, mask_prob=0.5):
    """
    Generate random masks for a given token length.
    
    Args:
        token_length (int): The length of the token sequence.
        num_samples (int): The number of random masks to generate.
        mask_prob (float): Probability of masking each token.
    
    Returns:
        List[List[bool]]: A list of random binary masks.
    """
    masks = []
    #print(range(num_samples))
    for _ in range(num_samples):
        mask = [random.random() > mask_prob for _ in range(token_length)]
        masks.append(mask)
    return masks

class ShapTextAttribution:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def explain(self, text, num_samples=100,offsets=None, mask_prob=0.5):
        # Step 1: Tokenize the text
        tokens = text
        print(tokens)
        # Step 2: Get the original prediction
        original_prediction = self.model_predict(tokens,offsets)[0]
        
        # Step 3: Calculate attributions using random masks
        attributions = self.calculate_attributions(text, self.tokenizer, original_prediction, num_samples, mask_prob)
        
        # Map attributions back to tokens for interpretation
        #token_attributions = list(zip(tokens, attributions.tolist()))
        
        return tokens,attributions

    def model_predict(self, masked_texts,cls=0, offsets=None):
        # Predict using the model
        predictions = []
        for mt in masked_texts:
            mt_tensor = torch.tensor(mt, dtype=torch.long).unsqueeze(0)  # Assuming single input, batch size = 1
            #print("Mask tensor: ",mt_tensor)
            preds= self.model.forward_with_embedding(mt_tensor,offsets)
            #print("Predictions: ",preds[0])
            predictions.append(preds[0].item())  # Assuming single output
            #print("Predictions: ",len(predictions))
        return torch.tensor(predictions)

    def calculate_attributions(self, text, tokenizer, original_prediction, num_samples=100, mask_prob=0.5):
        tokens = text
        attributions = torch.zeros(len(tokens))
        
        # Create random masks
        masks = create_random_masks(len(tokens), num_samples=num_samples, mask_prob=mask_prob)
        for i, mask in enumerate(masks):
            masked_text = [token if mask[j] else 0 for j, token in enumerate(tokens)]
            #print("Masked text: ",masked_text)
            prediction = self.model_predict(masked_text,offsets=None,cls=original_prediction.argmax().item())
            #print("Original prediction: ",original_prediction)
            #print("Prediction: ",prediction)
            attributions += (original_prediction - prediction)
        
        attributions /= num_samples  # Normalize the attributions
        return attributions
def lrp_linear(layer, R, X, eps=1e-9):
    """
    Performs LRP for a linear layer.
    
    Args:
        layer (nn.Linear): The linear layer.
        R (torch.Tensor): The relevance scores from the next layer.
        X (torch.Tensor): The input to the linear layer.
        eps (float): Small constant to avoid division by zero.
    
    Returns:
        torch.Tensor: The relevance scores for the current layer.
    """
    # Forward pass contributions
    Z = layer(X)
    
    # Backward pass relevance
    Z_eps = Z + eps * (Z >= 0).float() - eps * (Z < 0).float()
    S = R / Z_eps
    C = torch.autograd.grad(outputs=Z, inputs=X, grad_outputs=S, retain_graph=True)[0]
    R_j = X * C

    return R_j

def lrp_embedding(embedding_layer, R, X, offset, eps=1e-12):
    """
    LRP for an embedding layer.
    
    Args:
        embedding_layer (nn.Embedding): The embedding layer.
        R (torch.Tensor): The relevance scores from the next layer.
        X (torch.Tensor): The input indices to the embedding layer.
        eps (float): Small constant to avoid division by zero.
    
    Returns:
        torch.Tensor: The relevance scores for the embedding layer.
    """
    #print("R: ",R.shape)
    emb = embedding_layer(X)
    #print(offset.item())
    #print("emb: ",emb.shape)
    Z = emb + eps * (emb >= 0).float() - eps * (emb < 0).float()
    #print("Z: ",Z.shape)
    S = R / Z
    C = torch.autograd.grad(outputs=Z, inputs=emb, grad_outputs=S, retain_graph=True)[0]
    return C.sum(dim=-1)
# Neural Network Model
class TextClassificationModel(nn.Module):
    def __init__(self,vocab_size, embed_dim,hidden, num_class):
        super(TextClassificationModel, self).__init__()
        self.embeddingbag = nn.EmbeddingBag(vocab_size, embed_dim,padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input = nn.Linear(embed_dim, hidden)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(hidden,hidden)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(hidden,num_class)
        self.init_weights()
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.input.weight.data.uniform_(-initrange,initrange)
        self.input.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange,initrange)
        self.output.bias.data.zero_()
    def forward(self, text, offsets=None):
        # Assuming 'text' is already batched appropriately
        embedded = self.embeddingbag(text,offsets[:-1])  # Shape: (batch_size, seq_length, embed_dim)
        #print("embedding bag: ",embedded.size())
        # Averaging embeddings across the sequence length, simulating EmbeddingBag's default behavior
        l1 = self.input(embedded)
        l1 = self.act1(l1)
        l2 = self.l2(l1)
        l2 = self.act2(l2)
        return self.output(l2)
    def partial_forward(self, embeddings):
        l1 = self.input(embeddings)
        l1 = self.act1(l1)
        l2 = self.l2(l1)
        l2 = self.act2(l2)
        return self.output(l2)
    def forward_with_embedding(self, text, offsets=None):
        # Use nn.Embedding for LRP
        emb = self.embedding(text)  # Shape: (batch_size, seq_len, embed_dim)
        #print("embedding size: ",emb.size())
        if offsets is None:
            pooled_emb = emb.mean(dim=0)
        else:
            pooled_emb = torch.stack([emb[offsets[i]:offsets[i+1]].mean(dim=0) for i in range(len(offsets) - 1)])
        #print("pooled embedding size: ",pooled_emb.size())

        l1 = self.input(pooled_emb)
        l1 = self.act1(l1)
        l2 = self.l2(l1)
        l2 = self.act2(l2)
        out = self.output(l2)
        return out
    def get_embeddings(self,text,offsets):
        return self.embedding(text, offsets)
    def get_embeddingbag(self,text,offsets):
        return self.embeddingbag(text,offsets)
    def lrp(self, text, offsets=None, eps=1e-9):
        emb = self.embedding(text)  # Shape: (batch_size, seq_len, embed_dim)
        if offsets is None:
            pooled_emb = emb.mean(dim=0)
        else:
            pooled_emb = torch.stack([emb[offsets[i]:offsets[i+1]].mean(dim=0) for i in range(len(offsets) - 1)])
            #pooled_emb = torch.cat([pooled_emb, emb[offsets[-1]:].mean(dim=0, keepdim=True)])
        #print("Pooled emb: ",pooled_emb.size())
        l1 = self.input(pooled_emb)
        act1 = self.act1(l1)
        l2 = self.l2(act1)
        act2 = self.act2(l2)
        output = self.output(l1)
        
        # Start LRP from the output layer
        R_output = output.clone().detach()  # Initialize relevance at the output
        R_act2 = lrp_linear(self.output, R_output, act2, eps)
        R_l2 = lrp_linear(self.act2, R_act2, l2, eps)
        R_act1 = lrp_linear(self.l2, R_l2, act1, eps)
        R_l1 = lrp_linear(self.act1, R_act1, l1, eps)
        R_input = lrp_linear(self.input, R_l1, pooled_emb, eps)
        R_text = lrp_embedding(self.embedding, R_input, text, offsets, eps)
        return R_text
class PartialModel(nn.Module):
    def __init__(self, original_model):
        super(PartialModel, self).__init__()
        self.linear = original_model.input.requires_grad_()
        self.relu = original_model.act1.requires_grad_()
        self.output = original_model.output.requires_grad_()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.output(x).requires_grad_()
        return x.argmax(1)
#Integrated Gradients functions
def generate_baseline(text_length, embed_dim,model):
    padding_embedding = model.embedding(torch.tensor([0]))  # Padding token
    baseline =padding_embedding.repeat(text_length, 1)
    print("baseline: ",baseline.size())
    return baseline
def interpolate_input(baseline, input_emb, steps):
    # Create a set of interpolated inputs
    interpolated_inputs = [(baseline + (float(i) / steps) * (input_emb - baseline)) for i in range(steps + 1)]
    return torch.stack(interpolated_inputs)
def compute_gradients(model,embeddings, target_class):
    embeddings=embeddings.requires_grad_()
    #print("embeddings gradient required: ",embeddings.requires_grad)
    output = model.partial_forward(embeddings)
    target_output = output[:, target_class]
    gradients = torch.autograd.grad(
        outputs=target_output,
        inputs=embeddings,
        grad_outputs=torch.ones_like(target_output),
        create_graph=True,
        retain_graph=True
    )[0]
    return gradients
def integrate_gradients(gradients, inputs, baseline, steps):
    # Average the gradients over all steps
    avg_gradients = torch.mean(gradients, dim=0)
    # Compute the integrated gradients
    integrated_gradients = (inputs - baseline) * avg_gradients
    return integrated_gradients.sum(dim=-1)  # Sum over the embedding dimensions

def integrated_gradients(model, text, tokenizer, target_class, steps=50):
    # Tokenize the text
    token_ids = text
    embed_dim = model.embedding.embedding_dim
    text_length = len(token_ids)
    
    # Generate baseline (e.g., zero embeddings)
    baseline = generate_baseline(text_length, embed_dim,model)
    
    # Get the input embeddings
    input_emb = model.embedding(token_ids)
    #print(input_emb.size())
    # Interpolate between baseline and input embeddings
    interpolated_inputs = interpolate_input(baseline, input_emb, steps)
    #print(interpolated_inputs.size())
    # Compute gradients for each interpolated input
    gradients = []
    for i in range(steps + 1):
        grads = compute_gradients(model, interpolated_inputs[i], target_class)
        #print("gradients: ",grads.shape)
        gradients.append(grads)
    gradients = torch.stack(gradients)
    #print("gradients: ",gradients.size())
    # Compute integrated gradients
    integrated_gradients = integrate_gradients(gradients, input_emb, baseline, steps)
    
    return integrated_gradients

def collate_batch(batch,text_pipeline):
    label_list, text_list, offsets, titles, authors, text_raw = [], [], [0],[],[],[]
    for _,a,_text, _label in batch:
        titles.append(_)
        authors.append(a)
        text_raw.append(_text)
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        #print(processed_text.size(0))
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list=torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets).cumsum(dim=0)
    #print(offsets)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device),offsets.to(device),titles,authors,text_raw
cats = ["fiction","nonfiction","fantasy"]
def yield_tokens(data_iter,cats):
    print("yielding tokens")
    nlp = spacy.load('en_core_web_lg')
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    for _,a,p,text in data_iter:
        #print(p)
        if text in cats:
            p = re.sub(r'[^a-z ]', '',p.lower())
            #print(p)
            tokens  = nlp(p)
            tokens = " ".join([token.text for token in tokens if (token.text not in stopwords and token.text != "" and token.pos_ != "PUNCT" and token.pos_ != "PROPN")])
            #print("cleaned\n",tokens)
            yield tokenizer(tokens)
def removeAttribution(row):
    """
    Function to keep the first two elements in a tuple
    """
    return row[:4]
def train(dataloader,model,criterion,optimizer):
    model.train()
    total_acc, total_count =0, 0
    log_interval = 5
    start_time=time.time()
    for idx,(label, text, offsets, titles, authors, raw) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        #print(predicted_label)
        #print(label)
        loss = criterion(predicted_label, label)
        #print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        predicted_label = model.forward_with_embedding(text, offsets)
        #print(predicted_label)
        #print(label)
        loss = criterion(predicted_label, label)
        #print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time()-start_time
            print(
                "| epoch {:3d}| {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc/total_count
                 )
                )
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader,model,criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, ( label, text, offsets, titles,authors, raw) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count
#Interpretable functions



def preprocessing(data_path,batch_size=1):
    tokenizer = get_tokenizer("basic_english")
    data_pipe = dp.iter.IterableWrapper([data_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter=',', as_tuple=True)
    data_pipe = data_pipe.map(removeAttribution)
    trn,vld = data_pipe.random_split(total_length=2280,weights={"train":0.8,"valid":0.2},seed=314159)
    for sample in data_pipe:
        if sample[3] not in target_category:
            print(sample[3])
            target_category.append(sample[3])
    vocab = build_vocab_from_iterator(yield_tokens(data_pipe,target_category),specials= ['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    with open('GenreClassifier\\text_vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    train_dataloader = DataLoader(
        trn, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_batch(batch, text_pipeline)
    )
    test_dataloader = DataLoader(
        vld, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: collate_batch(batch, text_pipeline)
    )
    return train_dataloader,test_dataloader,vocab
def build_model(vocab,embedding_size,hidden_size,classes_num,lr):
    #add scheduler and early stopping
    num_class = len(classes_num)
    vocab_size = len(vocab)
    emsize = embedding_size
    hidden = hidden_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextClassificationModel(vocab_size,emsize,hidden,num_class).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    total_accu = None
    return model,criterion,optimizer

def feature_attributions(model,text,offsets,label):
    def forward_func(text,offsets):
        '''Remove the batch dimension for the embedding-bag model'''
        return model(text.squeeze(0),offsets)
    def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
        original_emb = model.embeddingbag(original_inp, None)
        perturbed_emb = model.embeddingbag(perturbed_inp, None)
        distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
        return torch.exp(-1 * (distance ** 2) / 2)
    def bernoulli_perturb(text, **kwargs):
        probs = torch.ones_like(text)* 0.5
        return torch.bernoulli(probs).long()
    def interp_to_input(interp_sample, original_input, **kwargs):
        return original_input[interp_sample.bool()].view(original_input.size(0),-1)
    lasso_lime_base = LimeBase(
    forward_func,
    interpretable_model=SkLearnLasso(alpha=0.08),
    similarity_func=exp_embedding_cosine_distance,
    perturb_func=bernoulli_perturb,
    perturb_interpretable_space=True,
    from_interp_rep_transform=interp_to_input,
    to_interp_rep_transform=None
    )
    print(text)
    print(label)
    print(offsets)
    attrs = lasso_lime_base.attribute(
    text.unsqueeze(0),
    target=label,
    additional_forward_args=(offsets,),
    n_samples = 20000,
    show_progress =True
    ).squeeze(0)
    print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())
    return attrs

def feature_attributions_shap(model,text,offset,label):
    svs = ShapleyValueSampling(model)
    attr = svs.attribute(text, additional_forward_args=(offset,), target=0, n_samples=200)
    return attr
def feature_attributions_lrp(model,text,offset,label):
   output = model.forward_with_embedding(text,offset)
   attr = model.lrp(text,offset)
   return attr


#print('Attribution range:', attrs.min().item(),'to', attrs.max().item())
def load_model(model_path):
    return torch.load(model_path, map_location=torch.device('cpu'))
def get_label(pred):
    return pred.argmax(1).item()
def get_cat(idx):
    return target_category[idx]
def show_text_attr(model,attrs,txt,label,text_pipeline,filename="test.html"):
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    atr_min = attrs.min().item()
    atr_max = attrs.max().item()
    neg_norm = lambda x: (x - atr_min) / (-atr_min+1e-9)
    pos_norm = lambda x: (x) / (atr_max+1e-9)
    #alpha values need to be scaled from 0-1
    #when x is 0, the alpha values should be 0
    #when x is less than zero, the alpha values should be scaled from 0 to 1 based on the minimum value
    #when x is greater than zero, the alpha values should be scaled from 0 to 1 based on the maximum value
    alpha = lambda x:  0 if x == 0 else (pos_norm(x) if x > 0 else neg_norm(x))
    direction = lambda x: "neg" if x < 0 else ("pos" if x > 0 else "nothing")
    token_marks = [
        f'<mark class={direction(attr)} style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
        for token, attr in zip(tokenizer(txt), attrs.tolist())
    ]
    label,text,offset,titles,authors,raw = collate_batch([([0],[0],txt,label)],text_pipeline)
    label=label.to(device)
    text=text.to(device)
    offset=offset.to(device)
    pred = model(text,offset)
    print(target_category[pred.argmax(1).item()])
    pred_c=target_category[pred.argmax(1).item()]
    with open(filename,"w+") as fl:
        fl.write('<p>' + ' '.join(token_marks) + '</p>')
        fl.close()
    display(HTML(' '.join(token_marks)))
    return ' '.join(token_marks)
def show_text_attr_lrp(model,attrs,txt,label,text_pipeline):
    #return list of boolean values indicating if the token is zero as 0  or 1
    label,text,offset,titles,authors, raw = collate_batch([([0],[0],txt,label)],text_pipeline)
    #print(tokenizer(txt))   
    tok_is_zero = [1 if x == 0 else 0 for x in text.squeeze(0).tolist()]
    #print(tok_is_zero)
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    atr_min = attrs.min().item()
    atr_max = attrs.max().item()
    neg_norm = lambda x: (x - atr_min) / (-atr_min+1e-9)
    pos_norm = lambda x: (x) / (atr_max+1e-9)
    #alpha values need to be scaled from 0-1
    #when x is 0, the alpha values should be 0
    #when x is less than zero, the alpha values should be scaled from 0 to 1 based on the minimum value
    #when x is greater than zero, the alpha values should be scaled from 0 to 1 based on the maximum value
    alpha = lambda x:  0 if x == 0 else (pos_norm(x) if x > 0 else neg_norm(x))
    #alpha = lambda x: x
    direction = lambda x: "neg" if x < 0 else ("pos" if x > 0 else "nothing")
    token_marks = []
    for i in range(len(attrs)):
        if tok_is_zero[i] == 1:
            token_marks.append(f'<mark class={"nothing"} style="background-color:rgba(255,255,255,0)">{tokenizer(txt)[i]}</mark>')
        else:
            token_marks.append(f'<mark class={direction(attrs[i])} style="background-color:rgba({rgb(attrs[i])},{alpha(attrs[i])})">{tokenizer(txt)[i]}</mark>')
    label=label.to(device)
    text=text.to(device)
    offset=offset.to(device)
    pred = model.forward_with_embedding(text,offset)
    print(target_category[pred.argmax(1).item()])
    pred_c=target_category[pred.argmax(1).item()]
    with open("test_lrp.html","w+") as fl:
        fl.write('<p>' + ' '.join(token_marks) + '</p>')
        fl.close()
    display(HTML(' '.join(token_marks)))
    return ' '.join(token_marks)
 
if __name__ == "__main__":
    #device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device =  torch.device('cpu')
    EMBEDDING_SIZE=16
    HIDDEN_SIZE=48
    CHECKPOINT="GenreClassifier/models/NNGenreClassifier_temp2.pth"
    test_label ='fantasy'
    test_line =('He always knew he was different.First there were the dreams.Then the deaths began. When Matt Freeman gets into trouble with the police, hes sent to be fostered in Yorkshire. Its not long before he senses theres something wrong with his guardian; with the whole village. Then Matt learns about the Old Ones and begins to understand just how he is different. But no one will believe him; no one can help. There is no proof. There is no logic. There is just the Gate.')
    EPOCHS = 600
    LR = .0002
    BATCH_SIZE=20
    USE_PRETRAIN=False
    vocab = None
    t_train,t_test,vocab = preprocessing("data\\genre_data.csv",BATCH_SIZE)
    #print(vocab)
    with open('GenreClassifier\\text_vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    print(len(vocab))
    test_labels,test_text,test_offsets,test_title,test_author, test_raw = collate_batch([([0],[0],test_line,test_label)],text_pipeline)
    print(test_offsets)
    test_text = test_text.to(device)
    test_offsets = test_offsets.to(device)
    print(target_category)
    model,criterion,optimizer = build_model(vocab,EMBEDDING_SIZE,HIDDEN_SIZE,target_category,LR)
   # apply_embeddingbag_rule(model)
    if not(USE_PRETRAIN):
        max_val = 0
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            train(t_train,model,criterion,optimizer)
            accu_val = evaluate(t_test,model,criterion)
            if accu_val > max_val:
                max_val = accu_val
                torch.save(model.state_dict(),CHECKPOINT)
            print("-" * 59)
            print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
            )
            print("-" * 59)
            if max_val >= 0.87:
                break
    json_file = "embeddings.json"
    embeddings_full = []
    entry_id = 1
    print("Generating embeddings from dataset")
    for idx,(label, text, offsets, titles, authors,raw) in enumerate(t_train):
        embeddings=model.get_embeddingbag(text,offsets).tolist()
        b=0
        for lbl,ttl,auth,r in zip(label.tolist(),titles,authors,raw):
            entry ={}
            entry["genreId"] = lbl
            entry["genre"]= target_category[lbl]
            entry["tokens"] = text[offsets[b]:offsets[b+1]].tolist()
            entry["wordCount"] = len(entry["tokens"])
            entry["title"] = ttl
            entry["author"] = auth
            entry["text"] = r
            entry["embedding"] = embeddings[b]
            entry["id"] = entry_id
            entry["group"] = "train"
            embeddings_full.append(entry)
            entry_id += 1
            b += 1
    print(len(embeddings_full))
    for idx,(label, text, offsets, titles, authors,raw) in enumerate(t_test):
        embeddings=model.get_embeddingbag(text,offsets).tolist()
        b=0
        for lbl,ttl,auth,r in zip(label.tolist(),titles,authors,raw):
            entry ={}
            entry["genreId"] = lbl
            entry["genre"]= target_category[lbl]
            entry["tokens"] = text[offsets[b]:offsets[b+1]].tolist()
            entry["wordCount"] = len(entry["tokens"])
            entry["title"] = ttl
            entry["author"] = auth
            entry["text"] = r
            entry["embedding"] = embeddings[b]
            entry["id"] = entry_id
            entry["group"] = "test"
            embeddings_full.append(entry)
            entry_id += 1
            b += 1
        #torch.save(model.state_dict(),CHECKPOINT)
    #print(len(vocab))
    print("full embeddings: ",len(embeddings_full))
    print("Saving embeddings to json file")
    with open(json_file,"w+") as fl:
        json.dump(embeddings_full,fl, indent=4)
    eb_model=TextClassificationModel(len(vocab),EMBEDDING_SIZE,HIDDEN_SIZE,3).to(device)
    eb_model.load_state_dict(load_model(CHECKPOINT))
    #apply_embeddingbag_rule(eb_model)
    eb_model.eval()
    #print(test_labels)
    
    probs = F.softmax(eb_model(test_text, test_offsets), dim=1).squeeze(0)
    print(probs)
    print('Prediction probability:', round(probs[test_labels[0]].item(), 4))
    attrs=feature_attributions(eb_model,test_text,test_offsets,test_labels)
    show_text_attr(eb_model,attrs,test_line,test_label,text_pipeline,filename="lime_attributions.html")
    attrs=feature_attributions_lrp(eb_model,test_text,test_offsets,test_labels)
    print(attrs.shape)
    show_text_attr(eb_model,attrs,test_line,test_label,text_pipeline,filename="lrp_attributions.html")
    print("SHAP")
    shap = ShapTextAttribution(eb_model.to(device),tokenizer)
    toks,attrs = shap.explain(test_text,offsets=test_offsets,num_samples=1000)
    print(attrs)
    show_text_attr(eb_model,attrs,test_line,test_label,text_pipeline,filename="shap_attributions.html")
    attrs=integrated_gradients(eb_model.to(device),test_text,tokenizer,0,steps=1000)
    print(attrs)
    show_text_attr(eb_model,attrs,test_line,test_label,text_pipeline,filename="Inte_grad_attributions.html")