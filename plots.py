import matplotlib
matplotlib.use('Agg')
from GenreClassifier.GenreClassifierNN import *
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors

def extract_titles(file):
     with open(file, "r") as f:
        data = json.load(f)
        title_list =[]
        for entry in data:
            title_list.append(entry["title"])
        df = pd.DataFrame(title_list, columns = ["title"])
        return df
def extract_embeddings(file):
    """Function to extract embeddings from the embeddings.json file and return a dataframe"""
    with open(file, "r") as f:
        data = json.load(f)

        embed_list =[]
        for entry in data:
            embed_list.append([entry["id"],entry["genre"],entry["genreId"],entry["embedding"],entry["group"],entry["tokens"],entry["title"],entry["text"]])
        df = pd.DataFrame(embed_list, columns = ["id","genre","genreId", "embedding","group","tokens","title","text"]) 
        return df
def create_projections(tsne=None,n_components=2):
    """Function to create projections of the embeddings"""
    df = extract_embeddings("embeddings.json")
    print("Total size: ",len(df))
    embeddings = np.stack(df["embedding"])
    #embeddings = embeddings+(1e-9)
    if(tsne is None):
        tsne = TSNE(n_components=n_components)
        tsne_results = tsne.fit_transform(embeddings)
    df["projected_embeddings"]=tsne_results.tolist()
    return df
def create_classified_scatter_plot():
    df = create_projections()
    vocab=None
    with open("text_vocab.pickle","rb") as f:
        vocab = pickle.load(f)
    print("Vocab size: ",len(vocab))
    model = TextClassificationModel(len(vocab),16,48,3)
    model.load_state_dict(torch.load("GenreClassifier\\models\\NNGenreClassifier_temp2.pth"))
    model.eval()
    print(model.parameters())
    for index, row in df[df["group"]=="test"].iterrows():
        print(index)
        out = model(torch.tensor(row["tokens"]).long(),torch.tensor([0,1000]).long())
        genre = torch.argmax(out).item()
        print(genre)
        df.loc[index,"genreId"] = genre
        df.loc[index,"genre"] = "fantasy" if genre == 0 else "fiction" if genre == 1 else "nonfiction" 
    figure = plt.figure(figsize=(10,10))
    df_group_label = df[['genre','group']].drop_duplicates()
    print(df_group_label)
    colors = {"fantasy":"red","fiction":"blue","nonfiction":"green"}
    test_count = 0
    train_count = 0
    for index, row in df_group_label.iterrows():
        label = row['genre']
        group = row['group']
        ix = np.where((df["genre"].to_numpy() == label) & (df["group"].to_numpy() == group))
        #print(ix)
        test_count += len(ix[0]) if group == "test" else 0
        train_count += len(ix[0]) if group == "train" else 0
        mark = "o" if group == "train" else "*"
        size = 40 if group == "train" else 200
        col = colors[label]
        embs=np.stack(df["projected_embeddings"])
        plt.scatter(embs[ix,0], embs[ix,1], c=col, label=label, marker=mark, alpha=0.5,s=size)
    print("Train count: ",train_count)
    print("Test count: ",test_count)
    plt.legend()
    plt.title('2D-tSNE Projection of Genre Embeddings from Dataset - test data classified by Model')
    plt.savefig("plots\\classified_scatter_plot.png")
    plt.show()
def create_scatter_plot():
    df = create_projections()
    figure = plt.figure(figsize=(10,10))
    df_group_label = df[['genre','group']].drop_duplicates()
    print(df_group_label)
    colors = {"fantasy":"red","fiction":"blue","nonfiction":"green"}
    test_count = 0
    train_count = 0
    for index, row in df_group_label.iterrows():
        label = row['genre']
        group = row['group']
        ix = np.where((df["genre"].to_numpy() == label) & (df["group"].to_numpy() == group))
        #print(ix)
        test_count += len(ix[0]) if group == "test" else 0
        train_count += len(ix[0]) if group == "train" else 0
        mark = "o" if group == "train" else "*"
        size = 40 if group == "train" else 200
        col = colors[label]
        embs=np.stack(df["projected_embeddings"])
        plt.scatter(embs[ix,0], embs[ix,1], c=col, label=label, marker=mark, alpha=0.5,s=size)
    print("Train count: ",train_count)
    print("Test count: ",test_count)
    plt.legend()
    plt.title('2D-tSNE Projection of Genre Embeddings from Dataset')
    plt.savefig("plots\\scatter_plot.png")
    plt.show()
def find_neighbors(input_features,output, neighbors=20, metric="euclidean"):
    """Function to find the nearest neighbors of a given input feature"""
    func_neighbors = NearestNeighbors(n_neighbors=neighbors,metric=metric)
    df = extract_embeddings("embeddings.json")
    tokens = input_features["tokens"]
    embedding = input_features["embedding"]
    text = input_features["text"]
    ttl = input_features["title"]
    label = "fantasy" if output == 0 else "fiction" if output == 1 else "nonfiction"
    entry = pd.DataFrame([[5000,label,output,embedding,"input",tokens,text,ttl]],columns=["id","genre","genreId","embedding","group","tokens","text","title"])
    df_train= pd.concat([df[df.group=="train"],entry],ignore_index=True)
    embeddings = np.stack(df_train["embedding"])
    func_neighbors.fit(embeddings)
    distances, indices = func_neighbors.kneighbors([embedding])
    print("Nearest neighbors for input features: ")
    sample_embeddings = []
    sample_groups = []
    sample_genres = []
    sample_titles = []

    for i in range(neighbors):
        print("Distance: ",distances[0][i])
        print("Index: ",indices[0][i])
        print("Genre: ",df_train.loc[indices[0][i],"genre"])
        print("Group: ",df_train.loc[indices[0][i],"group"])
        print("Tokens: ",df_train.loc[indices[0][i],"tokens"])
        print("Embedding: ",df_train.loc[indices[0][i],"embedding"])
        print("Title: ",df_train.loc[indices[0][i],"title"])
        print("Text: ",df_train.loc[indices[0][i],"text"])
        print("\n")
        sample_embeddings.append(df_train.loc[indices[0][i],"embedding"])
        sample_groups.append(df_train.loc[indices[0][i],"group"])
        sample_genres.append(df_train.loc[indices[0][i],"genre"])
        sample_titles.append(df_train.loc[indices[0][i],"title"])
    sample_embeddings = np.stack(sample_embeddings)
    tsne = TSNE(n_components=2,perplexity=15)
    tsne_results = tsne.fit_transform(sample_embeddings)
    figure = plt.figure(figsize=(8,8))
    colors = {"fantasy":"red","fiction":"blue","nonfiction":"green"}
    sample_dataframe = pd.DataFrame({
        "genre":sample_genres,
        "group":sample_groups,
        "title":sample_titles
    })
    df_unique = sample_dataframe[['genre','group']].drop_duplicates()
    
    print("Sample dataframe: ",sample_dataframe.head())
    for index, row in df_unique.iterrows():
        label = row['genre']
        group = row['group']  
        col = colors[label]
        mark = "o" if group == "train" else "*"
        size = 40 if group == "train" else 200
        xi = np.where((sample_dataframe["genre"].to_numpy() == row["genre"]) & (sample_dataframe["group"].to_numpy() == row["group"]))

        plt.scatter(tsne_results[xi,0],tsne_results[xi,1],c=col,marker=mark,alpha=0.8,s=size,label=group+"-"+label)
    titles = sample_dataframe["title"].to_list()
    for i in range (len(tsne_results)):
        plt.text(tsne_results[i,0],tsne_results[i,1],i, fontsize=8, alpha=0.8)
    plt.title('2D-tSNE Projection of Nearest Neighbors of Input Features')
    plt.legend()
    plt.savefig("plots\\nearest_neighbors.png")
    plt.show()
   

    

if __name__ == "__main__":
    vocab = None
    with open("text_vocab.pickle","rb") as f:
        vocab = pickle.load(f)
    itos = vocab.get_itos()
    toks = np.random.randint(0,26001,140)
    text = " ".join([itos[tok] for tok in toks])
    input_features ={  
        "tokens":toks,
        "embedding":np.random.rand(16),
        "text":text,
        "title":"Random Title"
    }
    output = 0
    find_neighbors(input_features,output,neighbors=30)

    