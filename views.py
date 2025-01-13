"""
Routes and views for the flask application.
"""
import tracemalloc
import io
from datetime import datetime
from flask import render_template,request,send_from_directory, send_file
from GenreClassifier import app
from GenreClassifier.GenreClassifierNN import *
from plots import *
from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from IPython.core.display import HTML, display
from collections import Counter
import pickle
import gc
import urllib.parse
@app.route('/list_farthest_neighbors')
def list_farthest_neighbors(input_features, output, neighbors=20, metric="euclidean"):
    df = extract_embeddings("embeddings.json")
    df_train = df[df.group == "train"]
    func_neighbors = NearestNeighbors(n_neighbors=len(df_train)+1, metric=metric)
    tokens = input_features["tokens"]
    embedding = input_features["embedding"][0]
    text = input_features["text"]
    ttl = input_features["title"]
    label = "fantasy" if output == 0 else "fiction" if output == 1 else "nonfiction"
    
    entry = pd.DataFrame([[5000, label, output, embedding, "input", tokens, text, ttl]],
                         columns=["id", "genre", "genreId", "embedding", "group", "tokens", "text", "title"])
    
    df_train = pd.concat([df[df.group == "train"], entry], ignore_index=True)
    embeddings = np.stack(df_train["embedding"])
    func_neighbors.fit(embeddings)
    
    # Get all neighbors, then select the farthest ones
    distances, indices = func_neighbors.kneighbors([embedding])
    
    # Sort distances in descending order and take the farthest ones
    farthest_indices = np.argsort(distances[0])[-neighbors:]
    
    sample_groups = []
    sample_genres = []
    sample_titles = []
    sample_passage = []
    
    for i in farthest_indices:
        sample_groups.append(df_train.loc[indices[0][i], "group"])
        sample_genres.append(df_train.loc[indices[0][i], "genre"])
        sample_titles.append(df_train.loc[indices[0][i], "title"])
        sample_passage.append(df_train.loc[indices[0][i], "text"])
    sample_groups.append("input")
    sample_genres.append(label)
    sample_titles.append(ttl)
    sample_passage.append(text)
    
    sample_dataframe = pd.DataFrame({
        "genre": sample_genres,
        "group": sample_groups,
        "title": sample_titles,
        "text": sample_passage
    })
    #move last row to first row
    sample_dataframe = pd.concat([sample_dataframe.iloc[[-1]],sample_dataframe.iloc[:-1]],ignore_index=True)
    
    table_html = sample_dataframe.to_html(classes="table table-striped table-hover table-bordered table-responsive", index=False)

    html = f"""<div class="container"> <h2>Farthest Neighbors for Inputted Title</h2> {table_html} </div>"""

    return HTML(html)
@app.route('/list_neighbors')
def list_neighbors(input_features,output, neighbors=20, metric="euclidean"):
    func_neighbors = NearestNeighbors(n_neighbors=neighbors,metric=metric)
    df = extract_embeddings("embeddings.json")
    tokens = input_features["tokens"]
    embedding = input_features["embedding"][0]
    text = input_features["text"]
    ttl = input_features["title"]
    label = "fantasy" if output == 0 else "fiction" if output == 1 else "nonfiction"
    entry = pd.DataFrame([[5000,label,output,embedding,"input",tokens,text,ttl]],columns=["id","genre","genreId","embedding","group","tokens","text","title"])
    #print("Entry: ",entry["embedding"])
    df_train= pd.concat([df[df.group=="train"],entry],ignore_index=True)
    #print(df_train["embedding"])
    embeddings = np.stack(df_train["embedding"])
    func_neighbors.fit(embeddings)
    distances, indices = func_neighbors.kneighbors([embedding])
    #print("Nearest neighbors for input features: ")
    sample_groups = []
    sample_genres = []
    sample_titles = []
    sample_passage =[]
    for i in range(neighbors):
        sample_groups.append(df_train.loc[indices[0][i],"group"])
        sample_genres.append(df_train.loc[indices[0][i],"genre"])
        sample_titles.append(df_train.loc[indices[0][i],"title"])
        sample_passage.append(df_train.loc[indices[0][i],"text"])
        sample_dataframe = pd.DataFrame({
        "genre":sample_genres,
        "group":sample_groups,
        "title":sample_titles,
        "text":sample_passage
    })
    table_html = sample_dataframe.to_html(classes="table table-striped table-hover table-bordered table-responsive",index=False)

    html =f"""<div class="container"> <h2>Nearest Neighbors for Inputted Title</h2> {table_html} </div>"""

    return HTML(html)

@app.route('/plot_nearest_neighbors',methods=['POST'])
def plot_nearest_neighbors():
    """Function to find the nearest neighbors of a given input feature"""
    print("Embedding")
    data = request.json
    input_features = data['input_features']
    output = data['output']
    output = int(output)
    print("JSONified input features: ")
    input_features = json.loads(input_features)
    neighbors=20
    metric="euclidean"
    print(input_features)
    print(output)
    func_neighbors = NearestNeighbors(n_neighbors=neighbors,metric=metric)
    df = extract_embeddings("embeddings.json")
    tokens = input_features["tokens"]
    embedding = input_features["embedding"][0]
    text = input_features["text"]
    ttl = input_features["title"]
    label = "fantasy" if output == 0 else "fiction" if output == 1 else "nonfiction"
    entry = pd.DataFrame([[5000,label,output,embedding,"input",tokens,text,ttl]],columns=["id","genre","genreId","embedding","group","tokens","text","title"])
    df_train= pd.concat([df[df.group=="train"],entry],ignore_index=True)
    embeddings = np.stack(df_train["embedding"])
    func_neighbors.fit(embeddings)
    distances, indices = func_neighbors.kneighbors([embedding])
    print(distances)
    print("Nearest neighbors for input features: ")
    print("generating Graph")
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

        if len(xi[0]) > 0:  # Ensure xi is not empty
            plt.scatter(tsne_results[xi, 0], tsne_results[xi, 1], c=col, marker=mark, alpha=0.8, s=size, label=group + "-" + label)
        else:
            print("Skipping empty xi")
    titles = sample_dataframe["title"].to_list()
    for i in range (len(tsne_results)):
        plt.text(tsne_results[i,0],tsne_results[i,1],i, fontsize=14, alpha=0.8)
    plt.title('2D-tSNE Projection of Nearest Neighbors of Input Features')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')
@app.route('/plot_farthest_neighbors',methods=['POST'])
def plot_farthest_neighbors():
    """Function to find the nearest neighbors of a given input feature"""
    data = request.json
    input_features = data['input_features']
    output = data['output']
    print("Embedding")
    print(request.args.get('input_features'))
    print("Output: ",output)
    output = int(output)
    print("JSONified input features: ")
    input_features = json.loads(input_features)
    neighbors=20
    metric="euclidean"
    print(input_features)
    print(output)
    df = extract_embeddings("embeddings.json")
    tokens = input_features["tokens"]
    embedding = input_features["embedding"][0]
    text = input_features["text"]
    ttl = input_features["title"]
    label = "fantasy" if output == 0 else "fiction" if output == 1 else "nonfiction"
    entry = pd.DataFrame([[5000,label,output,embedding,"input",tokens,text,ttl]],columns=["id","genre","genreId","embedding","group","tokens","text","title"])
    df_train= pd.concat([df[df.group=="train"],entry],ignore_index=True)
    embeddings = np.stack(df_train["embedding"])
    func_neighbors = NearestNeighbors(n_neighbors=len(df_train),metric=metric)
    func_neighbors.fit(embeddings)
    distances, indices = func_neighbors.kneighbors([embedding])
    farthest_indices = np.argsort(distances[0])[-neighbors:]
    sample_embeddings = []
    sample_groups = []
    sample_genres = []
    sample_titles = []

    for i in farthest_indices:
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
    sample_embeddings.append(embedding)
    sample_groups.append("input")
    sample_genres.append(label)
    sample_titles.append(ttl)
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
        print(tsne_results[xi])
        print(col)
        print(mark)
        print(size)
        print(group)
        print(label)
        if len(xi[0]) > 0:  # Ensure xi is not empty
            plt.scatter(tsne_results[xi, 0], tsne_results[xi, 1], c=col, marker=mark, alpha=0.8, s=size, label=group + "-" + label)
        else:
            print("Skipping empty xi")
    titles = sample_dataframe["title"].to_list()
    for i in range (len(tsne_results)):
        plt.text(tsne_results[i,0],tsne_results[i,1],i, fontsize=14, alpha=0.8)
    plt.title('2D-tSNE Projection of Farthest Neighbors of Input Features')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

@app.route('/embedding_plot')
def embedding_plot():
    # Example data generation (replace with your actual data)
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
        label_complete = label+"-"+group
        ix = np.where((df["genre"].to_numpy() == label) & (df["group"].to_numpy() == group))
        #print(ix)
        test_count += len(ix[0]) if group == "test" else 0
        train_count += len(ix[0]) if group == "train" else 0
        mark = "o" if group == "train" else "*"
        size = 40 if group == "train" else 200
        col = colors[label]
        embs=np.stack(df["projected_embeddings"])
        plt.scatter(embs[ix,0], embs[ix,1], c=col, label=label_complete, marker=mark, alpha=0.5,s=size)
    print("Train count: ",train_count)
    print("Test count: ",test_count)
    plt.legend()
    plt.title('2D-tSNE Projection of Genre Embeddings from Dataset')
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Analystrix-Genre Classifier',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
@app.route('/classify',methods=['POST'])
def classify():
    input_size = 16
    hidden_size = 48
    num_classes = 3
    content = None
    attrs = None
    toks = None
    method_name = None
    with open('text_vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    gc_model=TextClassificationModel(len(vocab),input_size,hidden_size,num_classes).to("cpu");
    text_pipeline = lambda x: vocab(tokenizer(x))
    print(request.form)
    gc_model.load_state_dict(load_model("GenreClassifier/models/NNGenreClassifier_temp2.pth"))
    gc_model.eval()
    exp_method = int(request.form['method'])
    print(exp_method)
    label,text,offset,titles,authors,raw = collate_batch([([0],[0],(request.form['description']),request.form['genre'])],text_pipeline)
    label = label.to("cpu")
    text = text.to("cpu")
    text_tokens=text.tolist()
    offset = offset.to("cpu")
    out = gc_model(text,offset)
    embedding = gc_model.embeddingbag(text,offset).tolist()
    probs = torch.nn.functional.softmax(out,dim=1)[0]
    print(probs)
    probs_map = dict(map(lambda x: (x[0],x[1]),zip(target_category,probs.tolist())))
    print(probs_map)
    labl = get_label(out)
    cat =get_cat(labl)
    features = {
        "text":request.form['description'],
        "title":request.form['title'],
        "genre":cat
        ,"embedding":embedding,
        "tokens":text_tokens}
    #need to convert features and cat into url parameters to be passed to the plot_nearest_neighbors function
    #stringify features
    str_features = json.dumps(features)
    str_features = urllib.parse.quote(str_features)
    #str_features needs to be url safe
    print("str_features: ")
    print(str_features)
    url_params = f"?input_features={str_features}&output={str(labl)}"
    print("URL Params: ",url_params)
    try:
        match exp_method:
            case 1:
                method_name = "LIME--Local Interpretable Model-agnostic Explanations"
                attrs = feature_attributions(gc_model,text,offset,labl)
                attrs =attrs.to("cpu")
                content=HTML(show_text_attr(gc_model,attrs,request.form['description'],cat,text_pipeline,"lime.html"))
            case 2:
                method_name = "SHAP--SHapley Additive exPlanations"
                shap = ShapTextAttribution(gc_model,tokenizer)
                toks,attrs = shap.explain(text,num_samples=100,offsets=offset,mask_prob=0.5)
                attrs =attrs.to("cpu")
                content=HTML(show_text_attr(gc_model,attrs,request.form['description'],cat,text_pipeline,"shap.html"))
            case 3:
                method_name = "Layer-wise Relevance Propagation"
                attrs = feature_attributions_lrp(gc_model,text,offset,labl)
                attrs =attrs.to("cpu")
                content=HTML(show_text_attr(gc_model,attrs,request.form['description'],cat,text_pipeline,"lrp.html"))
            case 4:
                method_name = "Integrated Gradients"
                attrs = integrated_gradients(gc_model,text,tokenizer,labl,steps=1000)
                attrs =attrs.to("cpu")
                content=HTML(show_text_attr(gc_model,attrs,request.form['description'],cat,text_pipeline,"ig.html"))
            case default:
                ValueError("Invalid Explanation Method")
    except Exception as e:
        print(e)
        content="Apologies It appears your description is not long enough. Please go Back and try again with a longer description."
    score_printout =""
    for k,v in probs_map.items():
        score = "<h4>{}--{}</h4><div class=\"progress\" style=\"height: 30px; padding:5px;\"><div class=\"progress-bar bg-success\"  role=\"progressbar\" style=\"width:{};\" aria-valuenow=\"{}\" aria-valuemin=\"0\" aria-valuemax=\"100\"></div></div>\n".format(k,'{:.2%}'.format(v),'{:.0%}'.format(v),'{:.0%}'.format(v))
        score_printout+=score
    score_printout = HTML(score_printout)
    gc_model = None
    attrs = None
    vocab = None
    text_pipeline = None
    label=None
    text=None
    out=None
    input_size = None
    hidden_size = None
    num_classes = None
    toks = None
    del gc_model
    del attrs
    del vocab
    del text_pipeline
    del label
    del text
    del out
    del input_size
    del hidden_size
    del num_classes
    del toks
    gc.collect()
    return render_template(
        'attributions.html',
        title='Attributions',
        year=datetime.now().year,
        message=content,
        pred=cat,
        method=method_name,
        probs = probs_map,
        genre = request.form['genre'],
        list_scores = score_printout,
        nn_params = url_params,
        nn_table = list_neighbors(features,labl),
        fn_table = list_farthest_neighbors(features,labl),
        input_features = str_features,
        output = labl,
    )
@app.route('/classify10',methods=['POST'])
def classify10():
    input_size = 32
    hidden_size = 64
    num_classes = 10
    content = None
    attrs = None
    with open('text_vocab10.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    gc_model=TextClassificationModel(len(vocab),input_size,hidden_size,num_classes)
    text_pipeline = lambda x: vocab(tokenizer(x))
    print(request.form)
    gc_model.load_state_dict(load_model("GenreClassifier/models/NNGenreClassifier10.pth"))
    gc_model.eval()
    gc_model
    label,text,offset = collate_batch([([0],[0],(request.form['description']),request.form['genre'])],text_pipeline)
    text=text.to("cpu")
    offset=offset.to("cpu")
    label=label.to("cpu")
    out = gc_model(text,offset)
    labl = get_label(out)
    cat =get_cat(labl)
    try:
        attrs = feature_attributions(gc_model,text,offset,labl)
        content=HTML(show_text_attr(gc_model,attrs,request.form['description'],get_cat(labl),text_pipeline))
    except Exception as e:
        content="Apologies It appears your description is not long enough. Please go Back and try again with a longer description."
    gc_model = None
    attrs = None
    vocab = None
    text_pipeline = None
    label=None
    text=None
    out=None
    labl=None
    input_size = None
    hidden_size = None
    num_classes = None
    del gc_model
    del attrs
    del vocab
    del text_pipeline
    del label
    del text
    del out
    del labl
    del input_size
    del hidden_size
    del num_classes
    gc.collect()
    return render_template(
        'attributions.html',
        title='Attributions',
        year=datetime.now().year,
        message=content,
        pred=cat,
        genre = request.form['genre']
    )
@app.route('/robots.txt')
@app.route('/sitemap.xml')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])
