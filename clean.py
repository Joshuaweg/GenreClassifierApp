import pandas as pd
genres = ["Fiction","Nonfiction","Fantasy"]
data = pd.read_csv("data\data_lang.csv")
print(data.Genres1.unique())
data = data[data.Genres1.isin(genres)]
print(data[["Title","Description","Genres1"]].head())
data.to_csv("cleaned_genredata.csv", encoding="utf-8")