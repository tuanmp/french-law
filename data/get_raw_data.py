import os

import pandas as pd
from datasets import load_dataset

DATASET_NAMES = [
    # "harvard-lil/cold-french-law",
    "VinceGx33/legi-french-law-2025"
]

COLUMNS = [
    # [
    #     "article_identifier",
    #     "article_num",
    #     "article_etat",
    #     "article_date_debut",
    #     "texte_nature",
    #     "texte_titre",
    #     "article_contenu_text"
    # ],
    [
        "article_identifier",
        "article_num",
        'article_etat',
        'date_debut',
        'texte_nature',
        'texte_titre',
        "article_contenu_text",
    ]
]

def main():

    data = pd.DataFrame()

    for name, cols in zip(DATASET_NAMES, COLUMNS):
        ds = load_dataset(name, split="train", cache_dir=os.environ["PSCRATCH"]).select_columns(cols)
        df = ds.to_pandas()
        # only select laws that are en vigueur
        df = df[df['article_etat'] == "VIGUEUR"].drop("article_etat", axis=1)
        df.rename({"date_debut": "article_date_debut"}, errors="ignore", inplace=True, axis=1)
        data = pd.concat([data, df], ignore_index=True)

    data = data.dropna(axis=0).drop_duplicates(['article_identifier', "article_num"])
    data.to_csv("legi-french.csv", index=False)



if __name__=="__main__":
    main()
