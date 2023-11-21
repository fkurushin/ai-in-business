import pandas as pd
import numpy as np
from string import punctuation

from tqdm import tqdm

import nltk 
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split



tqdm.pandas()
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))

def lemmatize_words(tokens):
    pos_tagged_text = nltk.pos_tag(tokens)
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def text_processing(text : str) -> str:
    # lowercase
    text = text.lower()
    
    # remove punctuation
    text = text.translate(str.maketrans('', '', punctuation))

    # remove stopwords
    tokens = [word for word in str(text).split() if word not in STOPWORDS]

    # lemmatization
    text = lemmatize_words(tokens)

    return text


if __name__ == "__main__":
    df = pd.read_csv("data/ecommerceDataset.csv", names=["class", "text"])

    print(f"\nshape: {df.shape}\n")

    df["class"] = df["class"].astype(str)
    df["text"] = df["text"].astype(str)

    df.loc[:,"len"] = df.text.apply(lambda x: len(x))
    df.loc[:,"n"] = df.text.apply(lambda x: len(x.split()))



    min_n = np.percentile(df["n"], 2)
    max_n = np.percentile(df["n"], 97)

    print(f"percrentiles 2%: {min_n}, 97%: {max_n}\n")

    df = df[(df["n"] > min_n) & (df["n"] < max_n)]


    lemmatizer = WordNetLemmatizer()
    wordnet_map = { 
        "N" : wordnet.NOUN, 
        "V" : wordnet.VERB, 
        "J" : wordnet.ADJ,
        "R" : wordnet.ADV
        }

    df["text"] = df["text"].progress_apply(lambda x: text_processing(x))

    df = df[["text", "class"]]

    df = df.drop_duplicates()
    df.reset_index(inplace=True, drop=True)

    print(f"\nEnd of processing shape: {df.shape}\n")

    train, test = train_test_split(
        df, test_size=0.33, shuffle=True, random_state=42, stratify=df["class"])

    with open('data/train.txt', 'w') as f:
        for each_text, each_label in zip(train['text'], test['class']):
            f.writelines(f'__label__{each_label} {each_text}\n')
            
    with open('data/test.txt', 'w') as f:
        for each_text, each_label in zip(train['text'], test['class']):
            f.writelines(f'__label__{each_label.replace(" ", "_")} {each_text}\n')

    print("write files for fasttext training in data/train.txt and data/test.txt\n")