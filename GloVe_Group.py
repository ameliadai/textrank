import numpy as np
import pandas as pd
import nltk
import re
import warnings
from rouge import Rouge
warnings.filterwarnings("ignore")
# nltk.download('punkt')  # download punkt
# nltk.download('stopwords')  # download stopwords

# input dataset
df = pd.read_csv("New_covid-19.csv", index_col=False)
df=df[~(df['abstract'].isnull())]  # delete empty rows
df=df[~(df['text_body'].isnull())]  # delete empty rows
df.drop([66], inplace=True)
df.index = range(len(df))



# train word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()




def rank(sentences, index):
    # clean the texts (including removing punctuation, numbers, Special characters and unifying into lowercase letters)
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]

    # remove the stopwors in sentences
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    # Eigenvectors of sentences
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # Similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    # Use cosine similarity to initialize the similarity matrix.
    from sklearn.metrics.pairwise import cosine_similarity
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    # Apply PageRank Algorithm
    import networkx as nx
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    # extract abstracts
    # generate abstract according to the top N
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    top_sentences = ''
    num = round(len(scores) * 0.09)
    for i in range(num):
        top_sentences += " " + ranked_sentences[i][1]

    def rouge(a,b):
        rouge = Rouge()  
        rouge_score = rouge.get_scores(a,b, avg=True)
        r1 = rouge_score["rouge-1"]["f"]
        return r1

    a = ''
    for s in df[index:(index + 10)]['abstract']:
        a += " " + s
    print(rouge(a,top_sentences))
    return(rouge(a,top_sentences))
    

# split text into seperated sentences
from nltk.tokenize import sent_tokenize
# sentences = []
score = 0
index = 0

for i in range(10):
    index += 1
    print(index, ":")
    sentences = []
    for s in df[i:(i+10)]['text_body']:
        sentences.append(sent_tokenize(s))  # sentences:[[,...,],[,...,],[,...,]]
    sentences = [y for x in sentences for y in x]  # sentences:[,...,]
    score += rank(sentences, index-1)
score = score/index
print("total score = ", score)
