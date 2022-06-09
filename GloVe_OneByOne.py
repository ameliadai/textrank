import numpy as np
import pandas as pd
import nltk
import re
import warnings
from rouge import Rouge
import math
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
    # We first obtain the vectors of all the constituent words of each sentence (obtained from the GloVe word vector file, with each vector size of 100 elements), 
    # then take the average value of these vectors and obtain the combination vector of this sentence as the feature vector of this sentence.
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v) 

    # Similarity matrix
    # Define an n-by-n zero matrix and fill it with cosine similarity between sentences, where n is the total number of sentences.
    sim_mat = np.zeros([len(sentences), len(sentences)])
    # Use cosine similarity to initialize the similarity matrix.
    from sklearn.metrics.pairwise import cosine_similarity
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), 
                sentence_vectors[j].reshape(1,100))[0,0]

    # Apply PageRank Algorithm
    # The similarity matrix Sim_mat is transformed into a graph structure. The nodes of this graph are sentences, 
    # and edges are represented by similarity scores between sentences. On this graph, we will apply the PageRank algorithm to get the sentence rankings.
    import networkx as nx
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    # extract abstracts
    # generate abstract according to the top N
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    top_sentences = ''
    num = math.ceil(len(scores) * 0.09)
    for i in range(num):
        top_sentences += " " + ranked_sentences[i][1]

    def rouge(a,b):
        rouge = Rouge()  
        rouge_score = rouge.get_scores(a,b, avg=True)
        r1 = rouge_score["rouge-1"]["f"]
        return r1

    a = df.loc[index, 'abstract']
    # print('original abstract: ', a)
    print('Predict abstract:', top_sentences)
    print(rouge(a,top_sentences))
    return(rouge(a,top_sentences))
    
    
# split text into seperated sentences
from nltk.tokenize import sent_tokenize
score = 0
index = 0
for s in df[:1]['text_body']:
    num = index
    index += 1
    print((index), ":")
    sentences = sent_tokenize(s)
    score += rank(sentences, num)
score = score/index
print("total score = ", score)
