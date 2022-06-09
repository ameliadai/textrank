import math
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from heapq import nlargest
from rouge import Rouge

# for the first time running nltk
# nltk.download('punkt')

# calculate the weight/similarity of Sentence 1 & Sentence 2
def calculate_similarity(s1,s2):
    num_of_same_word = 0
    for word in s1:
        if word in s2:
            num_of_same_word += 1
    if num_of_same_word == 0:
        return 0
    # to avoid zero denominator
    elif len(s1)==1 and len(s2)==1 and num_of_same_word == 1:
        return 1
    else:
        return num_of_same_word / (math.log(len(s1)) + math.log(len(s2)))

"""
generate the weights of the weighted graph
the input should be a list of sentences, and each sentence is decomposed into a list of words
i.e. text = [s1,s2,...,sn], where si =[w1_i,w2_i,...,wN_i]
"""
def generate_weights(text):
    n = len(text)
    # original weighted graph: nxn 0 matrix
    weights = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # avoid empty sentences
            if i!= j and len(text[i])>0 and len(text[j])>0:
                weights[i][j] = calculate_similarity(text[i] , text[j])
    return weights
 
# for vextex i
def calculate_score(weights, scores, i):
    n = len(scores)
    # damping factor d, which is usually set to 0.85
    d = 0.85
    summation = 0
    for j in range(n):
        if sum(weights[j]) != 0:
            summation += weights[j][i]*scores[j]/sum(weights[j])

    score_i = (1 - d) + d * summation
    # return the score of vertex i
    return score_i

def converge(scores, old_scores):
    cvg = True
    threshold = 0.0001
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= threshold:
            cvg = False
            break
    return cvg

def textrank(weights):
    n=len(weights)
    # initial value does not matter
    # scores = [1 for _ in range(n)]
    scores = np.ones(n)
    old_scores = np.zeros(n)
 
    # start iteration
    while not converge(scores, old_scores):
        old_scores = scores.copy()
        scores = [calculate_score(weights, scores, i) for i in range(n)]

    return scores

 
def Summarize(text, prec, stem =True, stop=True):
    # get the list of sentences of the text
    sentences = np.array(nltk.sent_tokenize(text), dtype=object)

    # decompose each sentence into a list of words
    # i.e. decomposed_sentences[i]: i-th sentence
    #      decomposed_sentences[i][j]: j-th word in the i-th sentence
    decomposed_sentences = []
    for s in sentences:
        words = nltk.word_tokenize(s)
        words = [word.lower() for word in words if word.isalpha()]
        decomposed_sentences.append(words)
    
    if stop == True:
        # remove the stopping words
        for i in range(len(decomposed_sentences)):
            for word in decomposed_sentences[i]:
                if word in stopwords:
                    decomposed_sentences[i].remove(word)
    
    if stem == True:
        # stemming the word
        stemmer = PorterStemmer() 
        for i in range(len(decomposed_sentences)):
            for j in range(len(decomposed_sentences[i])):
                decomposed_sentences[i][j] = stemmer.stem(decomposed_sentences[i][j])

    similarity_graph = generate_weights(decomposed_sentences)
    scores = textrank(similarity_graph)

    n = math.ceil(len(sentences)*prec)
    sent_index = list(map(scores.index, nlargest(n, scores)))

    print(f'top {n}: ', nlargest(n, scores))
    print(f'top {n}: ', sent_index)

    sent_index.sort()
    str = ' '
    summary =  str.join([sentences[i] for i in sent_index])
    return summary
 

# evaluation
def rouge(a,b):
    rouge = Rouge()  
    rouge_score = rouge.get_scores(a,b, avg=True)
    r1 = rouge_score["rouge-1"]["f"]
    return r1

global stopwords
stopwords = np.array([line.strip() for line in open("stop_words_english.txt", 'r').readlines()])

df=pd.read_csv('New_covid-19.csv') 
abstracts = df.abstract.drop(index=66)
texts= df.text_body.drop(index=66)
n = 101
r1 = []
for i in range(n):
    try:
        print(f'\n==========Text {i} Summary============\n')
        summary = Summarize(texts[i],0.05)
        abstract = abstracts[i]
        r = rouge(summary, abstract)
        r1.append(r)
        print(r)
    except:
        pass

print("=======final results======")
print("mean(r1): ",np.mean(r1))