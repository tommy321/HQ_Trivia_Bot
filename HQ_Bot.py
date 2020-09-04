
import pickle
import mss
import time
import numpy as np
import cv2
import urllib


import datetime
from PIL import Image
import pytesseract
from googleapiclient.discovery import build
import json
from whoosh import index, qparser
from whoosh.qparser import QueryParser
from whoosh.collectors import TimeLimitCollector, TimeLimit
from functools import partial
from multiprocessing import Process
#multiprocessing.set_start_method('spawn')
from colorama import init, Fore, Style

from sklearn.preprocessing import StandardScaler

import zmq
import re
import string

import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

import statistics


init()




pytesseract.pytesseract.tesseract_cmd = 'E://Program Files (x86)//Tesseract-OCR//tesseract'
#load the negative words
negative_words = json.loads(open("settings.json").read())["negative_words"]
#load question stop words
stop_words = json.loads(open("settings.json").read())["remove_words"]
#english stop words
english_stop_words = set(stopwords.words('English'))





def load_model():
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense
    #load the model 
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    model._make_predict_function()# have to initialize before threading
    return model

def threshold_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)[1] #was 100
    #
    #imS = thresh
    #cv2.imshow('thresh',thresh)
    return thresh

def find_contours(thresh, frame):
    thresh = cv2.dilate(thresh, None, iterations=4)
    #imS = cv2.resize(thresh, (480, 290))
    #cv2.imshow('threshold',imS)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(hierarchy)
    q_area = None
    #find the contour with the correct aspect ratio
    if contours:
        for c in contours:
            #c = max(contours, key = cv2.contourArea)
            #area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            aspect = w/h
            #print(aspect)
            #the question rectange seems to have an aspect of 0.78 or so.
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(frame,str(aspect),(x,y), font, 0.5,(0,0,255),2)            
            if aspect < 1.0 and aspect > 0.74:
                #print("Here!")
                #print(aspect)
                

                q_area = frame[y:y+h, x:x+w]
                #q_rows,q_cols,q_color = q_area.shape
                #M_q_area = cv2.getRotationMatrix2D((q_cols/2,q_rows/2),-90,1)
                #rot =  cv2.warpAffine(q_area,M_q_area,(cols,rows))
                #imS = cv2.resize(rot, (480, 290))
                #cv2.imshow('q_area',imS)

                #check if q_area is mostly white
                q_thresh = threshold_image(q_area)
                #q_rows,q_cols,q_color = q_thresh.shape
                #M_q_area = cv2.getRotationMatrix2D((q_cols/2,q_rows/2),-90,1)
                #rot =  cv2.warpAffine(q_area,M_q_area,(cols,rows))
                n_white_pix = np.sum(q_thresh == 255)
                total_pix = w*h
                white_ratio = n_white_pix/total_pix
                #if not enough white pixels are there, throw out the frame
                if white_ratio <0.8:
                    q_area = None

                #rint(white_ratio)
                #mS = cv2.resize(q_thresh, (480, 290))
                #v2.imshow('q_thresh',imS)


        #cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        #rot =  cv2.warpAffine(frame,M,(cols,rows))
        #imS = cv2.resize(rot, (480, 290))
        #cv2.imshow('contours',frame)

    return q_area

def get_text(q_area, old_text = None, count = 0, previous_question = ''):
    confidence_flag = False
    gray = cv2.cvtColor(q_area, cv2.COLOR_BGR2GRAY)
    q_area = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)[1] #was 100
    q_area = q_area[70:int(rows*0.68), 0:cols] #crop the top bit out
    q_area = cv2.resize(q_area, (0,0), fx = 2, fy = 2)
    #q_area = cv2.cvtColor(q_area, cv2.COLOR_BGR2RGB)
    
    
    #cv2.imshow('q_text', q_area)
    img = Image.fromarray(q_area)
    
    img = img.rotate(-90, expand = True)
    #increase the contrast
    #enh = ImageEnhance.Contrast(img)
    #img = enh.enhance(1.5)
    #img = img.filter(ImageFilter.SHARPEN)
    
    #crop the top 15%portion of theq image off to get rid of the countdown.
    w, h = img.size
    #crop_dims = (0, int(h*0.15), w, h)
    #img = img.crop(crop_dims)
    
    #img.show()
    #s_img = img.convert('RGB')
    #s_img.save("frame%d.jpg" % count)
    
    text = pytesseract.image_to_string(img, lang = 'eng')
    #check if the question text is the same as the previous question
    q = " ".join(text.split()).split('?')[0].strip()
    prev_q = previous_question.strip('?').strip()
    #print(q)
    #print(prev_q)
    if q == prev_q:
        #it's the same as the last question
        #print("Same as Last question")
        text = ""
        confidence_flag = True
    
    if "".join(text.split()) == "".join(old_text.split()):
        confidence_flag = True
    #speed things up by always setting confidence flag to True
    #confidence_flag = True
    return(text, confidence_flag)

def clean_text(text):
    #take the raw string that comes from tesseract and return a dictionary of question and 3 answers.
    cleaned_text = text.split('\n')
    #print(cleaned_text)
    Q = ""
    A1 = ""
    A2 = ""
    A3 = ""
    idx = 0
    for line in cleaned_text:
        if idx == 0:
            #build question
            Q += " " + line
            if "?" in line:
                idx += 1
                line = line.strip()
                Q = Q.strip()
        elif idx == 1:
            if not line.isspace() and len(line)>0:
                A1 = line
                idx += 1
        elif idx == 2:
            if not line.isspace() and len(line)>0:
                A2 = line
                idx += 1
        elif idx == 3:
            if not line.isspace() and len(line)>0:
                A3 = line
                idx += 1
                
    #check that there is a question and 3 answers
    if len(Q)>0 and len(A1)>0 and len(A2)>0 and len(A3)>0:
        question = {"Q":Q, "A1":A1, "A2":A2, "A3":A3}
    else:
        question = None
    
    #print(cleaned_text)
    #if question is not None: 
    #    print(question)
    return question

#send a question to google 4 times and return the question, and the google results. 
def google_search(q, service):
    
    searches = [q['Q'], q['A1'], q['A2'], q['A3']]
    #if pool == None:
        #fire up the parallel search pool
    
    #do all 4 google searches in parallel
    #[res, res_A1, res_A2, res_A3] = google_cse_parallel(searches, service)
    
                
    
    res = google_cse(q['Q'], service)
    
    res_A1 = google_cse(q['A1'], service)
    res_A2 = google_cse(q['A2'], service)
    res_A3 = google_cse(q['A3'], service)
    
    #add the google searches to the question dictoinary
    q['Q_res'] = res
    q['A1_res'] = res_A1
    q['A2_res'] = res_A2
    q['A3_res'] = res_A3
    
    return(q)

#this block has the functions for processing questions into features


def google_cse(search, service):
    #print(search)
    cx2 = 'google_token'
    cx1 = 'google_token'
    res = service.cse().list(
      q=search,
      cx=cx1, num = 10
    ).execute()
    return res

#return the snippets of pages from the google search page.
def return_snippets(res):
    snippet = ""
    if 'items' in res:
        for item in res['items']:
            snippet += item['snippet']
    return snippet

def search_term_from_question(q):
    search_term = []
    #check for quotation marks in the question
    #print(q)
    q = q.replace('“', '"').replace('”','"')
    #print(q)
    first_quote = q.find('\"')
    if first_quote != -1:
        second_quote = q.find('\"',first_quote+1)
        search_term.append(q[first_quote+1:second_quote])
    
    #check for capitalized words
    caps = re.findall('([A-Z][a-z]+)', q)
    #print(caps)
    caps = [x for x in caps if x.lower() not in stop_words]
    if len(caps)>0:
        search_term.append(" ".join(caps))
        
    #if nothing above was found, just return the question minus stop words
    if len(search_term)==0:
        q_word_list = [x for x in q.split() if x.lower() not in stop_words]
        search_term.append(" ".join(q_word_list))
    return search_term

def remove_stop_words(text, stop_words):
    #print("Input Text: \n %s"%text)
    #remove punctuation
    exclude = set(string.punctuation)
    translator = str.maketrans("","", string.punctuation)
    text = text.translate(translator)
    
    text_list = text.lower().split(" ")
    stopped_list = [x.strip() for x in text_list if x not in english_stop_words] #remove whitespaces from words
    l = len(stopped_list)
    if l == 0:
        l = 1
    #print(stopped_list)
    stopped_text = " ".join(stopped_list)
    stopped_text = lemmatize_text(stopped_text)
    return stopped_text, l
    
def question_to_numpy(question, negative_words, stop_words):
    question = create_features(question, negative_words, stop_words)
    new = np.asarray([question['Neg'],
        question['A1_count'],
        #question['A1_dist'],
        question['A2_count'],
        #question['A2_dist'],
        question['A3_count'],
        #question['A3_dist'],
        question['Q_A1_count'],
        #question['Q_A1_dist'],
        question['Q_A2_count'],
        #question['Q_A2_dist'],
        question['Q_A3_count'],
        #question['Q_A3_dist'],
        question['Q_W1_present'],
        question['Q_W1_count'],
        #question['Q_W1_dist'],
        question['Q_W2_present'],
        question['Q_W2_count'],
        #question['Q_W2_dist'],
        question['Q_W3_present'],
        question['Q_W3_count'],
        #question['Q_W3_dist'],
        question['A1_cross_count'],
        question['A2_cross_count'],
        question['A3_cross_count'],
        question['A1_total_results'],
        question['A2_total_results'],
        question['A3_total_results'],
        question['A1_context_count'],
        #question['A1_context_dist'],
        question['A2_context_count'],
        #question['A2_context_dist'],
        question['A3_context_count']])
        #question['A3_context_dist']])
    X = new.T
    #print(X)
    #Y = np.array(question['ANS'])
    X = np.reshape(X, (new.shape[0],1))
    Y = None #np.array(question['ANS'])
    #X = np.reshape(X, (new.shape[0],1))
    return(question, X, Y)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    #print("Input Text:")
    #print(text)
    text_list = text.split(" ")
    lemmas = []
    for word in text_list:
            lem = lemmatizer.lemmatize(word)
            #print(lem, word)
            lemmas.append(lem)
    out_text = " ".join(lemmas)  
    #print("Output Text:")
    #print(out_text)
    return out_text
    
def get_total_results(A_res):
    #print(A_res.keys())
    total_results = int(A_res['queries']['request'][0]['totalResults'])
    
    #snippet = ""
    #if 'items' in res:
    #    for item in res['items']:
    #        snippet += item['snippet']
    #return snippet
    return total_results

#give a corpus and keywords
#return the normalized keyword count for the corpus. 
#also, return a measure of how close together the keywords appeared.
def count_and_dist(corpus, keywords):
    count = 0
    dist = 0
    dist_list = []
    corpus_list = corpus.split(" ")
    corpus_len = len(corpus_list)
    if corpus_len == 0:
        corpus_len = 1
    #print(keywords)
    #keyword_set = set(keywords)
    #print(keyword_set)
    for word in corpus_list:
        #print(word)
        if word in keywords:
            count+=1
            dist_list.append(dist)
            dist = 0
        else:
            dist+=1
    if len(dist_list)>0:
        #print(dist_list)  
        #print(statistics.median(dist_list))
        #remove the elements in the list that are greater than the median
        median = statistics.median(dist_list)
        dist_list = [x for x in dist_list if x <= median]
        #print(median)
        #print(dist_list)
        #return the mean of the remaining items
        dist = statistics.mean(dist_list)
    else:
        dist = -1
    #print(count)
    count = count / corpus_len
    #print(count)
    return count, dist
  
    
#build the features for a question. 
#take a question dict, and add the features required to it. 
def create_features(question, negative_words, stop_words):
    negative = -1
    qwords = question['Q'].lower().split()
    if [i for i in qwords if i in negative_words]:
            negative = 1
        
    #count the number of instances of A1, A2, A3 in Q_res snippets
    snippets = return_snippets(question['Q_res'])
    
    #count how many times each answer appears in the result
    A1_list = lemmatize_text(question['A1'].lower()).split()
    #A1_list = [" " + word + " " for word in A1_list]
    A2_list = lemmatize_text(question['A2'].lower()).split()
    #A2_list = [" " + word + " " for word in A2_list]
    A3_list = lemmatize_text(question['A3'].lower()).split()
    #A3_list = [" " + word + " " for word in A3_list]
    #print(A1_list)
    #print(A2_list)
    #print(A3_list)
    
    #remove stop works from the snippets
    snippets, l = remove_stop_words(snippets, english_stop_words)
    #count
    #print("Snippets: ", snippets)
    #print("A1_list: ", A1_list)
    A1_count, A1_dist = count_and_dist(snippets, A1_list)
    #print("A1_count: ", str(A1_count))
    A2_count, A2_dist = count_and_dist(snippets, A2_list)
    A3_count, A3_dist = count_and_dist(snippets, A3_list)
    
    #count how many times words from the question appear in the results from the answers
    q_word_list = lemmatize_text(question['Q'].lower()).strip('?').split()
    #print(q_word_list)
    #remove the stop words
    #q_word_list = [" " + x + " " for x in q_word_list if x not in stop_words]
    #print(q_word_list)
    
    snippets_A1 = return_snippets(question['A1_res'])
    snippets_A1, l_A1 = remove_stop_words(snippets_A1, english_stop_words)
    
    snippets_A2 = return_snippets(question['A2_res'])
    snippets_A2, l_A2 = remove_stop_words(snippets_A2, english_stop_words)
    
    snippets_A3 = return_snippets(question['A3_res'])
    snippets_A3, l_A3 = remove_stop_words(snippets_A3, english_stop_words)
    
    Q_A1_count, Q_A1_dist = count_and_dist(snippets_A1, q_word_list)
    Q_A2_count, Q_A2_dist = count_and_dist(snippets_A2, q_word_list)
    Q_A3_count, Q_A3_dist = count_and_dist(snippets_A3, q_word_list)
    
    #google searches with context
    #get the number of search results
    A1_total_results = get_total_results(question['A1_res_with_context'])
    A2_total_results = get_total_results(question['A2_res_with_context'])
    A3_total_results = get_total_results(question['A3_res_with_context'])
    
    #word counds from the question
    snippets_A1_context = return_snippets(question['A1_res_with_context'])
    snippets_A1_context, l_A1 = remove_stop_words(snippets_A1_context, english_stop_words)
    A1_context_count, A1_context_dist = count_and_dist(snippets_A1_context, q_word_list)
    
    snippets_A2_context = return_snippets(question['A2_res_with_context'])
    snippets_A2_context, l_A2 = remove_stop_words(snippets_A2_context, english_stop_words)
    A2_context_count, A2_context_dist = count_and_dist(snippets_A2_context, q_word_list)
    
    snippets_A3_context = return_snippets(question['A3_res_with_context'])
    snippets_A3_context, l_A3 = remove_stop_words(snippets_A3_context, english_stop_words)
    A3_context_count, A3_context_dist = count_and_dist(snippets_A3_context, q_word_list)
    
    
    A1_wiki_text = question['A1_wiki_text']
    A2_wiki_text = question['A2_wiki_text']
    A3_wiki_text = question['A3_wiki_text']
    
    if A1_wiki_text is not None:
        A1_wiki_text, A1_wiki_text_len = remove_stop_words(A1_wiki_text, english_stop_words)
        
    if A2_wiki_text is not None:
        A2_wiki_text, A2_wiki_text_len = remove_stop_words(A2_wiki_text, english_stop_words)
        
    if A3_wiki_text is not None:
        A3_wiki_text, A3_wiki_text_len = remove_stop_words(A3_wiki_text, english_stop_words)
    
    if A1_wiki_text is not "":
        Q_W1_count, Q_W1_dist = count_and_dist(A1_wiki_text, q_word_list)
        Q_W1_present = 1
    else:
        Q_W1_count = 0
        Q_W1_present = 0
        Q_W1_dist = -1
    if A2_wiki_text is not "":
        Q_W2_count, Q_W2_dist = count_and_dist(A2_wiki_text, q_word_list)
        Q_W2_present = 1
    else:
        Q_W2_count = 0
        Q_W2_present = 0
        Q_W2_dist = -1
    if A3_wiki_text is not "":
        Q_W3_count, Q_W3_dist = count_and_dist(A3_wiki_text, q_word_list)
        Q_W3_present = 1
    else:
        Q_W3_count = 0
        Q_W3_present = 0
        Q_W3_dist = -1
    
    #do the counts from the online wiki searches
    Q_online_wiki_text, l = remove_stop_words(question['Q_online_wiki_text'], english_stop_words)
    #A1_online_wiki_count = sum([Q_online_wiki_text.lower().count(A_word) for A_word in A1_list])/l
    #A2_online_wiki_count = sum([Q_online_wiki_text.lower().count(A_word) for A_word in A2_list])/l
    #A3_online_wiki_count = sum([Q_online_wiki_text.lower().count(A_word) for A_word in A3_list])/l
    #print((A1_wiki_count, A2_wiki_count, A3_wiki_count))
    
    
    #A1_online_wiki_text, A1_online_wiki_text_len = remove_stop_words(question['A1_online_wiki_text'], english_stop_words)
    #Q_online_W1_count = sum([A1_online_wiki_text.count(q_word) for q_word in q_word_list])/A1_online_wiki_text_len
    
    #A2_online_wiki_text, A2_online_wiki_text_len = remove_stop_words(question['A2_online_wiki_text'], english_stop_words)
    #Q_online_W2_count = sum([A2_online_wiki_text.count(q_word) for q_word in q_word_list])/A2_online_wiki_text_len
    
    #A3_online_wiki_text, A3_online_wiki_text_len = remove_stop_words(question['A3_online_wiki_text'], english_stop_words)
    #Q_online_W3_count = sum([A3_online_wiki_text.count(q_word) for q_word in q_word_list])/A3_online_wiki_text_len
    
    
    #cross count words
    Q_text, l = remove_stop_words(return_snippets(question['Q_res']) + Q_online_wiki_text, english_stop_words)
    A1_text, l = remove_stop_words(" ".join([return_snippets(question['A1_res']), str(A1_wiki_text), return_snippets(question['A1_res_with_context'])]), english_stop_words)
    A2_text, l = remove_stop_words(" ".join([return_snippets(question['A2_res']), str(A2_wiki_text), return_snippets(question['A2_res_with_context'])]), english_stop_words)
    A3_text, l = remove_stop_words(" ".join([return_snippets(question['A3_res']), str(A3_wiki_text), return_snippets(question['A3_res_with_context'])]), english_stop_words)


    A1_cross_count = cross_count_words(Q_text, A1_text)
    A2_cross_count = cross_count_words(Q_text, A2_text)
    A3_cross_count = cross_count_words(Q_text, A3_text)
    
    
    question['Neg'] = negative
    question['A1_count'] = A1_count
    question['A2_count'] = A2_count
    question['A3_count'] = A3_count
    question['Q_A1_count'] = Q_A1_count
    question['Q_A2_count'] = Q_A2_count
    question['Q_A3_count'] = Q_A3_count
    question['Q_W1_present'] = Q_W1_present
    question['Q_W1_count'] = Q_W1_count
    question['Q_W2_present'] = Q_W2_present
    question['Q_W2_count'] = Q_W2_count
    question['Q_W3_present'] = Q_W3_present
    question['Q_W3_count'] = Q_W3_count
    question['A1_cross_count'] = A1_cross_count
    question['A2_cross_count'] = A2_cross_count
    question['A3_cross_count'] = A3_cross_count
    question['A1_total_results'] = A1_total_results
    question['A2_total_results'] = A2_total_results
    question['A3_total_results'] = A3_total_results
    question['A1_context_count'] = A1_context_count
    question['A2_context_count'] = A2_context_count
    question['A3_context_count'] = A3_context_count
    question['A1_dist'] = A1_dist
    question['A2_dist'] = A2_dist
    question['A3_dist'] = A3_dist
    question['Q_A1_dist'] = Q_A1_dist
    question['Q_A2_dist'] = Q_A2_dist
    question['Q_A3_dist'] = Q_A3_dist
    question['A1_context_dist'] = A1_context_dist
    question['A2_context_dist'] = A2_context_dist
    question['A3_context_dist'] = A3_context_dist
    question['Q_W1_dist'] = Q_W1_dist
    question['Q_W2_dist'] = Q_W2_dist
    question['Q_W3_dist'] = Q_W3_dist
    return(question)

#lookup terms in the local copy of wikipedia
def get_wiki_articles(answers, ix):
    qp = QueryParser("title", schema=ix.schema)
    #print("Searching For: %s"%answers)
    q = qp.parse(answers)

    with ix.searcher() as s:
        results = s.search(q, limit=1)
        fname = None
        if len(results)>0:
            for result in results:
                fname = result['file_path']
                title = result['title']
                #print("Found: %s"%result['title'])
                text = get_article_text(fname, title)
        else:
            text = None
    return text

def get_article_text(fname, search_text):
    text = None
    with open(fname, encoding = 'utf-8') as f:
        content = f.readlines()
    for line in content:
        data = json.loads(line)
        if data['title'] == search_text:
            text = data['text']
    return(text)

def online_wikipedia(search_text):
    #print("Search Text: %s"%search_text)
    root_url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch='
    TAG_RE = re.compile(r'<[^>]+>')
    url = root_url + urllib.parse.quote(search_text)
    #print(url)
    f = urllib.request.urlopen(url)
    result = json.loads(f.read())
    result_text = ""
    for item in result['query']['search']:
        #print(item['title'])
        snippet = TAG_RE.sub('', item['snippet'])
        #print(snippet)
        result_text += " " + item['title'] + snippet
    return result_text    
    
def online_full_wikipedia(search_text):
    #print("Search Text: %s"%search_text)
    root_url = 'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&titles=%s&redirects=true'
    TAG_RE = re.compile(r'<[^>]+>')
    url = root_url%urllib.parse.quote(search_text)
    #print(url)
    f = urllib.request.urlopen(url)
    result = json.loads(f.read())
    if 'pages' in result['query']:
        page_num = list(result['query']['pages'].keys())
        #print(page_num)
        if page_num[0] == '-1':
            text = ''
        else:
            #print("Title: %s"%result['query']['pages'][page_num[0]]['title'])
            text = TAG_RE.sub('', result['query']['pages'][page_num[0]]['extract'])
    else:
        text = ''
    return text
    
def get_prediction(X, predict_send, predict_receive):
    prediction = None
    #prediction = model.predict(X.T)
    predict_send.send_pyobj(X)
    
    poller = zmq.Poller()
    poller.register(predict_receive, zmq.POLLIN)
    while True:
        socks = dict(poller.poll())
        if socks.get(predict_receive) == zmq.POLLIN:
            prediction = predict_receive.recv_pyobj()
            break
        

    
    
    return(prediction)

def open_predict_socket():
    print("Initializing Predict Socket")
    # Initialize a zeromq context
    context = zmq.Context()

    # Set up a channel to send work
    predict_send = context.socket(zmq.PUSH)
    predict_send.bind("tcp://127.0.0.1:5560")
    
    predict_receive = context.socket(zmq.PULL)
    predict_receive.bind("tcp://127.0.0.1:5561")
    

    # Give everything a second to spin up and connect
    time.sleep(1)
    print("Done...")
    return (predict_send, predict_receive)
    
    
def print_red(text):
    #str_start = '\x1b[31m'
    #str_end = '\x1b[0m'
    
    return Fore.RED + text + Style.RESET_ALL
    
def print_green(text):
    return Fore.GREEN + text + Style.RESET_ALL
    
    
def open_ventilator_socks():
    print("Initializing Ventilator")
    # Initialize a zeromq context
    context = zmq.Context()

    # Set up a channel to send work
    ventilator_send = context.socket(zmq.PUSH)
    ventilator_send.bind("tcp://127.0.0.1:5057")

    # Give everything a second to spin up and connect
    time.sleep(1)
    print("Done...")
    return ventilator_send
    
def ventilator(ventilator_send, search_list):
    
    # Send the numbers between 1 and 1 million as work messages
    for search in search_list:
        #print("Ventilator sending %s\n"%search['header'])
        work_message = search
        ventilator_send.send_json(work_message)

def worker(wrk_num):
    #open the wikipedia index
    index_path = 'G:\\Wikipedia Dump\\indexdir\\'
    ix = index.open_dir(index_path, indexname = "title_index")
    
    #start the google service
    service = build("customsearch", "v1",
        developerKey="AIzaSyA-oXw1zepxqybzDFCqCCi_CID59xQEWJY")
        
    # Initialize a zeromq context
    context = zmq.Context()

    # Set up a channel to receive work from the ventilator
    work_receiver = context.socket(zmq.PULL)
    work_receiver.connect("tcp://127.0.0.1:5057")

    # Set up a channel to send result of work to the results reporter
    results_sender = context.socket(zmq.PUSH)
    results_sender.connect("tcp://127.0.0.1:5058")

    # Set up a channel to receive control messages over
    control_receiver = context.socket(zmq.SUB)
    control_receiver.connect("tcp://127.0.0.1:5559")
    control_receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set up a poller to multiplex the work receiver and control receiver channels
    poller = zmq.Poller()
    poller.register(work_receiver, zmq.POLLIN)
    poller.register(control_receiver, zmq.POLLIN)        
    # Loop and accept messages from both channels, acting accordingly
    while True:
        socks = dict(poller.poll())
        res = "results"
        #print(socks)
        #print(socks.get())
        # If the message came from work_receiver channel, square the number
        # and send the answer to the results reporter
        if socks.get(work_receiver) == zmq.POLLIN:
            
            work_message = work_receiver.recv_json()
            type = work_message['type']
            #print("Worker %d got request %s, %s\n"%(wrk_num, work_message['header'], work_message['search_text']))
            if type == "Google":
                #do the google search here
                res = google_cse(work_message['search_text'], service)
                #print(work_message['search_text'])
                
            if type == "Wiki":
                #do the wiki search here
                if work_message['header'] == "Q_wiki_text":
                    Q_wiki_text = ""
                    #print(work_message['search_text'])
                    for term in work_message['search_text']:
                        #term = work_message['search_text']
                        #print(term)
                        text = get_wiki_articles(term, ix)
                        if text is not None:
                            Q_wiki_text += text
                    res = Q_wiki_text
                else:
                    res = get_wiki_articles(work_message['search_text'], ix)
                #print(work_message['search_text'])
            if type == "Wiki_online":
                res = online_full_wikipedia(work_message['search_text'])
                
            answer_message = { 'header' : work_message['header'], 'result' : res }
            results_sender.send_json(answer_message)
            #print("worker %d replying with answer."%wrk_num)
        # If the message came over the control channel, shut down the worker.
        if socks.get(control_receiver) == zmq.POLLIN:
            control_message = control_receiver.recv()
            if control_message == b"FINISHED":
                print("Worker %i received FINSHED, quitting!" % wrk_num)
                break
    
def open_result_socks():
    print("Initializing Results Controller Sockets")
    # Initialize a zeromq context
    context = zmq.Context()
    
    # Set up a channel to receive results
    results_receiver = context.socket(zmq.PULL)
    results_receiver.bind("tcp://127.0.0.1:5058")

    # Set up a channel to send control commands
    control_sender = context.socket(zmq.PUB)
    control_sender.bind("tcp://127.0.0.1:5559")
    time.sleep(1)
    print("Done...")
    return (results_receiver, control_sender)

def result_manager(results_receiver, num_searches):
    results = []
    for task_nbr in range(num_searches):
        #print("Task number %d\n"%task_nbr)
        result_message = results_receiver.recv_json()
        #print("Search returned is: %s" % (result_message['header']))
        results.append(result_message)
        #for r in results:
        #    print(r['header'])
    return(results)

def close_workers(control_sender):
    # Signal to all workers that we are finsihed
    print("Sending Finish Signal")
    control_sender.send_string("FINISHED")
    time.sleep(5)

def open_cursor_control():
    print("Initializing cursor control")
    # Initialize a zeromq context
    context = zmq.Context()

    # Set up a channel to send work
    cursor_control = context.socket(zmq.PUSH)
    cursor_control.bind("tcp://127.0.0.1:5562")

    # Give everything a second to spin up and connect
    time.sleep(1)
    print("Done...")
    return cursor_control
    
def update_cursor_state(cursor_control, cursor_state):
    cursor_control.send_json(cursor_state)

def cursor_animator():
    # Initialize a zeromq context
    context = zmq.Context()

    # Set up a channel to receive work from the ventilator
    cursor_receiver = context.socket(zmq.PULL)
    cursor_receiver.connect("tcp://127.0.0.1:5562")

    #initialize cursor type:
    type = 0 #nothing
    #1 = spinney
    #2 = printing dots '.'
    
    
    # Set up a poller to listen for the cursor state.
    poller = zmq.Poller()
    poller.register(cursor_receiver, zmq.POLLIN)
    
    #spinney cursor characters
    cursors = ['\\', '|', '/', '-']
    
    # Loop and accept messages from both channels, acting accordingly
    while True:
        socks = dict(poller.poll(1))
        
        if socks.get(cursor_receiver) == zmq.POLLIN:
            work_message = cursor_receiver.recv_json()
            type = work_message['State']
            
        if type == 1:
            #do the spinney cursor thing
            for i, cursor in enumerate(cursors):
                sys.stdout.write(cursor)
                time.sleep(0.1)
                sys.stdout.write('\r')
                
        if type == 2:
            #print some dots
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(0.1)

def cross_count_words(text1, text2):
    words1 = set(re.split("[^a-zA-Z]+", text1.lower()))
    words2 = set(re.split("[^a-zA-Z]+", text2.lower()))
    intersection = words1 & words2
    #print(intersection)
    #print("Text1 Len: %d"%len(text1.split()))
    #print("Text2 Len: %d"%len(text2.split()))
    l = len(text2.split())
    if l == 0:
        l = 1
    return(len(intersection)/l)

def question_to_log_file(question, X, prediction, f):
    
    f.write("%s\n"%question['Q'])
    for i, (pred, a) in enumerate(zip(list(prediction[0]), [question['A1'],question['A2'],question['A3']])):
        ans_string = 'A%d: %0.2f %s'%(i+1, pred,a)
        f.write("%s\n"%ans_string)
    f.write("X: %s\n"%X)
    f.write('Negative: %d\n'%question['Neg'])
    f.write('A1 Count: %f\n'%question['A1_count'])
    f.write('A2 Count: %f\n'%question['A2_count'])
    f.write('A3 Count: %f\n'%question['A3_count'])
    f.write('Q A1 Count: %f\n'%question['Q_A1_count'])
    f.write('Q A2 Count: %f\n'%question['Q_A2_count'])
    f.write('Q A3 Count: %f\n'%question['Q_A3_count'])
    f.write('Q W1 Count: %f\n'%question['Q_W1_count'])
    f.write('Q W2 Count: %f\n'%question['Q_W2_count'])
    f.write('Q W3 Count: %f\n'%question['Q_W3_count'])
    f.write('Q A1 Context Count: %f\n'%question['A1_context_count'])
    f.write('Q A2 Context Count: %f\n'%question['A2_context_count'])
    f.write('Q A3 Context Count: %f\n'%question['A3_context_count'])
    #f.write('Q Online wiki A1 Count: %f\n'%question['Q_online_wiki_A1_count'])
    #f.write('Q Online wiki A2 Count: %f\n'%question['Q_online_wiki_A2_count'])
    #f.write('Q Online wiki A3 Count: %f\n'%question['Q_online_wiki_A3_count'])
    #f.write('Q Online wiki W1 Count: %f\n'%question['Q_online_W1_count'])
    #f.write('Q Online wiki W2 Count: %f\n'%question['Q_online_W2_count'])
    #f.write('Q Online wiki W3 Count: %f\n'%question['Q_online_W3_count'])
    f.write('A1 Cross Count: %f\n'%question['A1_cross_count'])
    f.write('A2 Cross Count: %f\n'%question['A2_cross_count'])
    f.write('A3 Cross Count: %f\n'%question['A3_cross_count'])
    
    f.write("\nNumber of pages returned:\n")
    f.write("\nA1: %d"%get_total_results(question['A1_res_with_context']))
    f.write("\nA2: %d"%get_total_results(question['A2_res_with_context']))
    f.write("\nA3: %d"%get_total_results(question['A3_res_with_context']))
    
    
    f.write("\nGoogle Question Search:\n")
    f.write("%s\n\n"%return_snippets(question['Q_res']))
    f.write("Google A1 Search:\n")
    f.write("%s\n\n"%return_snippets(question['A1_res']))
    f.write("Google A2 Search:\n")
    f.write("%s\n\n"%return_snippets(question['A2_res']))
    f.write("Google A3 Search:\n")
    f.write("%s\n\n"%return_snippets(question['A3_res']))
    
    f.write("\nGoogle Search with context:\n")
    f.write("Google A1 Search:\n")
    f.write("%s\n\n"%return_snippets(question['A1_res_with_context']))
    f.write("Google A2 Search:\n")
    f.write("%s\n\n"%return_snippets(question['A2_res_with_context']))
    f.write("Google A3 Search:\n")
    f.write("%s\n\n"%return_snippets(question['A3_res_with_context']))
   
    f.write("Wiki A1 search:\n")
    f.write("%s\n\n"%question['A1_wiki_text'])
    
    f.write("Wiki A2 search:\n")
    f.write("%s\n\n"%question['A2_wiki_text'])
    
    f.write("Wiki A3 search:\n")
    f.write("%s\n\n"%question['A3_wiki_text'])
    
    f.write("Online Wiki Q Search:\n")
    f.write("%s\n\n"%question['Q_online_wiki_text'])
    #f.write("Online Wiki A1 Search:\n")
    #f.write("%s\n\n"%question['A1_online_wiki_text'])
    #f.write("Online Wiki A2 Search:\n")
    #f.write("%s\n\n"%question['A2_online_wiki_text'])
    #f.write("Online Wiki A3 Search:\n")
    #f.write("%s\n\n"%question['A3_online_wiki_text'])
    
    
    
    
    f.write("\n\n Done\n")
    
    
    #new = np.asarray([question['Neg'],
        # question['A1_count'],
        # question['A2_count'],
        # question['A3_count'],
        # question['Q_A1_count'],
        # question['Q_A2_count'],
        # question['Q_A3_count'],
        # question['Q_W1_present'],
        # question['Q_W1_count'],
        # question['Q_W2_present'],
        # question['Q_W2_count'],
        # question['Q_W3_present'],
        # question['Q_W3_count'],
        # question['Q_online_wiki_A1_count'],
        # question['Q_online_wiki_A2_count'],
        # question['Q_online_wiki_A3_count'],
        # question['Q_online_W1_count'],
        # question['Q_online_W2_count'],
        # question['Q_online_W3_count'],
        # question['A1_cross_count'],
        # question['A2_cross_count'],
        # question['A3_cross_count']])        
    
    
    
    
    #for key in question.keys():
    ##    print("%s:"%key)
    #    print(question[key])
    return None
    
def scaled_X(X, scaler):
    #print(X.shape)
    #print(X.T)
    scaled_X_train = scaler.transform(X.T)
    scaled_X_train[:,0] = X.T[:,0]
    #print(scaled_X_train)
    return scaled_X_train
    
def get_continuous_chunks(text):
    pos_tags = pos_tag(word_tokenize(text))
    chunked = ne_chunk(pos_tags)
    prev = None
    continuous_chunk = []
    current_chunk = []
    #print(chunked)
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if continuous_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)
    out_chunk = []
    for chunk in continuous_chunk:
        if chunk.lower() not in english_stop_words:
            if chunk is not "":
                out_chunk.append(chunk)
    #if we found no named entity, just return the nouns and adjectives minus stop words
    if len(out_chunk)==0:
        #print("return Nouns")
        for i in pos_tags:
            #print(i)
            #print(type(i))
            if i[1].startswith("NN") or i[1].startswith("JJ"): #if it's a noun
                if i[0].lower() not in english_stop_words: #and it's not a stop word
                    out_chunk.append(i[0])
    return out_chunk
    
    
    
    


if __name__ == '__main__':
    #multiprocessing.set_start_method('spawn')
    
    #model = None
    blended_q_area = None
    confidence_flag = False
    got_text_flag = False
    got_question_flag = False
    q_text = ""
    count = 0
    M = None
    blend_count = 0
    confidence_thresh = 0.70

    #question_list = pickle.load(open("questions.p", 'rb'))
    question_list = []
    print("Loaded question file")
    print("Number of questions in file: %d"%len(question_list))
    previous_question = ""

    #model = load_model()
    #pool_size = 4
    #pool = multiprocessing.Pool(processes=pool_size)
    
    #load the scaler
    scaler = pickle.load(open('scaler.p', "rb"))
    
    
    #start the workers
    worker_pool = range(12)
    jobs = []
    print("Starting Workers")
    for wrk_num in range(len(worker_pool)):
        p = Process(target=worker, args=(wrk_num,))
        p.start()
        #print(p, p.is_alive())
        jobs.append(p)
    #start the cursor animator worker    
    print("Starting Cursor Animator")
    p = Process(target=cursor_animator)
    p.start()
    print("Started.")
    jobs.append(p)
    
    #cursor control
    cursor_control = open_cursor_control()
    
    
    time.sleep(1)
    
    # Start the ventilator!
    ventilator_send = open_ventilator_socks()
    # open the result receiver sockets
    results_receiver, control_sender = open_result_socks()
    (predict_send, predict_receive) = open_predict_socket()
    
    
    #log file
    log_fname = "HQ_log-" + datetime.datetime.now().strftime('%y-%m-%d-%H_%M_%S') + '.log'
    #log_file = open(log_fname, 'w', encoding = 'utf-8')
    

    
    
    
    with mss.mss() as sct:
        monitor_number = 2
        mon = sct.monitors[monitor_number]
        # Part of the screen to capture
        monitor = {
            'top': mon['top'] + 57,  # 100px from the top
            'left': mon['left'] + 968,  # 100px from the left
            'width': 324,
            'height': 561,
            'mon': monitor_number,
        }

        while 'Screen capturing':
            update_cursor_state(cursor_control, {'State':1})
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            frame = np.array(sct.grab(monitor))
            if M is None:
                rows,cols,color = frame.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
            
            #frame =  cv2.warpAffine(frame,M,(rows,cols))
            # Display the picture in grayscale
            # cv2.imshow('OpenCV/Numpy grayscale',
            #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

            #print('fps: {0}'.format(1 / (time.time()-last_time)))

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        
            #rotate the frame 90 degrees

            #frame =  cv2.warpAffine(frame,M,(cols,rows))
            imS = frame


            cv2.imshow('raw input',imS)
            #cv2.imwrite("frame%d.jpg" % count, frame)
            #count = count + 1


            thresh = threshold_image(frame)
            #q_area = find_white_card(thresh, frame)
            q_area = find_contours(thresh, frame)

            if q_area is None:
                blended_area = None
                blend_count = 0
                confidence_flag = False
                got_text_flag = False
                got_question_flag = False


            if q_area is not None:
                if blend_count == 0:
                    #the first time we're here. Start the clock. 
                    start = datetime.datetime.now()
                #if blended_q_area is None: 
                #blended_q_area = q_area
                blend_count += 1
                #else:
                #    (x,y,c) = q_area.shape
                #    blended_q_area = cv2.resize(blended_q_area, (y,x))
                #    blended_q_area = cv2.addWeighted(blended_q_area, 0.1, q_area, 0.9, 0)

                #print(blend_count)
                if confidence_flag == False and blend_count > 0:
                    #print("".join(q_text.split()))
                    count += 1


                    q_text, confidence_flag = get_text(q_area, q_text, count, previous_question)
                    #print(q_text)
                if confidence_flag == True and got_text_flag == False and q_text is not "":
                    question = clean_text(q_text)
                    if question is not None: 
                        print("Searching...")
                        update_cursor_state(cursor_control, {'State':2})
                        got_text_flag = True
                        previous_question = question['Q']
                        #question = google_search(question)
                        
                        #get some keyworks from the question text
                        named_entities = get_continuous_chunks(question['Q'])
                        search_A1 = question['A1'] + " " + " ".join(named_entities)
                        search_A2 = question['A2'] + " " + " ".join(named_entities)
                        search_A3 = question['A3'] + " " + " ".join(named_entities)

                        #do the google searches
                        search_list = [{'type':'Google','header': 'Q_res','search_text':question['Q']},
                                        {'type':'Google','header': 'A1_res','search_text':question['A1']},
                                        {'type':'Google','header': 'A2_res','search_text':question['A2']},
                                        {'type':'Google','header': 'A3_res','search_text':question['A3']},
                                        {'type':'Wiki_online','header': 'A1_wiki_text','search_text':question['A1']},
                                        {'type':'Wiki_online','header': 'A2_wiki_text','search_text':question['A2']},
                                        {'type':'Wiki_online','header': 'A3_wiki_text','search_text':question['A3']},
                                        {'type':'Wiki_online','header': 'Q_online_wiki_text','search_text':question['Q']},
                                        {'type':'Google','header': 'A1_res_with_context','search_text':search_A1},
                                        {'type':'Google','header': 'A2_res_with_context','search_text':search_A2},
                                        {'type':'Google','header': 'A3_res_with_context','search_text':search_A3},
                                        #{'type':'Wiki_online','header': 'A1_online_wiki_text','search_text':question['A1']},
                                        #{'type':'Wiki_online','header': 'A2_online_wiki_text','search_text':question['A2']},
                                        #{'type':'Wiki_online','header': 'A3_online_wiki_text','search_text':question['A3']},
                                        ]
                                        
                        ventilator(ventilator_send, search_list)
                        #print("Sent  %d searches"%(len(search_list)))
                        results = result_manager(results_receiver, len(search_list))
                        #print("Got Results")
                        for result in results:
                            question[result['header']] = result['result']
                        #check that we got q_wiki_text
                        #print(question['Q_wiki_text'])
                        #print("Making Features")
                        question, X, Y = question_to_numpy(question, negative_words, stop_words)
                        X = scaled_X(X, scaler)
                        prediction = get_prediction(X, predict_send, predict_receive)
                        #print("Got Prediction")
                        #print(Y)
                        #print(prediction)
                        ANS = np.argmax(prediction[0])
                        NOT_ANS = np.argmin(prediction[0])
                       
                        print("Q:  %s"%question['Q'])
                        for i, (pred, a) in enumerate(zip(list(prediction[0]), [question['A1'],question['A2'],question['A3']])):
                            ans_string = 'A%d: %0.2f %s'%(i+1, pred,a)
                            if ANS == i:
                                #this is the right answer
                                print(print_green(ans_string))
                            elif ANS != i and prediction[0][ANS]>confidence_thresh:
                                print(print_red(ans_string))
                            else:
                                print(ans_string)
                        end = datetime.datetime.now()
                        delta = end - start
                        #print(question)
                        #print(answer)
                        print("Getting Question took: %s"%delta)
                        print("\n\n")
                        update_cursor_state(cursor_control, {'State':0})
                        #question_to_log_file(question, X, prediction, log_file)
                        question_list.append(question)
                        
                        
                        

                #M_q_area = cv2.getRotationMatrix2D((q_cols/2,q_rows/2),-90,1)
                #rot =  cv2.warpAffine(blended_q_area,M_q_area,(cols,rows))
                #imS = cv2.resize(blended_q_area, (480, 290))
                #cv2.imshow('blended_q_area',imS)


    #cap.release()
    cv2.destroyAllWindows()  # destroy all the opened windows
    #log_file.close()
    #close the workers
    close_workers(control_sender)
    print("Question Log has %d questions"%len(question_list))
    print("Saving Questions to questions.p file")
    pickle.dump(question_list, open( "questions.p", "wb"))
    for p in jobs:
        print("Terminating Process")
        p.terminate()
        