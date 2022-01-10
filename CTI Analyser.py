#CTI Analyser
#Nathan Andrew Deguara
#S16109197
#Masters Project

#Import required librarys
import pandas as pd
import numpy as np
import gensim
import warnings
from stix.core import STIXPackage
from stix.report import Report
from stix.report.header import Header
from mixbox.idgen import set_id_namespace
from mixbox.namespaces import Namespace
from stix.indicator import Indicator
from stix.ttp import TTP

#Removes warning messages from output window
warnings.filterwarnings("ignore")

#Import Data
content = pd.read_csv('IntentDetection/IntentDetectionData(Content).csv', usecols = [0,1,2])
hackers = pd.read_csv('IntentDetection/IntentDetectionData(Author).csv')

#Infuencial Hackers
iHackers = hackers.loc[hackers['Reputation'] > 10, 'Id']

#Get Posts
content.rename({"Source": "Id"}, axis=1, inplace=True)
iContent = pd.merge(iHackers, content, on=['Id'])

#Word2Vec Analyser
class Model():
    posts = iContent.Post.apply(gensim.utils.simple_preprocess)
    analyser = gensim.models.Word2Vec(window=10, min_count=2)
    analyser.build_vocab(posts, progress_per=1000)
    #analyser.train(posts, total_examples=analyser.corpus_count, epochs=analyser.epochs)
    
class Analyser():
    #Find CTI search words based on similarity score
    word1List = ["sql", "virus", "compromise", "malware", "script", "xss", "phishing", "spoofing", "rat", "encryption"]
    word2List = list(Model.analyser.wv.vocab)
    searchWords = []   
    for i in word1List:
        for j in word2List:
            ctiCheck = Model.analyser.wv.similarity(w1=i, w2=j)
            if ctiCheck >= 0.9:
                searchWords.append(i)
                searchWords.append(j)
    
    #Remove duplicates
    searchWords = list(dict.fromkeys(searchWords))
    
    #Find posts containing search words
    ctiFound = []
    for i in searchWords:
        ctiSearch = iContent[iContent['Post'].str.contains(i)]
        ctiFound.append(ctiSearch)
        
    #Combine the list of dataframes into one dataframe
    ctiList = pd.DataFrame({'Id' : []})
    for i in ctiFound:
        ctiRow = pd.concat([i])
        ctiList = pd.concat([ctiList, ctiRow])
            
    #Remove duplicates
    ctiList = ctiList.drop_duplicates(subset='Post', keep="last")
    
    #Create CSV
    ctiList.to_csv("CTI.csv", mode="w")
            
class Report_Generator():
    for i, r in Analyser.ctiList.iterrows():
        ns = Namespace("http://CRIReport.com", "CTI Report")
        set_id_namespace(ns)
        sp = STIXPackage()
        sr = Report()
        sr.header = Header()
        sr.header.description = "CTI Report"
        ind = Indicator()
        ind.title = r['Post']
        ttpTitle = r['Id']
        activity = TTP(title=ttpTitle)
        sp.add_indicator(ind)
        sp.add_ttp(activity)
        sp1 = str(sp.to_xml())
        sfn= "CTI%d.xml" %i
        f = open("Reports/%s" %sfn, "w", encoding="utf=8")
        f.write(sp1)
        f.close()
        
#Main
def main():
    Model.analyser.train(Model.posts, total_examples = Model.analyser.corpus_count, epochs = Model.analyser.epochs)
    print("---- Analysis Complete ----")
main()