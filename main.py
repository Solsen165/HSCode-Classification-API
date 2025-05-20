import pandas as pd
import numpy
import csv
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score
import pickle
from pydantic import BaseModel

from fastapi import FastAPI
import uvicorn

def load_hsCodes():
    path = 'hsCodesWithChapterNames.csv'
    df = pd.read_csv(path, sep=';')
    return df

def preprocess(df):
    chapters = df['Chapter']

    def cleanText(text):
        # Remove non letters
        #print(text)
        text = re.sub("[^a-zA-Z]",' ', text)

        # Remove white space
        text = ' '.join(text.split())

        # Convert to lower case
        text = text.lower()
        return text

    def removeStopWords(text):
        stopWords = stopwords.words('english')
        noStopWords = [word for word in text.split() if not word in stopWords]
        return ' '.join(noStopWords)
    
    df['clean_text'] = df['Description'].apply(cleanText)
    df['clean_text'] = df['clean_text'].apply(removeStopWords)

    return df

def textAugmentation(df):
    def synonym_replacement(text):
        words = text.split()
        new_words = []
        for word in words:
            synonyms = wordnet.synsets(word)
            if synonyms:
                #print(synonyms)
                new_words.append(synonyms[0].lemmas()[0].name())
            else:
                new_words.append(word)
        return ' '.join(new_words)

    df['augmented'] = df['clean_text'].apply(synonym_replacement)
    df['clean_text'] = df['clean_text'] + ' ' + df['augmented']

    return df


def trainParentModel(df):

    X = list(df['clean_text'])
    y = list(df['Chapter'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 64, shuffle = True)

    model = make_pipeline(TfidfVectorizer(), LinearSVC())

    model.fit(X_train, y_train)
    
    pickle.dump(model, open('HSCodeChapterClassifier.sav', 'wb'))

def trainChildModels(df):
    childModels = {}
    uniqueChapters = numpy.unique(df['Chapter'])
    for chapterName in uniqueChapters:
        rows = df.loc[df['Chapter'] == chapterName]
        X = list(rows['clean_text'])
        y = list(rows['Heading'])
        if len(numpy.unique(y)) < 2:
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 64, shuffle = True)
    
        Model = make_pipeline(TfidfVectorizer(),LinearSVC())
        Model.fit(X_train, y_train)
    
        y_pred = Model.predict(X_test)
    
        acc = round(accuracy_score(y_test, y_pred) * 100,2)
        #print(acc)
        childModels[chapterName] = Model

    pickle.dump(childModels, open('HSCodeHeadingClassifiers.sav', 'wb'))

def loadParentModel():
    return pickle.load(open('HSCodeChapterClassifier.sav','rb'))

def loadChildrenModels():
    return pickle.load(open('HSCodeHeadingClassifiers.sav','rb'))

def predict_probas(model, text):
    probas = model.decision_function([text])
    probas = [softmax(proba) for proba in probas]
    probas[0] = [round(proba*100,2) for proba in probas[0]]
    top_n_lables_idx = numpy.argsort(probas)[:,::-1]
    #print(top_n_lables_idx)
    top_n_probs = numpy.sort(probas)[:,::-1]
    top_n_probs = list(map(str,top_n_probs[0]))
    #print(top_n_probs[0])
    top_n_labels = [model.classes_[i] for i in top_n_lables_idx]
    #print(top_n_labels[0])
    results = list(zip(top_n_labels[0], top_n_probs))
    #results = [list(x) for x in zip(top_n_labels,top_n_probs)]
    return results

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)


#data = load_hsCodes()
#data = preprocess(data)
#data = textAugmentation(data)
#trainParentModel(data)
#trainChildModels(data)

#ParentModel = loadParentModel()
#ChildrenModels = loadChildrenModels()
#
#res = predict_probas(ParentModel, 'melons')
#print(res[:3])
#
#res2 = predict_probas(ChildrenModels[res[0][0]], 'melons')
#print(res2[:3])

def prepare():
    data = load_hsCodes()
    data = preprocess(data)
    data = textAugmentation(data)
    trainParentModel(data)
    trainChildModels(data)

class DescriptionRequest(BaseModel):
    text: str
    modelName: str

app = FastAPI()
prepare()
ParentModel = loadParentModel()
ChildrenModels = loadChildrenModels()

@app.post("/predict")
def predict(request: DescriptionRequest):
    print("Hello Post")
    print(request)
    if request.modelName == "Parent":
        return {"predictions": predict_probas(ParentModel, request.text)[:5]}
    else:
        return {"predictions": predict_probas(ChildrenModels[request.modelName], request.text)} 

@app.get('/')
def root():
    print("Hello World")
    return {"message": "Hello World"}



if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0", port=3000)



