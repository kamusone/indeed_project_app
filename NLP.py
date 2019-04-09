# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:50:57 2019

@author: Administrateur
"""
#source : https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#data: https://github.com/WillKoehrsen/Data-Analysis/tree/master/random_forest_explained
 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Read in data and display first 5 rows
df = pd.read_csv('final_indeed_bdd.csv')
df.columns
df.drop(['Nom','Titre','Titre2','Nom2','Metier_y'], axis=1, inplace=True)
df = df.drop_duplicates() 
df.dropna(axis=0, how='all',inplace=True)


#############################DATAPROCESSING###########################################
        #Adresse
df.Adresse=df.Adresse.str.lstrip('+-')
df.Adresse=df.Adresse.str.lstrip()
df['Adresse'].isna().sum()
df = df.drop(5415)  #c'est comme ca que ca fonctionne !!! ouf !
        #salaire
df_salaire = (df['Salaire']
            .str.replace(" ", "")
            .str.findall(r'(\d+\.?\d+)')
            .apply(lambda x: pd.Series(x).astype(float))
            .mean(1) #Moyenne si intervalle
            .to_frame('Salaire_integer')) 
df_salaire['per_Salaire'] = df['Salaire'].str.extract(r'(par\s\w+)')
df = pd.concat([df, df_salaire], axis=1)
df.drop(['Salaire'], axis=1, inplace=True)
df.loc[df['per_Salaire'] == 'par mois', 'Salaire_integer' ] = df['Salaire_integer'] * 12
df.loc[df['per_Salaire'] == 'par semaine', 'Salaire_integer' ] = df['Salaire_integer'] * 46
df.loc[df['per_Salaire'] == 'par heure', 'Salaire_integer' ] = df['Salaire_integer'] * 1600
df.loc[df['per_Salaire'] == 'par jour', 'Salaire_integer' ] = df['Salaire_integer'] * 228
#sous une même unité #https://stackoverflow.com/questions/36909977/update-row-values-where-certain-condition-is-met-in-pandas
df.rename(columns={'Salaire_integer':'Salaire_annuel'},inplace=True)
df.drop(['per_Salaire'], axis=1, inplace=True)
#df.hist(column='Salaire_annuel', by = "Metier_x")
#df["Salaire_annuel"] = df.groupby("Metier_x").transform(lambda x: x.fillna(x.mean()))
#or
#train et test set pour les salaires.
mask = df['Salaire_annuel'].isna()
df1 = df[mask] #test set
df2 = df[~mask] #train set

        #3.contrat       
df_contrat = (df['Contrat']
            .str.findall(r'(?:CDI|CDD|Stage|Freelance \/ Indépendant|Apprentissage \/ Alternance|Intérim)')
            .to_frame('Contract_type'))
df_contrat_time = (df['Contrat']
            .str.findall(r'(?:Temps plein|Temps partiel)')
            .to_frame('Contract_time'))
df = pd.concat([df, df_contrat,df_contrat_time], axis=1)
df.drop(['Contrat'], axis=1, inplace=True)

#np.nan
values = {'Contract_type': 'NS_type'}
df.fillna(value=values,inplace=True)
values = {'Contract_time': 'NS_time'}
df.fillna(value=values,inplace=True)

#empty list #d'abord supprimer les np.nan sinon ca marche pas !
df['Contract_time']  = df['Contract_time'].apply(lambda y: np.nan if len(y)==0 else y)
df['Contract_type']  = df['Contract_type'].apply(lambda y: np.nan if len(y)==0 else y)
df['Contract_time'].isna().sum()
df['Contract_type'].isna().sum()

#nan une seconde fois
values = {'Contract_type': 'NS_type'}
df.fillna(value=values,inplace=True)
values = {'Contract_time': 'NS_time'}
df.fillna(value=values,inplace=True)


#split list [] into dummies :
s1 = pd.Series(df['Contract_type'])
s2 = pd.Series(df['Contract_time'])
dummie1 = pd.get_dummies(s1.apply(pd.Series).stack()).sum(level=0)
dummie2 = pd.get_dummies(s2.apply(pd.Series).stack()).sum(level=0)
dummie1.isna().sum()
dummie2.isna().sum()
df = pd.concat([df, dummie1, dummie2], axis=1)
df.drop(['Contract_time','Contract_type'], axis=1, inplace=True)

        #Date #30 = 30+ j
df_date = (df['Date']
            .str.replace(" ", "")
            .str.findall(r'(\d+\.?)')
            .apply(lambda x: pd.Series(x).astype(float))
            .mean(1)
            .to_frame('date_integer')) #nom col.
df_date['per_date'] = df['Date'].str.extract(r'([A-Za-z])')

df_date.loc[df_date['per_date'] == 'h', 'date_integer' ] = df_date['date_integer'] / 24
df_date.loc[df_date['per_date'] == 'm', 'date_integer' ] = df_date['date_integer'] * 30
df_date.loc[df_date['per_date'] == 'w', 'date_integer' ] = df_date['date_integer'] * 7
#https://stackoverflow.com/questions/36909977/update-row-values-where-certain-condition-is-met-in-pandas
df_date.rename(columns={'date_integer':'since_x_days'},inplace=True)                    
df_date.drop(['per_date'], axis=1, inplace=True)
df = pd.concat([df,df_date], axis=1)
df.drop(['Date'], axis=1, inplace=True)
#nan
df['since_x_days'].isna().sum()
df.hist(column='since_x_days')
df['since_x_days'].fillna((df['since_x_days'].mean()), inplace=True) #OK !!

        #Colonne Metier
df['Metier_x'].isna().sum()

        #diploma from descpost

df_Diplome = (df['Descpost']
            .astype(str)
            .str.replace(" ", "")
            .str.findall(r'(BAC\s*\+\s*\d|BTS|DUT|LICENCE|MASTER|DEUG|M1|M2)')
            .to_frame('Diplome'))

df_Diplome.Diplome.apply(sorted).transform(tuple).unique()


#BAC+ Integer
df_nbre_annee = (df_Diplome['Diplome']
            .astype(str)
            .str.replace(" ", "")
            .str.findall(r'BAC\s*\+(\d)')
            .apply(lambda x: pd.Series(x).astype(float))
            .min(1) #Moyenne si intervalle
            .to_frame('nbre_annee'))


df_diplome = (df_Diplome['Diplome']
            .astype(str)
            .str.replace(" ", "")
            .str.findall(r'(BTS|DUT|LICENCE|MASTER|DEUG|M1|M2)')
            .to_frame('type_diplome'))
df_diplome.type_diplome.apply(sorted).transform(tuple).unique()
#extraire le nombre
mapping = {"M2":5, "M1":4, "MASTER" : 5, "LICENCE" :3 , "DEUG" :2 , "BTS":2 , "DUT" :2}
df_diplome['nbre_annee']=df_diplome['type_diplome'].apply(lambda x : [mapping[y] for y in x]) #x =liste, #y=valaur de la list
#extraire le min
df_diplome['nbre_annee'] = df_diplome['nbre_annee'].apply(lambda x : pd.Series(x).min()) # pd.Series(x) à rajouter !! OK !!
df_diplome.drop(['type_diplome'], axis=1, inplace=True)


#merge df_diplome et df_nbre_anne                     #nbre_anne_x = df_diplome
df_merged = df_diplome.merge(df_nbre_annee, how='outer',#nbre_annee_y = df_nbre_annee
                             left_index=True, right_index=True) 
df_merged.hist(column='nbre_annee_x')
df_merged.hist(column='nbre_annee_y')
#nan dans df group by métier
df = pd.concat([df,df_merged], axis=1)
df.hist(column='nbre_annee_x', by = 'Metier_x')
df.hist(column='nbre_annee_y', by = 'Metier_x')
df['nbre_annee_x'] = df.groupby('Metier_x')['nbre_annee_x'].transform(lambda x: pd.Series(x).fillna(x.value_counts().index[0]))
df['nbre_annee_y'] = df.groupby('Metier_x')['nbre_annee_y'].transform(lambda x: pd.Series(x).fillna(x.value_counts().index[0]))
#mettre ['nbre_annee_y'] après df.groupby('Metier_x') !!!!
df['nbre_annee_x'].isna().sum()
df['nbre_annee_y'].isna().sum()

        #Notes
        
df.info()      
#as float
df["NoteGlobale"].astype(int)
  
df["Avis"].isna().sum()  #Out[269]: 6826
df["NoteGlobale"].isna().sum()  #Out[270]: 6826
df["Équilibre_vie_professionnelle_personnelle"].isna().sum() #Out[271]: 6831
df["Salaire_Avantages_sociaux"].isna().sum()  #Out[272]: 6831
df["Sécurité_emploi_Évolution_carrière"].isna().sum()  #Out[273]: 6831
df["Management"].isna().sum()    #Out[274]: 6831
df["Culture_d'entreprise"].isna().sum()  #Out[275]: 6831
df.drop(['Avis','NoteGlobale',"Équilibre_vie_professionnelle_personnelle",
         "Salaire_Avantages_sociaux","Sécurité_emploi_Évolution_carrière",
         "Management","Culture_d'entreprise"], axis=1, inplace=True)





#############################################################################
                    #RANDOM FOREST
                    

df_num =df[['Salaire_annuel','since_x_days','nbre_annee_x',
                    'nbre_annee_y']] 
# Descriptive statistics for each column
round(df[['Salaire_annuel','since_x_days','nbre_annee_x',
                    'nbre_annee_y']] .describe(), 2)  

#get dummies
df.drop(['Descpost'], axis=1, inplace=True)
df=pd.get_dummies(df,drop_first=True) #COLINEARITE :drop_first=True
df.info()
    
#fix label and features
labels = df['Salaire_annuel']
features= df.drop('Salaire_annuel', axis = 1)


# Convert to numpy array if necessary
#labels = np.array(features['actual'])  
#features = np.array(features)
    
    #Split data
    
#train et test set pour les salaires.
mask = df['Salaire_annuel'].isna()
test_set = df[mask] #test set
train_set = df[~mask] #train set

train_features = train_set.drop('Salaire_annuel', axis = 1) 
train_labels= train_set['Salaire_annuel'] 

test_features = test_set.drop('Salaire_annuel', axis = 1) 
test_labels= test_set['Salaire_annuel'] 
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_features, #on split le train
                                train_labels, test_size = 0.25, random_state = 42) 

from sklearn.ensemble import  RandomForestRegressor
model = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Fit on training data
model.fit(X_train, y_train)
y_pred_train_set = model.predict(X_test) #... OK !!!

#il faut faire un score avant de de faire .predict(test_features) 
from sklearn.metrics import mean_squared_log_error
score =  np.sqrt(mean_squared_log_error(y_pred_train_set,y_test))
score
#Out[383]: 0.5552535630659917

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_train_set))
#0.0058103930675675874

################################################################################
  #Améliorer le score : Variable Importances
fi = pd.DataFrame({'feature': list(train_features.columns),
                   'importance': model.feature_importances_}).sort_values('importance', ascending = False)
   
fi
      
 # New random forest 2 most important variables 
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

train_features = train_set[['since_x_days','Adresse_Paris (75)','Metier_x_data_scientist','nbre_annee_x','Adresse_Pessac (33)']] 
train_labels= train_set['Salaire_annuel'] 

test_features = test_set[['since_x_days','Adresse_Paris (75)', 'Metier_x_data_scientist','nbre_annee_x','Adresse_Pessac (33)']] 
test_labels= test_set['Salaire_annuel'] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_features, #on split le train
                                train_labels, test_size = 0.25, random_state = 42)

# Train the random forest on the train set
rf_most_important.fit(X_train, y_train)
predictions = rf_most_important.predict(X_test)

#il faut faire un score avant de de faire .predict(test_features) 
from sklearn.metrics import mean_squared_log_error
score =  np.sqrt(mean_squared_log_error(predictions,y_test))
score
#Out[399]: 0.5635125697577884

from sklearn.metrics import r2_score
print(r2_score(y_test,predictions))
#0.16894840671056333
###################################################################################



#dernière étape: prédire avec la premier essai
y_pred_test_set = model.predict(test_features)  #predict sur le test set
y_pred_test_set

print(model.score(test_features,y_pred_test_set))
#1.0



################################################################################"


train_features = train_set.drop('Salaire_annuel', axis = 1) 
train_labels= train_set['Salaire_annuel'] 

test_features = test_set.drop('Salaire_annuel', axis = 1) 
test_labels= test_set['Salaire_annuel'] 
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_features, #on split le train
                                train_labels, test_size = 0.25, random_state = 42) 


from sklearn.svm import SVR
regressor=SVR(kernel='rbf',epsilon=1.0)
regressor.fit(X_train,y_train)
pred_train=regressor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,pred_train))
#-0.004122618626146934

pred_test=regressor.predict(test_features)
print(regressor.score(test_features,pred_test))
#1.0












##############################################################################"
    #10. Arbre ou forest representation
    
from sklearn.tree import export_graphviz
import pydot
    # Pull out one tree from the forest
tree = rf.estimators_[5]
    # Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
graph.write_png('tree.png')
    
    # Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
    # Extract the small tree
tree_small = rf_small.estimators_[5]
    # Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');
 
###############################################################################
                    #RANDOM FOREST 2 : 
    #Visualization bar plot most important variable
    
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
    
 
    #visualization en série temporelle 
    
# Use datetime for creating date objects for plotting
import datetime
# Dates of training values
months = features.iloc[:, feature_list.index('month')] #mettre .iloc feature est un df
days = features.iloc[:, feature_list.index('day')]
years = features.iloc[:, feature_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions
months = test_features.iloc[:, feature_list.index('month')] 
days = test_features.iloc[:, feature_list.index('day')]
years = test_features.iloc[:, feature_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})

# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values'); 
    
    #Visualization avec amplitude des variables par rapport à la moyenne
    
# Make the data accessible for plotting
true_data['temp_1'] = features.iloc[:, feature_list.index('temp_1')]
true_data['average'] = features.iloc[:, feature_list.index('average')]
true_data['friend'] = features.iloc[:, feature_list.index('friend')]

# Plot all the data as lines
plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
plt.plot(true_data['date'], true_data['temp_1'], 'y-', label  = 'temp_1', alpha = 1.0)
plt.plot(true_data['date'], true_data['average'], 'k-', label = 'average', alpha = 0.8)
plt.plot(true_data['date'], true_data['friend'], 'r-', label = 'friend', alpha = 0.3)
# Formatting plot
plt.legend(); plt.xticks(rotation = '60');
# Lables and title
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual Max Temp and Variables');
    

##################################LDA###########################################
        
        #Descposte
import re, nltk, spacy, gensim
# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
%matplotlib inline

df['Descpost'].isna().sum()


data = df.Descpost.tolist()
# Remove Emails
data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]
#supprime les sauts de lignes
data = [re.sub(r'\s+', ' ', sent) for sent in data]
#supprime apostrophes
data = [re.sub(r"\'", "", sent) for sent in data]
print(data[:1])

    #1.Tokenize  
#list of words : removing punctuations and unnecessary characters 
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
print(data_words[:1]) 

    #2.lemmatization = roots of words
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): 
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner']) # Initialize spacy 'en' model # Run in terminal: python -m spacy download en, #explications:https://spacy.io/usage/processing-pipelines
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'VERB']) 
print(data_lemmatized[:2])#take times !!!!

    #3. Create the Document-Word matrix
#CountVectorizer =  words that has occurred at least 10 times
#remove built-in french stopwords 
#convert all words to lowercase
#word can contain numbers, alphabets of at least length 3 
from spacy.lang.fr.stop_words import STOP_WORDS #francais

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,
# minimum reqd occurences of a word 
                             stop_words=list(STOP_WORDS),  #ici           
# remove stop words
                             lowercase=True,                   
# convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  
# num chars > 3
                             max_features=50000),             
# max number of uniq words    
data_vectorized = vectorizer.fit_transform(data_lemmatized) #problem type tuple ....écrire "vectorizer[0]"
print(data_vectorized.toarray())  


    #4.Build LDA.fit
lda_model = LatentDirichletAllocation(n_components=20,  # set Number of topics
                                      max_iter=10,               
# Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          
# Random state
                                      batch_size=128,            
# n docs in each learning iter
                                      evaluate_every = -1,       
# compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               
# Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized) #takes times...
print(lda_model)  #print the  Model attributes
###############################################################################
    #4bis. Diagnose model performance with perplexity and log-likelihood
print("Log Likelihood: ", lda_model.score(data_vectorized)) #Higher the better
#out : Log Likelihood:  -8 509 466.557993239
print("Perplexity: ", lda_model.perplexity(data_vectorized)) #Lower the better. Perplexity = exp(-1. * log-likelihood per word)
#out: Perplexity:  1 039.767935888455
print(lda_model.get_params()) #print the lda_paramètres
###############################################################################"

    #5. Use Grid-Search.fit & .best_estimator_ to have the best LDA model.n_components  ?
# Define Grid-Search Param
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
lda = LatentDirichletAllocation(max_iter=5, learning_method='online', 
                                learning_offset=50., random_state=0)

model = GridSearchCV(lda, param_grid=search_params)
model.fit(data_vectorized) #takes time !

#Grid-Search constructs multiple LDA models for all possible combinations of param values
GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
             n_topics=None, perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0), 
            fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]}, #learnin_decay = learning rate entre 0.5 et 1
       pre_dispatch='2*n_jobs', 
       refit=True, 
       return_train_score='warn',
       scoring=None, 
       verbose=0)

best_lda_model = model.best_estimator_
print("Best Model's Params: ", model.best_params_)
#Out: Best Model's Params:  {'learning_decay': 0.7, 'n_components': 10}

#############################################################################
print("Best Log Likelihood Score: ", model.best_score_)
#out: Best Log Likelihood Score:  -2 941 380.969213958
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))     
#out Model Perplexity:  1 016.3610577087106
###############################################################################

    #6.Distribution de probabilité des numéros de topics puis topic dominant

# dataframe lda_output
lda_output = best_lda_model.transform(data_vectorized)
#numéroter les topics et les doc
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
docnames = ["Doc" + str(i) for i in range(len(data))]
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
# Get dominant topic in new column 
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic


    #7. Distribution score entre topic et mots clés
df_topic_keywords = pd.DataFrame(best_lda_model.components_)
# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()  
df_topic_keywords.index = topicnames

    # list : n_words=20 keywords
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20)#top 20 keywords 

    # Dataframe (10,20)
df_topic_keywords = pd.DataFrame(topic_keywords) #to dataframe
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords

    # rename 
Topics = ["transport aérien et drône",
          "stage et formation",
          "software, ingeneering, innovation",
          "projet developpeur", 
          "secteur des technologies",
          "programmeur devop",
          "wordpress gestion administration", 
          "dev front web plateforme",
          "business solution customer",
          "insurance"]
df_topic_keywords["Topics"]=Topics
df_topic_keywords


#RESULT 2/2 : concat into the df
#df_topic_number=df_document_topic.iloc[:,-1] 
#df = pd.concat([df,df_topic_number], axis=1)
hist = df_topic_number['dominant_topic'].hist(bins=3)

#problème : rajoute lignes supplémantaire et des nan !

##########################################################################################


    
    