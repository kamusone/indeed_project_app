from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import defaultdict
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import pandas as pd
import numpy as np
import re
import csv

##############################################################################################
##importer les données indeed de la base mongodb
def import_data_from_mongodb():
    try:
        client=MongoClient('mongodb://localhost:27017')
    except ConnectionFailure as e:
        print("Could not connect to MongoDB: %s" % e)
    db = client.Indeed_DataBase
    Indeed = db.Indeed
    Indeed = pd.DataFrame(list(Indeed.find()))
    return Indeed


##############################################################################################

langages=["PYTHON","C\s?\+\+","C\s?\#","VBA",
          "MATLAB","SPARK","GITHUB","AIRFLOW","POSTGRES","EC2","EMR","RDS",
          "REDSHIFT","AZURE","AWS","JAVA","HTML","CSS","JS","PHP","SQL",
          "RUBY","SAS","TCL","PERL","ORACLE","MYSQL","MONGODB","REDIS",
          "NEO4J","TWITTER","LDAP","FILE","HADOOP","DB2","SYBASE","AS400",
          "ACCESS","FLASK","BOOTSTRAP"]

##############################################################################################

diplomes = ['BTS', 'DUT', 'LICENCE', 'MASTER', 'DEUG', 'BAC\s?\+\s?1', 
            'BAC\s?\+\s?2', 'BAC\s?\+\s?3', 'BAC\s?\+\s?4', 'BAC\s?\+\s?5',
            'L\s?\3', 'M\s?1', 'M\s?2']

##############################################################################################

list_pattern = ["\s(\d)\sANS D’EXPERIENCE", "\s(\d)\sANS \(REQUIS\)", "\s(\d)\sAN \(REQUIS\)", 
                "\s(\d)\sANS \(SOUHAITE\)", "\s(\d)\sAN \(SOUHAITE\)", "\sEXPERIENCE DE\s(\d*)",
                "\sEXPERIENCE SIGNIFICATIVE DE\s(\d)", "\sEXPERIENCE PROFESSIONNELLE DE\s(\d)",
                "\sEXPERIENCE DE\s(\d)\sANNEE", "\sEXPERIENCE DE\s(\d)\sANNEES", "\sMOINS\s(\d)\sAN D'EXPERIENCE",
                "\sMOINS\s(\d)\sANNEES D'EXPERIENCE", "\sMOINS\s(\d)\sANNEE D'EXPERIENCE",
                "\s(\d)\sANNEES D'EXPERIENCE", "\s(\d)\sANNEE D'EXPERIENCE", "\s(\d)\sAN D’EXPERIENCE",
                "\s(\d)\sYEARS OF EXPERIENCE", "\s(\d)\sYEAR OF EXPERIENCE", "\sMOINS\s(\d)\sANS",
                "\sMOINS\s(\d)\sAN", "\/(\d)\sANS D’EXPERIENCE"]

##############################################################################################

list_debutant =["STAGE","STAGIAIRE","INTERNSHIP","TRAINEE", "ALTERNANCE"]

##############################################################################################

list_job = ['DATA SCIENTIST', 'DATA ANALYST', 'BUSINESS INTELLIGENCE', 'DEVELOPPEUR']

##############################################################################################

list_contrat= ['Apprentissage / Alternance', 'CDD', 'CDD, CDI', 'CDD, Freelance / Indépendant, CDI']

list_paris = ['PARIS', '75', '92', '93', '94', '95', '78', '77', '91',
              'ÎLEDEFRANCE', 'ILEDEFRANCE', 'HAUTSDESEINE', 'VALDEMARNE',
              'SEINESAINTDENIS']

##############################################################################################
##Clean adresse (location)

def return_ville(adresse):
    for ville in list_paris:
        if ville in adresse:
            return 'Paris'
        else:
            pass
    
    if '69' in adresse or 'AUVERGNERHÔNEALPES' in adresse or 'RHÔNE' in adresse or '01' in adresse \
                       or 'SAINTQUENTINFALLAVIER' in adresse:
        return 'Lyon'
    else:
        pass
    if '31' in adresse or 'HAUTEGARONNE' in adresse or 'OCCITANIE' in adresse:
        return 'Toulouse'
    else:
        pass
    if '44' in adresse or 'LOIREATLANTIQUE' in adresse or 'PAYS DE LA LOIRE' in adresse:
        return 'Nantes'
    else:
        pass
    if '33' in adresse or 'GIRONDE' in adresse or 'FRANCE' in adresse or 'NOUVELLEAQUITAINE' in adresse:
        return 'Bordeaux'
    else:
        pass
    
##############################################################################################    
##Labelisation de la variable ville

def add_ville_variable(df):
     for i in range(len(df.index)):
        adresse = str(df.loc[i,'Adresse'])
        ville = return_ville(adresse.upper())
        if ville:
            if ville== 'Paris':
                df.loc[i,'Ville'] = 5
            elif ville== 'Lyon':
                df.loc[i,'Ville'] = 4
            elif ville == 'Bordeaux' or ville == 'France':
                df.loc[i,'Ville'] = 3
            elif ville == 'Nantes':
                df.loc[i,'Ville'] = 2
            elif ville == 'Toulouse':
                df.loc[i,'Ville'] = 1
            else:
                df.loc[i,'Ville'] = 0       
     return df 

##############################################################################################    
##Return la ville

def add_ville(df):
     for i in range(len(df.index)):
        adresse = str(df.loc[i,'Adresse'])
        ville = return_ville(adresse.upper())
        if ville:
            if ville== 'Paris':
                df.loc[i,'Ville'] = 'Paris'
            elif ville== 'Lyon':
                df.loc[i,'Ville'] = 'Lyon'
            elif ville == 'Bordeaux' or ville == 'France':
                df.loc[i,'Ville'] = 'Bordeaux'
            elif ville == 'Nantes':
                df.loc[i,'Ville'] = 'Nantes'
            elif ville == 'Toulouse':
                df.loc[i,'Ville'] = 'Toulouse'
            else:
                df.loc[i,'Ville'] = np.nan      
     return df   
        
##############################################################################################
##Récupérer les langages et les technologies dans la description et labelisation de ces variables  

def add_lanages_variables(df):
    df['C'] = list(map(int, df["Descposte"].str.contains('[\s,.;](C)[\s*,.;!?]',regex=True, na=False)))
    df['R'] = list(map(int, df["Descposte"].str.contains('[\s,.;](R)[\s*,.;!?]',regex=True, na=False)))
    for langage in langages:
        df[langage] = list(map(int, df["Descposte"].str.contains(langage,regex=True, na=False)))
    
    df.rename(columns={ 'C\s?\+\+': 'langage C++',
                         'C\s?\#': 'langage C#'}, inplace=True)
                        
    return df

##############################################################################################
##Labelisation de la variable metier 

def add_job_variables(df):
     for job in list_job:
         df[job] = list(map(int, df["Titre"].str.contains(job,regex=True, na=False)))
                        
     return df
        
##############################################################################################           
##Récupération des diplomes dans la description
##Labelisation de la variable diplome 
def add_diplome_variables(df):
     for diplome in diplomes:
         df[diplome] = list(map(int, df["Descposte"].str.contains(diplome,regex=True, na=False)))
         df.rename(columns={'BAC\s?\+\s?1': 'BAC+1', 'BAC\s?\+\s?2': 'BAC+2', 
                             'BAC\s?\+\s?3': 'BAC+3', 'BAC\s?\+\s?4': 'BAC+4', 
                             'BAC\s?\+\s?5': 'BAC+5', 'L\s?\3': 'L3', 'M\s?1':'M1', 
                             'M\s?2': 'M2'}, inplace=True)
     return df 
      
##############################################################################################     
#return niveau selon le nombre d'années d'expériences récupérées dans la description
# Stagiaire débutant 0 année expérience
# Junior 1-2 années expériences
# Confirmé 3-4 années expériences
# Sénior 5-10 années expériences
# Expert >10 années expériences

def return_niveau_by_date(descreption):
    for pattern in list_pattern:
        code_niv = re.findall(pattern , descreption)
        if code_niv:
            code_niv = [x for x in code_niv if x]
            if (int(code_niv[0]) >= 1 and int(code_niv[0]) <= 2):
                return 'JUNIOR'
            elif (int(code_niv[0]) >= 3 and int(code_niv[0]) <= 4):
                return 'CONFIRME'
            elif (int(code_niv[0]) >= 5 and int(code_niv[0]) <= 10):
                return 'SENIOR'
            elif (int(code_niv[0]) > 10):
                return 'EXPERT'
            else:
                return 'DEBUTANT'
         
############################################################################################## 
##Labelisation de niveau récupéré par date
def add_niveau_variables_by_date(df,i):
    descp = str(df.loc[i,'Descposte'])
    level = return_niveau_by_date(descp.upper())
    if level:
        if level== 'STAGIAIRE':
            df.loc[i,'level'] = 1
        elif level== 'DEBUTANT':
            df.loc[i,'level'] = 2
        elif level == 'JUNIOR':
            df.loc[i,'level'] = 3
        elif level == 'CONFIRME':
            df.loc[i,'level'] = 4
        elif level == 'SENIOR':
            df.loc[i,'level'] = 5
        elif level == 'EXPERT':
            df.loc[i,'level'] = 6
        else:
            df.loc[i,'level'] = 0
    else:
        df.loc[i,'level'] = 0
            
    return df
                
############################################################################################## 
##Return niveau s'il est dans la description ou le titre
def return_niveau_text(descreption, titre):
    if 'JUNIOR' in descreption or 'JUNIOR' in titre:
        return 'JUNIOR'
    elif 'SENIOR' in descreption or 'SENIOR' in titre:
        return 'SENIOR'
    elif 'DEBUTANT' in descreption or 'DEBUTANT' in titre:
        return 'DEBUTANT'
    else:
        for niveau in list_debutant:
            if niveau in descreption or niveau in titre:
                return "STAGIAIRE"
        else:
            pass 
 
############################################################################################## 
##récupérer le niveau dans le titre et la description et faire appel à la fonction 
##add_niveau_variables_by_text 

def add_niveau_variables_by_text(df):
    for i in range(len(df.index)):
        titre = str(df.loc[i,'Titre'])
        descp = str(df.loc[i,'Descposte'])
        level = return_niveau_text(descp.upper(), titre.upper())
        if level:
            if level== 'STAGIAIRE':
                df.loc[i,'level'] = 1
            if level== 'DEBUTANT':
                df.loc[i,'level'] = 2
            elif level == 'JUNIOR':
                df.loc[i,'level'] = 3
            elif level == 'CONFIRME':
                df.loc[i,'level'] = 4
            elif level == 'SENIOR':
                df.loc[i,'level'] = 5
            else:
                df.loc[i,'level'] = 0
        else:
            add_niveau_variables_by_date(df,i)
            
    return df

############################################################################################## 
##Fonction qui appelle toutes les fonctions précédentes pour générer 
##un dataframe avec les variables ajoutées 

def add_variables(df):
    X = add_niveau_variables_by_text(df)
    X = add_ville_variable(X)
    X = add_job_variables(X)
    X = add_diplome_variables(X)
    X = add_lanages_variables(X)

    return X

##############################################################################################
##Ajouter la colonnes classe salaire qui nous indique si le salaire est faible ou haut, 
##Après avoircalculé la median de salaire annuel moyen

def return_features_model():
    df = import_data_from_mongodb()
    df = add_variables(df)
    df = df[~pd.isnull(df["Salaire_annuel_moyen"])]
    salaire_median = np.median(df["Salaire_annuel_moyen"])
    df["class_salaire"] = list(map((lambda x: "FAIBLE" if x < salaire_median else "HAUT"),
                                    df["Salaire_annuel_moyen"])) 
    df["label_class_salaire"] = df["class_salaire"].replace({"HAUT":1, "FAIBLE":0})
    return df 
 

############################################################################################## 

drop_columns = ['_id','Nom', 'Titre','Adresse', 'Salaire', 'Salaire_annuel_min', 
                'Salaire_annuel_max', 'Contrat', 'Descposte', 'Date', 
                'Salaire_annuel_moyen', 'class_salaire']

##############################################################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import FrenchStemmer
from sklearn.preprocessing import StandardScaler 

##Fonction qui renvoie la fréquence d'apparition des mots les plus communs ou les plus rares dans la Description

def vectorize_descreption():
    fs =FrenchStemmer()
    df = return_features_model()
    df['Descposte']=[fs.stem(k) for k in df['Descposte']]
    tfidf=TfidfVectorizer()
    tfidf.fit_transform(df['Descposte'])
    tfidf_col  = pd.DataFrame(tfidf.fit_transform(df['Descposte']).todense(),
                              columns=tfidf.get_feature_names())
    df= df.reset_index()
    df_final = pd.merge(df.drop(columns=drop_columns, axis = 1),
                        tfidf_col,right_index=True, left_index=True)
    return df_final



##############################################################################################
##Save ideed avec toutes les variables ajoutées
def save_indeed():
    df = import_data_from_mongodb()
    df = add_ville(df)
    df = add_job_variables(df)
    df = add_diplome_variables(df)
    df = add_lanages_variables(df)
    df.to_csv("C:/Users/Bouhafs/Documents/Ecole IA Microsoft/Indeed/Projet_Indeed/indeed.csv", index=False)
    return df

##############################################################################################
df = vectorize_descreption()
##############################################################################################
##Model RandomForest :
##Return accuracy_rfc

def get_accuracy_rfc():
    y = df["label_class_salaire"]
    X = df.drop(["label_class_salaire"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                         random_state=42)
    model = RandomForestClassifier(n_estimators=300, random_state=90)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_rfc =  accuracy_score(y_test, y_pred) * 100
    return accuracy_rfc

##############################################################################################
##Model SVM :
##Return accuracy_svm

def get_accuracy_svm():
    y = df["label_class_salaire"]
    X = df.drop(["label_class_salaire"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                         random_state=42)
    model = SVC(gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_svm =  accuracy_score(y_test, y_pred) * 100
    return accuracy_svm

##############################################################################################
##Save accuracy

def save_accuracy_csv():
    list_item =[]
    dic={}
    dic['accuracy_rfc'] = get_accuracy_rfc()
    dic['accuracy_svm'] = get_accuracy_svm()
    list_item.append(dic)
    url = "C:/Users/Bouhafs/Documents/Ecole IA Microsoft/Indeed/Projet_Indeed/accuracy.csv"
    csv_columns = ['accuracy_rfc', 'accuracy_svm']
    try:
        with open(url, 'w', encoding='utf-8',newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for dt in list_item:
                writer.writerow(dt)
    except IOError:
        print("I/O error")

##############################################################################################
##Save matrix_confusion RF

def save_confusion_matrix_csv():
    y = df["label_class_salaire"]
    X = df.drop(["label_class_salaire"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                         random_state=42)
    model_rfc = RandomForestClassifier(n_estimators=300, random_state=90)
    model_rfc.fit(X_train, y_train)
    y_pred_rfc= model_rfc.predict(X_test)
    confusion_matrix_rfc = confusion_matrix(y_test,y_pred_rfc)
    df_rfc = pd.DataFrame(data=confusion_matrix_rfc, columns=['Predected No', 'Predected Yes'])
    df_rfc.index=['Actual No', 'Actual Yes']
    df_rfc.to_csv("C:/Users/Bouhafs/Documents/Ecole IA Microsoft/Indeed/Projet_Indeed/confusion_matrix.csv") 

##############################################################################################
##Save matrix_confusion RF

def save_confusion_matrix_svm_csv():
    y = df["label_class_salaire"]
    X = df.drop(["label_class_salaire"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                         random_state=42)
    model = SVC(gamma='scale')
    model.fit(X_train, y_train)
    y_pred_svm = model.predict(X_test)
    confusion_matrix_svm = confusion_matrix(y_test,y_pred_svm)
    df_rfc = pd.DataFrame(data=confusion_matrix_svm, columns=['Predected No', 'Predected Yes'])
    df_rfc.index=['Actual No', 'Actual Yes']
    df_rfc.to_csv("C:/Users/Bouhafs/Documents/Ecole IA Microsoft/Indeed/Projet_Indeed/confusion_matrix_svm.csv")


##############################################################################################
##Fonction qui renvoi un dictionnaire pour le transformer en DataFrame
target_names = ['class 0', 'class 1']

def report2dict(cr):
    
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    measures = tmp[0]
    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data

##############################################################################################
##Save classification_report random forest

def save_classification_report_rfc_csv():
    y = df["label_class_salaire"]
    X = df.drop(["label_class_salaire"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                         random_state=42)
    model_rfc = RandomForestClassifier(n_estimators=300, random_state=90)
    model_rfc.fit(X_train, y_train)
    y_pred_rfc = model_rfc.predict(X_test)
    report = pd.DataFrame(report2dict(classification_report(y_test, y_pred_rfc, target_names=target_names))).T
    report.to_csv("C:/Users/Bouhafs/Documents/Ecole IA Microsoft/Indeed/Projet_Indeed/classification_rfc_report.csv") 

##############################################################################################
##Save classification_report svm

def save_classification_report_svm_csv():
    y = df["label_class_salaire"]
    X = df.drop(["label_class_salaire"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                         random_state=42)
    model = SVC(gamma='scale')
    model.fit(X_train, y_train)
    y_pred_svm = model.predict(X_test)
    report = pd.DataFrame(report2dict(classification_report(y_test, y_pred_svm, target_names=target_names))).T
    report.to_csv("C:/Users/Bouhafs/Documents/Ecole IA Microsoft/Indeed/Projet_Indeed/classification_svm_report.csv") 


##############################################################################################
##Fonction d'envoi de mail
def send_mail():
    fromaddr = "bouhafsbachir@gmail.com"
    password = "Salut01011982"
    toaddr = "bouhafsbachir@gmail.com"
    # instance of MIMEMultipart 
    msg = MIMEMultipart() 
    # storing the receivers email address
    msg['To'] = toaddr 

    # storing the subject
    msg['Subject'] = "Test"

    # string to store the body of the mail 
    body = "Bouh"

    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 

    # open the file to be sent
    filename = "rapport_indeed.html"
    attachment = open("C:/Users/Bouhafs/Documents/Ecole IA Microsoft/Indeed/Projet_Indeed/rapport_indeed.html", "rb") 

    # instance of MIMEBase and named as p 
    p = MIMEBase('application', 'octet-stream') 

    # To change the payload into encoded form 
    p.set_payload((attachment).read()) 

    # encode into base64 
    encoders.encode_base64(p) 

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 

    # attach the instance 'p' to instance 'msg' 
    msg.attach(p) 

    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 

    # start TLS for security 
    s.starttls() 

    # Authentication 
    s.login(fromaddr, password) 

    # Converts the Multipart msg into a string 
    text = msg.as_string() 

    # sending the mail 
    s.sendmail(fromaddr, toaddr, text) 

    # terminating the session 
    s.quit()








