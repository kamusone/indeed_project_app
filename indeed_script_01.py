
###############################################################################
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd
import numpy as np
from threading import Thread, RLock
import time
import re
import math

###############################################################################
##une fonction qui génère un temps d'attente selon la loi normal 
def generate_delay():
        mean = 1
        sigma = 0.4
        return math.fabs(np.random.normal(mean,sigma,1)[0])
    
###############################################################################
##return les offres de la page
def get_nb_items(browser):
    return browser.find_elements_by_class_name('summary')

###############################################################################
##clean salaire
def get_salaire(text):
    is_salaire = re.compile(r'\d+')
    if is_salaire.match(text):
        return text
    else:
        return np.nan
    
###############################################################################
##transformer les salaires min en salaires annuels
#s = salaire
#l_s = liste salaires
#s_a_min = salaire annuel minimum
       
def get_salaire_min_annuel(s):
    s = str(s)
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    s = s.replace("€","").replace("-"," ").replace(",",".")
    l_s = [float(sal) for sal in s.split() if sal.replace(".","").isdecimal()] 
    if "par an" in s:
        return l_s[0]
    elif "par mois" in s: 
        return l_s[0]*12
    elif "par semaine" in s: 
        return l_s[0]*52
    elif "par jour" in s: 
        return l_s[0]*(365-125)
    elif "par heure" in s: 
        return l_s[0]*8*(365-125)
    else:
        return np.nan


###############################################################################
##transformer les salaires max en salaires annuels
#s_a_min = salaire annuel maximum
    
def get_salaire_max_annuel(s):
    s = str(s)
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    s = s.replace("€","").replace("-"," ").replace(",",".")
    l_s = [float(sal) for sal in s.split() if sal.replace(".","").isdecimal()]  
    if "par an" in s:
        s_a_max = l_s[1] if len(l_s)>1 else l_s[0] 
    elif "par mois" in s: 
        s_a_max = l_s[1]*12 if len(l_s)>1 else l_s[0]*12
    elif "par semaine" in s: 
        s_a_max = l_s[1]*52 if len(l_s)>1 else l_s[0]*52
    elif "par jour" in s: 
        s_a_max = l_s[1]*(365-125) if len(l_s)>1 else l_s[0]*(365-125)
    elif "par heure" in s: 
        s_a_max = l_s[1]*8*(365-125) if len(l_s)>1 else l_s[0]*8*(365-125)
    else:
        s_a_max = np.nan
    return s_a_max

###############################################################################
##transformer les salaires max en salaires annuels 

def get_salaire_moyen_annuel(s):
    s = str(s)
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    s = s.replace("€","").replace("-"," ").replace(",",".")
    salaire = np.mean([float(sal) for sal in s.split() if sal.replace(".","").isdecimal()])
    if "par an" in s:
        return salaire
    elif "par mois" in s: 
        return salaire*12
    elif "par semaine" in s: 
        return salaire*52
    elif "par jour" in s: 
        return salaire*(365-125)
    elif "par heure" in s: 
        return salaire*8*(365-125)
    else:
        return np.nan
    
###############################################################################
#Clean contrat

def get_contrat(text):
    if any(i.isdigit() for i in text):
        return np.nan
    else:
        return text

###############################################################################
##Clean date
def get_date(text):
    dates = re.findall('\d+\s\w{1}|\d+\+\s\w{1}', text)
    if (len(dates)>0):
        return dates[0]
    else:
        return text

###############################################################################
##Clean adresse

def get_adress(text):
    return text.replace('-','').lstrip().upper()
     
###############################################################################
##fermer popup

def block_popup(browser):
    popup = browser.find_element_by_xpath("""//*[@id="popover-close-link"]""")
    popup.click()
    time.sleep(generate_delay())

###############################################################################
##Vérifier si la page contient "suivant"
def is_contain_next(browser):
    pagenext = browser.find_elements_by_class_name('np')
    return 'Suivant' in pagenext[0].text or 'Next' in pagenext[0].text

###############################################################################
##Save dans mongodb sans doublon
def save_in_mongodb(job):
    try:
        conn=MongoClient('mongodb://localhost:27017')
    except ConnectionFailure as e:
        print("Could not connect to MongoDB: %s" % e)
    db = conn['Indeed_DataBase']
    Indeed = db.Indeed
    if not Indeed.find_one(job):
        Indeed.insert_one(job)
        
        
    #Indeed.replace_one(job, job, upsert=True)
        
############################################################################################################################################
##le fonction qui scrape et renvoi des données sous un dictionnaire
def Get_scraping_data(browser):
    try:
        jobtitle = browser.find_element_by_xpath('//*[@id="vjs-jobtitle"]').text
    except NoSuchElementException:                  
        jobtitle = np.nan
        
    try:
        nom_boite = browser.find_element_by_xpath('//*[@id="vjs-cn"]').text
    except NoSuchElementException:
        nom_boite = np.nan
        
    try:
        adresse =  browser.find_element_by_xpath('//*[@id="vjs-loc"]').text
        adresse = get_adress(adresse)
    except NoSuchElementException:
        adresse = np.nan
        
    try:
        salaire = browser.find_element_by_css_selector('#vjs-jobinfo > div:nth-child(3) > span:nth-child(1)').text
        salaire = get_salaire(salaire)
        salaire_min = get_salaire_min_annuel(salaire)
        salaire_mean = get_salaire_moyen_annuel(salaire)
        salaire_max = get_salaire_max_annuel(salaire)   
    except NoSuchElementException:
        salaire = np.nan
        salaire_min = np.nan
        salaire_mean = np.nan
        salaire_max = np.nan
        
    try:                                              
        contrat = browser.find_element_by_css_selector('#vjs-jobinfo > div:nth-child(3) > span:nth-child(1)').text 
        contrat = get_contrat(contrat)
        if contrat == np.nan:
            contrat = browser.find_element_by_css_selector('#vjs-desc > p:nth-child(15)').text
            contrat = get_contrat(contrat)
    except NoSuchElementException:
        contrat = np.nan
        
    try:
        desc_poste = browser.find_element_by_xpath('//*[@id="vjs-desc"]').text
    except NoSuchElementException:
        desc_poste = np.nan
        
    try:
        date = browser.find_element_by_id('vjs-footer').text
        date = get_date(date)
    except NoSuchElementException:
        date = np.nan
        
        
    return({ "Titre":jobtitle, "Nom":nom_boite, "Adresse":adresse, "Salaire":salaire, 
             'Salaire_annuel_min':salaire_min , 'Salaire_annuel_moyen':salaire_mean, 
             'Salaire_annuel_max':salaire_max, 'Contrat':contrat, "Descposte":desc_poste, "Date":date})

############################################################################################################################################
##parcourir toutes les pages et appeler la fonction de scaping
def scrap_data(url):
    browser = webdriver.Firefox()
    browser.get(url)
    browser.maximize_window()
    time.sleep(3)
    while True:
        i = 0
        while i < len(get_nb_items(browser)):
            liste_emplois = get_nb_items(browser)
            liste_emplois[i].click()
            time.sleep(generate_delay())
            job_info = Get_scraping_data(browser)
            if not job_info:
                pass
            else:
                save_in_mongodb(job_info)
            time.sleep(generate_delay())
            i+=1
        pagenext = browser.find_elements_by_class_name('np')   
        if len(pagenext) > 1:
            pagenext[1].click()
            time.sleep(generate_delay())
            try :
                block_popup(browser)
            except :
                pass
        elif len(pagenext) == 1 and is_contain_next(browser):
            pagenext[0].click()
            time.sleep(generate_delay())
            try :
                block_popup(browser)
            except :
                pass 
        else:
            print("The last page")
            break
        
###############################################################################
##function global qui appelle toutes les fonctions
def global_fuction(job, city):
    lock = RLock()
    threads = []
    with lock:
        url = "http://www.indeed.fr/emplois?q=%s&l=%s" % (job, city)
        t = Thread(target=scrap_data, args=(url,))
        threads.append(t)
        t.start()
        time.sleep(1)

