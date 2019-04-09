from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models import LabelSet, widgets
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, Paragraph, Div
from bokeh.transform import cumsum
from math import *
import pandas as pd
import numpy as np


output_file("C:/Users/Bouhafs/Documents/Ecole IA Microsoft/Indeed/Projet_Indeed/rapport_indeed.html")

##Fonction de visualisation

def show_shar():

    df = pd.read_csv('indeed.csv')
    df = df[~pd.isnull(df["Salaire_annuel_moyen"])]
    ##############################################################################################

    df_ville = df.groupby('Ville')['Salaire_annuel_moyen'].median()
    source1 = ColumnDataSource(pd.DataFrame(df_ville))
    ville = source1.data['Ville'].tolist()
    salaire = source1.data['Salaire_annuel_moyen'].tolist()
    
    ##############################################################################################

    df_job= df[['DATA SCIENTIST','DATA ANALYST', 'BUSINESS INTELLIGENCE', 'DEVELOPPEUR']].sum()
    df_job = pd.DataFrame({'Job':df_job.index, 'count_job':df_job.values})
    job = df_job['Job']
    count_job = (df_job['count_job']/df_job['count_job'].sum())*100

    ##############################################################################################

    df_langage= df[["C", "R","PYTHON", "langage C++", "langage C#", "VBA", "MATLAB",
                    "SPARK","GITHUB","AIRFLOW","POSTGRES","EC2","EMR","RDS",
                    "REDSHIFT","AZURE","AWS","JAVA","HTML","CSS","JS","PHP","SQL",
                    "RUBY","SAS","TCL","PERL","ORACLE","MYSQL","MONGODB","REDIS",
                    "NEO4J","TWITTER","LDAP","FILE","HADOOP","DB2","SYBASE","AS400",
                    "ACCESS","FLASK","BOOTSTRAP"]].sum()

    ##############################################################################################

    div1 = Div(text="""<h1>Projet Indeed</h1>
                        <h2>Ce projet a été développé en trois phases:</h2>
                          <ul>
                              <li>Collection des données en scrapant le site indeed.</li>
                              <li>Traitement de ces données (préprocessing).</li>
                              <li>Application des models RandomForest et SVM pour predire les salaires.</li>
                          </ul>
                          <h2>Etude Statistique:</h2>
                       """,
                width=800, height=180)

    div2 = Div(text="""
                        <h2>Etude Model Random Forest:</h2>
                       """,
                width=300, height=30)
    

    div3 = Div(text="""
                        <h3>Confusion matrix:</h3>
                       """,
                width=300, height=20)

    div4 = Div(text="""
                        <h2>Etude Model SVM:</h2>
                       """,
                width=300, height=30)

    div5 = Div(text="""
                        <h3>Confusion matrix:</h3>
                       """,
                width=300, height=20)

    div6 = Div(text="""
                        <h3>Classification report rfc:</h3>
                       """,
                width=300, height=20)

    div7 = Div(text="""
                        <h3>Classification report svm:</h3>
                       """,
                width=300, height=20)
######################################################################################
                    
    df_job = pd.DataFrame({'langage':df_langage.index, 'count_langage':df_langage.values})
    langage = df_job['langage']
    count_langage = (df_job['count_langage']/df_job['count_langage'].sum())*100

    ##############################################################################################
    
    df_accurancy = pd.read_csv('accuracy.csv')
    accuracy_rfc = round(float(df_accurancy['accuracy_rfc'][0]), 2)
    accuracy_svm = round(float(df_accurancy['accuracy_svm'][0]), 2)

    ##############################################################################################

    p1 = figure(x_range=ville, plot_height=300, plot_width=400, title="Le salaire moyen annuel par ville",
           toolbar_location=None, tools="",  
           x_axis_label = "Ville", y_axis_label = "Salaire")
    p1.vbar(x=ville, top=salaire, width=0.2, color='deepskyblue')
    p1.xgrid.grid_line_color = None
    p1.y_range.start = 0
    p1.title.text_font_size = "25px"

    ##############################################################################################

    p2 = figure(x_range=job, plot_height=300, plot_width=650, title="Le pourcentage des métiers demandés",
           toolbar_location=None, tools="",
           x_axis_label = "Métier", y_axis_label = "pourcentage")
    p2.vbar(x=job, top=count_job, width=0.2)
    p2.xgrid.grid_line_color = None
    p2.y_range.start = 0
    p2.title.text_font_size = "25px"
    ################################################################################################

    p3 = figure(y_range=langage, plot_width=600, plot_height=900, title = 'Le pourcentag des langages demandés',
                x_axis_label='Pourcentage', y_axis_label='Langage', tools=""
    )
    p3.hbar(y=langage, right=count_langage, height=0.3, color='mediumblue')
    p3.title.text_font_size = "25px"
    ################################################################################################
    ###Random Forest
    chart_colors = ['#44e5e2', '#e29e44', '#e244db',
                    '#d8e244', '#eeeeee', '#56e244', '#007bff', 'black']
    
    

    x_rfc = { 'True values': round(accuracy_rfc) , 'False values': round(100 - accuracy_rfc)}

    data = pd.Series(x_rfc).reset_index(name='value').rename(columns={'index':'accuracy_rfc'})
    data['angle'] = data['value']/sum(x_rfc.values())*2*pi
    data['color'] = chart_colors[:len(x_rfc)]


    p4 = figure(plot_height=350, title="Accuracy", toolbar_location=None,
        tools="hover", tooltips="@accuracy_rfc: @value")

    p4.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend='accuracy_rfc', source=data)
    p4.title.text_font_size = "25px"

    data["value"] = data['value'].astype(str)
    data["value"] = data["value"].str.pad(35, side = "left")
    source = ColumnDataSource(data)

    labels = LabelSet(x=0, y=1, text='value', level='glyph',
        angle=cumsum('angle', include_zero=True), source=source, render_mode='canvas')

    p4.add_layout(labels)

    ################################################################################################
    ###confusion_matrix Random Forest

    df_confusion_matrix = pd.read_csv('confusion_matrix.csv')
    source = ColumnDataSource(df_confusion_matrix)
    columns = [
    widgets.TableColumn(field=c, title=c) for c in df_confusion_matrix.columns
     ]
    data_matrix_rfc = DataTable(source=source, columns=columns,width=400, height=100)

    ################################################################################################
    ###classification_report Random Forest

    df_classification_report_rfc = pd.read_csv('classification_rfc_report.csv')
    source = ColumnDataSource(df_classification_report_rfc)
    columns = [
    widgets.TableColumn(field=c, title=c) for c in df_classification_report_rfc.columns
     ]
    data_repport_rfc = DataTable(source=source, columns=columns,width=800, height=200)

    #################################################################################################
    ###SVM
    x_svm = { 'True values': round(accuracy_svm) , 'False values': round(100 - accuracy_svm)}

    data = pd.Series(x_svm).reset_index(name='value').rename(columns={'index':'accuracy_svm'})
    data['angle'] = data['value']/sum(x_svm.values())*2*pi
    data['color'] = chart_colors[:len(x_svm)]


    p5 = figure(plot_height=350, title="Model SVM", toolbar_location=None,
        tools="hover", tooltips="@accuracy_svm: @value")

    p5.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend='accuracy_svm', source=data)
    p5.title.text_font_size = "25px"

    data["value"] = data['value'].astype(str)
    data["value"] = data["value"].str.pad(35, side = "left")
    source = ColumnDataSource(data)

    labels = LabelSet(x=0, y=1, text='value', level='glyph',
        angle=cumsum('angle', include_zero=True), source=source, render_mode='canvas')

    p5.add_layout(labels)

    ################################################################################################
    ###confusion_matrix SVM

    df_confusion_matrix_svm = pd.read_csv('confusion_matrix_svm.csv')
    source = ColumnDataSource(df_confusion_matrix_svm)
    columns = [
    widgets.TableColumn(field=c, title=c) for c in df_confusion_matrix_svm.columns
     ]
    data_matrix_svm = DataTable(source=source, columns=columns,width=400, height=100)

    ################################################################################################
    ###classification_report SVM

    df_classification_report_svm = pd.read_csv('classification_svm_report.csv')
    source = ColumnDataSource(df_classification_report_svm)
    columns = [
    widgets.TableColumn(field=c, title=c) for c in df_classification_report_svm.columns
     ]
    data_repport_svm = DataTable(source=source, columns=columns,width=800, height=200)


    ################################################################################################

    grid = gridplot([div1, p1, p2, p3, div2, p4, div3, 
                      widgetbox(data_matrix_rfc), div6, 
                      widgetbox(data_repport_rfc), div4, 
                      p5, div5, widgetbox(data_matrix_svm), 
                      div7, widgetbox(data_repport_svm)], ncols=1)
    show(grid)

    save(grid)


