# Analisi di Abbandono
Il progetto è un programma che punta a prevedere eventuale abbandono 
degli utenti ad abbonamenti e servizi di tipo telecomunicazionistico.

## Filetree
.  
├── data  
│   ├── clean  
│   │   └── telco_customer_churn_clean.csv  
│   ├── data_analysis.ipynb  
│   ├── preprocessing.ipynb    
│   └── raw    
│       └── telco_customer_churn.csv    
│    
├── doc    
│   └── project_specification.pdf    
│    
├── models    
│   └── sgdc  
│       ├── churn_feature_columns_sgdc.joblib    
│       ├── churn_label_encoder_sgdc.joblib    
│       └── churn_pipeline_sgdc.joblib    
│    
├── src    
│   └── esempio.py    
│  
├── training    
│   └── sgdc    
│       └── sgdc.py    
│  
├── pyproject.toml    
├── README.md    
└── uv.lock    

La cartella `doc` contiene le specifiche del progetto, e 
la documentazione riguardante le scelte progettuali.
La cartella `data` contiene il dataset, e le operazioni su esso effettuate
per una prima valutazione e le operazioni di preprocessing
La cartella `training` contiene delle sottocartelle, nominate per tipo di modello, 
all'interno delle quali è possibile trovare il codice relativo al training dei modelli.
La cartella `models` contiene nelle sottocartelle i modelli che potranno essere
utilizzati dal server

Il [dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/code?datasetId=13996&sortBy=voteCount) preso come riferimento

## API
Per fare una richiesta al server con i modelli disponibili, è sufficiente chiamare
```sh
curl -d "" <url_sito>/models
```
e verrà restituito un json, composto da una lista dei modelli disponibili, ognuno
identificato univocamente dal campo "nome"
