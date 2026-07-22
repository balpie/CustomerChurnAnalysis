# Analisi di Abbandono
Il progetto è un programma che punta a prevedere eventuale abbandono 
degli utenti ad abbonamenti e servizi di tipo telecomunicazionistico.
Per informazioni dettagliate riguardanti i modelli utilizzati, il relativo 
training, e la struttura dell'applicazione web fare riferimento a `doc/project_report.pdf`.

## Presentazione
La presentazione del progetto è stata creata con l'auito di canva, ed è possibile 
trovarla al seguente link: [presentazione](https://canva.link/8isla41msnknw1p)

## Docker 
Per realizzare l'immagine docker, utilizzare il comando:
```sh  
docker build -t app_flask .
```
per creare e avviare un nuovo container a partire dall'immagine docker appena creata, utilizzare il comando:
```sh  
docker run -p 5000:5000 app_flask
```
In seguito l'applicazione sarà disponible al link: http://127.0.0.1:5000/
