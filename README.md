# ActivePDB: Active Probabilistic Databases - Demo

### Osnat Drien, Matanya Freiman and Yael Amsterdamer from Bar-Ilan University

ActivePDB is a novel framework for uncertain data management. It provides an end-to-end solution to determine the correct output of the query, while asking the oracle to verify as few tuples as possible.

## Preprocessing

For the query you wish to determine its output, generate Boolean Provenance expression. Then load the datset, and the query information to the framework.

## Running The App


```bash
python main.py
```

## Viewing The App

Go to `http://127.0.0.1:5000`

## Flow:

Select from the checkbox in the page a query. The query info will appear on the square area below.

By click on the "Automatic evaluation" button, you can choose between automatic or menual evaluation.

Finally, click the "Start Evaluation" button to start the process.

![home page.](/data/images/home.PNG "This is the home page.")