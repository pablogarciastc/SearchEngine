The MOOCs dataset contains the descriptions found on the webpages of around 23,000 MOOCs (Massive Open Online Courses). The files that describe the dataset are:

* moocs.dat contains the content of the webpages of all the MOOCs; each MOOC's main page occupies exactly one line in the file. 

* moocs.dat.names contains the names and the URLs of the MOOCs. The entry on line x in this file corresponds to the MOOC on line x in moocs.dat.

* moocs-queries.txt contains a set of queries that can be used to evaluate the effectiveness of a search engine.

* moocs-qrels.txt contains the relevance judgments corresponding to the queries in moocs-queries.txt. Each line in moocs-qrels.txt has the following format: (querynum documentID 1). This means that the document represented by documentID is a relevant document for the query whose number is querynum. 
