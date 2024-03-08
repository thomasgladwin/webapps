# webapps

Repository for Flask apps running online, linked here [apps.tegladwin.com](https://www.tegladwin.com/apps.php) (hosted on Heroku). The idea was to turn some research software into more accessible web apps.

Currently:

  SemSim: Calculates the semantic similarity between target words and positive versus negative attribute word-lists.
  
  SemTag: Calculates the emotional context of text. Output includes the primary emotions per paragraph, which can be fed into SemSim for further analysis.
  
  SemNull: Generates a null-hypothesis distribution and p-values for similarity scores, over random words with a given role in a template sentence.
  
  SemCluster: Finds clusters of words with simlar meanings (a kind of automated affinity mapping).
  
  textCoder: Very simple text analyzer. It tries to extract topics and attributes assigned to the topics, with counts of how many times a topic-attribute association occurred, at most once per paragraph.

To re-use - make sure to uncomment the lines for NLTK downloads when first running.

[![DOI](https://zenodo.org/badge/602762837.svg)](https://zenodo.org/badge/latestdoi/602762837)

