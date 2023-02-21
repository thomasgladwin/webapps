# webapps

Repository for Flask apps running on apps.tegladwin.com (hosted on Dreamhost). The idea was to turn some research software into more accessible web apps.

Currently:

  SemSim: Calculates the semantic similarity between target words and positive versus negative attribute word-lists.
  
  EmoCont: Calculates the emotional context of text. Output includes the primary emotions per paragraph, which can be fed into SemSim for further analysis.
  
  SemNull: Generates a null-hypothesis distribution and p-values for similarity scores, over random words with a given role in a template sentence.

To re-use - make sure to uncomment the lines for NLTK downloads when first running.

[![DOI](https://zenodo.org/badge/602762837.svg)](https://zenodo.org/badge/latestdoi/602762837)

