# Re-implementation of OWL2Vec* 

## Disclaimer
This repository provides a reproducible source code of OWL2Vec* for our study purpose. The original paper is accessible online here:  https://link.springer.com/article/10.1007/s10994-021-05997-6 and the original Github is https://github.com/KRR-Oxford/OWL2Vec-Star. 

## Change Log 

In order to make the original source code runnable, we have detected some small bugs and fixed them. We also commented the code for more readability for other users who would like to try learning how OWL2Vec* works. The change are as follows. 

| Date        | Changed By  | Change Location (Path & Line Number)          | Action (Add / Remove / Edit)          |
|------       |------------ |--------------------------------------         |------------------------------         |
| 2023/06/14  | Chavakan    | case_studies/helis_membership/OWL2Vec_Plus.py | added comments                        |
| 2023/06/15  | Chavakan    | owl2vec_star/owl2vec_star/lib/RDF2Vec_Embed.py| added comments                        |
|             |             |                                               |                                       |

## Issue Report

Apart from the above modification, it still remains some issues that have NOT been fixed or reported the developers as follows.  

| No. | Issue's Description                                                                           | Found By |
|-----|---------------------                                                                          |----------|
| 1   | Pre-trained gensim word2vec for using with the pre-trained option not included in the library | Chavakan |
|     |                                                                                               |          |
|     |                                                                                               |          |

## Remarks 

By viewing this repository, you can see our added comments. Note that these comments are written based on our understanding. 
