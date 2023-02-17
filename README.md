# NS4RS
This repository collects 200 papers related to negative sampling methods in Recommendation Systems (**RS**).

Existing negative sampling methods can be roughly divided into five categories: **Static Negative Sampling**, **Hard Negative Sampling**, **Adversarial Sampling**, **Graph-based Sampling** and **Additional data enhanced Sampling**.

- [Category](#Category)
  - [Static Negative Sampling](#static-negative-sampling)
  - [Hard Negative Sampling](#hard-negative-sampling)
  - [Adversarial Sampling](#adversarial-sampling)
  - [Graph-based Sampling](#graph-based-sampling)
  - [Additional data enhanced Sampling](#additional-data-enhanced-sampling)

- [Future Outlook](#Future-Outlook)
  - [False Negative Problem](#false-negative-problem)
  - [Curriculum Learning](#curriculum-learning)
  - [Negative Sampling Ratio](#negative-sampling-ratio)
  - [Debiased Sampling](#debiased-sampling)
  - [Non-Sampling](#non-sampling)

Category
----
### Static Negative Sampling

-	BPR: Bayesian Personalized Ranking from Implicit Feedback. `UAI(2009)` **[RS]** **[[PDF](https://arxiv.org/pdf/1205.2618.pdf)]**

-	Real-Time Top-N Recommendation in Social Streams. `RecSys(2012)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2365952.2365968)]**

-	Fast Matrix Factorization for Online Recommendation with Implicit Feedback. `SIGIR(2016)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489)]**

-	Word2vec applied to Recommendation: Hyperparameters Matter. `RecSys(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3240323.3240377)]**

-	Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph. `WSDM(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3437963.3441773)]**


### Hard Negative Sampling

-	Optimizing Top-N Collaborative Filtering via Dynamic Negative Item Sampling. `SIGIR(2013)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2484028.2484126)]**

-	Improving Pairwise Learning for Item Recommendation from Implicit Feedback. `WSDM(2014)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2556195.2556248)]**

-	Improving Latent Factor Models via Personalized Feature Projection for One Class Recommendation. `CIKM(2015)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2806416.2806511)]**

-	RankMBPR: Rank-aware Mutual Bayesian Personalized Ranking for Item Recommendation. `WAIM(2016)` **[RS]** **[[PDF](http://www.junminghuang.com/WAIM2016-yu.pdf)]**


-	WalkRanker: A Unified Pairwise Ranking Model with Multiple Relations for Item Recommendation. `AAAI(2018)` **[RS]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/11866/11725)]**



-	Simplify and Robustify Negative Sampling for Implicit Collaborative Filtering. `arXiv(2020)`  **[RS]** **[[PDF](https://arxiv.org/pdf/2009.03376)]**


-	Bundle Recommendation with Graph Convolutional Networks. `SIGIR(2020)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3397271.3401198)]**


-	Curriculum Meta-Learning for Next POI Recommendation. `KDD(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467132)]**




### Adversarial Sampling

-	Neural Memory Streaming Recommender Networks with Adversarial Training. `KDD(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3220004)]**


-	CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks. `CIKM(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3269206.3271743)]**



-	Deep Adversarial Social Recommendation. `IJCAI(2019)` **[RS]** **[[PDF](https://www.ijcai.org/proceedings/2019/0187.pdf)]**


-	Regularized Adversarial Sampling and Deep Time-aware Attention for Click-Through Rate Prediction. `CIKM(2019)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3357384.3357936)]**


-	Adversarial Binary Collaborative Filtering for Implicit Feedback. `AAAI(2019)` **[RS]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/4460/4338)]**



-	IPGAN: Generating Informative Item Pairs by Adversarial Sampling. `TNLLS(2020)`  **[RS]** **[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9240960)]**


-	PURE: Positive-Unlabeled Recommendation with Generative Adversarial Network. `KDD(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467234)]**



-	Adversarial Feature Translation for Multi-domain Recommendation. `KDD(2021)` **[RS]** **[[PDF](http://nlp.csai.tsinghua.edu.cn/~xrb/publications/KDD-2021_AFT.pdf)]**






### Graph-based Sampling

-	ACRec: a co-authorship based random walk model for academic collaboration recommendation. `WWW(2014)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2567948.2579034)]**

-	GNEG: Graph-Based Negative Sampling for word2vec. `ACL(2018)` **[NLP]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3219890)]**

-	Graph Convolutional Neural Networks for Web-Scale Recommender Systems. `KDD(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3219890)]**

-	SamWalker: Social Recommendation with Informative Sampling Strategy. `WWW(2019)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3308558.3313582)]**

-	Reinforced Negative Sampling over Knowledge Graph for Recommendation. `WWW(2020)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3366423.3380098)]**

-	MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems. `KDD(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467408)]**

-	SamWalker++: recommendation with informative sampling strategy. `TKDE(2021)` **[RS]** **[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9507306)]**

-	DSKReG: Differentiable Sampling on Knowledge Graph for Recommendation with Relational GNN. `CIKM(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3459637.3482092)]**


### Additional data enhanced Sampling

-	Leveraging Social Connections to Improve Personalized Ranking for Collaborative Filtering. `CIKM(2014)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2661829.2661998)]**

-	Social Recommendation with Strong and Weak Ties. `CIKM(2016)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2983323.2983701)]**

-	Bayesian Personalized Ranking with Multi-Channel User Feedback. `RecSys(2016)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/2959100.2959163)]**

-	Joint Geo-Spatial Preference and Pairwise Ranking for Point-of-Interest Recommendation. `ICTAI(2017)` **[RS]** **[[PDF](https://www.researchgate.net/profile/Fajie-Yuan/publication/308501951_Joint_Geo-Spatial_Preference_and_Pairwise_Ranking_for_Point-of-Interest_Recommendation/links/59bc0406aca272aff2d47bda/Joint-Geo-Spatial-Preference-and-Pairwise-Ranking-for-Point-of-Interest-Recommendation.pdf)]**

-	A Personalised Ranking Framework with Multiple Sampling Criteria for Venue Recommendation. `CIKM(2017)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3132847.3132985)]**

-	An Improved Sampling for Bayesian Personalized Ranking by Leveraging View Data. `WWW(2018)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3184558.3186905)]**

-	Reinforced Negative Sampling for Recommendation with Exposure Data. `IJCAI(2019)` **[RS]** **[[PDF](https://www.ijcai.org/Proceedings/2019/0309.pdf)]**

-	Geo-ALM: POI Recommendation by Fusing Geographical Information and Adversarial Learning Mechanism. `IJCAI(2019)` **[RS]** **[[PDF](https://www.ijcai.org/Proceedings/2019/0250.pdf)]**

-	Bayesian Deep Learning with Trust and Distrust in Recommendation Systems. `WI(2019)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3350546.3352496)]**

-	Socially-Aware Self-Supervised Tri-Training for Recommendation. `arXiv(2021)` **[RS]** **[[PDF](https://arxiv.org/pdf/2106.03569)]**

-	DGCN: Diversified Recommendation with Graph Convolutional Networks. `WWW(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3442381.3449835)]**


Future Outlook
----
### False Negative Problem



### Curriculum Learning


### Negative Sampling Ratio
-	SimpleX: A Simple and Strong Baseline for Collaborative Filtering. `CIKM(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3459637.3482297)]**


### Debiased Sampling
-	Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems. `KDD(2021)` **[RS]** **[[PDF](https://dl.acm.org/doi/pdf/10.1145/3447548.3467102)]**

### Non-Sampling
-	Efficient Heterogeneous Collaborative Filtering without Negative Sampling for Recommendation. `AAAI(2020)` **[RS]** **[[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/5329/5185)]**

