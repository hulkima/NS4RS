# NS4RS
This repository collects 237 papers related to negative sampling methods in Recommendation Systems (**RS**). Detailed information can be found in our [survey paper](https://arxiv.org/pdf/1205.2618.pdf).
We propose a overall ontology of Negative Sampling in Recommendation, which has been divided into five categories: **Static Negative Sampling**, **Dynamic Negative Sampling**, **Adversarial Negative Generation**, **Importance Re-weighting** and **Knowledge-enhanced Negative Sampling**.

- [Ontology](#Ontology)
  - [Static Negative Sampling](#static-negative-sampling)
  - [Dynamic Negative Sampling](#hard-negative-sampling)
  - [Adversarial Negative Generation](#adversarial-sampling)
  - [Importance Re-weighting](#graph-based-sampling)
  - [Knowledge-enhanced Negative Sampling](#additional-data-enhanced-sampling)

- [Scenarios](#Scenarios)
  - [Collaborative-guided Recommendation](#static-negative-sampling)
  - [Sequential Recommendation](#hard-negative-sampling)
  - [Multi-modal Recommendation](#adversarial-sampling)
  - [Multi-behavior Recommendation](#graph-based-sampling)
  - [Cross-domain Recommendation](#additional-data-enhanced-sampling)
  - [CL-enhanced Recommendation](#additional-data-enhanced-sampling)

Category
----
### Static Negative Sampling
#### Uniform Static Negative Sampling
- Atrank: An attention-based user behavior modeling framework for recommendation. `AAAI (2018)` **[[PDF]()]**
- ReCODE: Modeling Repeat Consumption with Neural ODE. `SIGIR (2024)` **[[PDF](https://dl.acm.org/doi/abs/10.1145/3626772.3657936)]**
-	BPR: Bayesian Personalized Ranking from Implicit Feedback. `UAI (2009)` **[[PDF](https://arxiv.org/pdf/1205.2618.pdf)]**
- Neural collaborative filtering. `WWW (2017) - check` **[[PDF]()]**
- Simplifying Graph-based Collaborative Filtering for Recommendation. `WSDM (2023)` **[[PDF]()]**
- GCRec: Graph-Augmented Capsule Network for Next-Item Recommendation. `TNNLS (2023)` **[[PDF]()]**
- Multi-behavior hypergraph-enhanced transformer for sequential recommendation. `KDD (2022)` **[[PDF]()]**
- Knowledge Enhanced Multi-intent Transformer Network for Recommendation. `WWW (2024) - check` **[[PDF]()]** 
- Generative-contrastive graph learning for recommendation. `SIGIR (2023)` **[[PDF]()]**
- Graph bottlenecked social recommendation. `KDD (2024)` **[[PDF]()]**
- Graph-Augmented Co-Attention Model for Socio-Sequential Recommendation. `SMC (2023) - check` **[[PDF]()]**
- Graph-Augmented Social Translation Model for Next-Item Recommendation. `TII (2023) - check` **[[PDF]()]**
- Sequential recommendation with multiple contrast signals. `TOIS (2023)` **[[PDF]()]**
- Enhanced generative recommendation via content and collaboration integration. `arxiv (2024) - check` **[[PDF]()]**
- Contrastive Cross-Domain Sequential Recommendation. `CIKM (2022)` **[[PDF]()]**
- Making Non-overlapping Matters: An Unsupervised Alignment enhanced Cross-Domain Cold-Start Recommendation. `TKDE (2024)` **[[PDF]()]**
- Align-for-Fusion: Harmonizing Triple Preferences via Dual-oriented Diffusion for Cross-domain Sequential Recommendation. `arxiv (2025) - check` **[[PDF]()]**
- GeoMF: joint geographical modeling and matrix factorization for point-of-interest recommendation. `KDD (2014)` **[[PDF]()]**
- Neural news recommendation with topic-aware news representation. `ACL (2019)` **[[PDF]()]**
- Adversarial mahalanobis distance-based attentive song recommender for automatic playlist continuation. `SIGIR (2019)` **[[PDF]()]**
- CLEAR: Contrastive Learning for API Recommendation. `ICSE (2022)` **[[PDF]()]**
- Learning tree-based deep model for recommender systems. `KDD (2018)` **[[PDF]()]**
- Effective and Efficient Training for Sequential Recommendation using Recency Sampling. `RecSys (2022)` **[[PDF]()]**


#### Predefined Static Negative Sampling
- Efficient latent link recommendation in signed networks. `KDD (2015)` **[[PDF]()]**
- Adaptive implicit friends identification over heterogeneous network for social recommendation. `CIKM (2018)` **[[PDF]()]**
- Group-based deep transfer learning with mixed gate control for cross-domain recommendation. `IJCNN (2021)` **[[PDF]()]**
- On Practical Diversified Recommendation with Controllable Category Diversity Framework. `WWW (2024)` **[[PDF]()]**
- DRN: A deep reinforcement learning framework for news recommendation. `WWW (2018)` **[[PDF]()]**
- Efficient latent link recommendation in signed networks. `KDD (2015)` **[[PDF]()]**
- Adaptive implicit friends identification over heterogeneous network for social recommendation. `KDD (2018)` **[[PDF]()]**
- Sequential recommendation with dual side neighbor-based collaborative relation modeling. `WSDM (2020)` **[[PDF]()]**
- Group-based deep transfer learning with mixed gate control for cross-domain recommendation. `IJCNN (2021)` **[[PDF]()]**
- Rella: Retrieval-enhanced large language models for lifelong sequential behavior comprehension in recommendation. `WWW (2024)` **[[PDF]()]**
- Towards open-world recommendation with knowledge augmentation from large language models. `RecSys (2024)` **[[PDF]()]**
- Uncovering ChatGPT\'s Capabilities in Recommender Systems. `arxiv (2023)` **[[PDF]()]**
- Ctrl: Connect collaborative and language model for ctr prediction. `RecSys (2023)` **[[PDF]()]**


#### Popularity-based Static Negative Sampling
- Personalized ranking for non-uniformly sampled items. `KDD Cup - check (2012)` **[[PDF]()]**
- Learning recommender systems with implicit feedback via soft target enhancement. `SIGIR (2021)` **[[PDF]()]**
- Point-of-interest recommendation: Exploiting self-attentive autoencoders with neighbor-aware influence. `CIKM (2018)` **[[PDF]()]**
- Improving pairwise learning for item recommendation from implicit feedback. `WSDM (2014)` **[[PDF]()]**
- Alleviating cold-start problems in recommendation through pseudo-labelling over knowledge graph. `WSDM (2021)` **[[PDF]()]**
- Multi-task feature learning for knowledge graph enhanced recommendation. `WWW (2019)` **[[PDF]()]**
- Learning from history and present: Next-item recommendation via discriminatively exploiting user behaviors. `KDD (2018)` **[[PDF]()]**
- Fast matrix factorization for online recommendation with implicit feedback. `SIGIR (2016)` **[[PDF]()]**
- Hyperbolic graph learning for social recommendation. `TKDE (2023)` **[[PDF]()]**
- Effective and Efficient Training for Sequential Recommendation using Recency Sampling. `RecSys (2022)` **[[PDF]()]**
- Distributed Representations of Words and Phrases and their Compositionality. `NeurIPS (2013)` **[[PDF]()]**


#### Non-sampling Static Negative Sampling
- Efficient heterogeneous collaborative filtering without negative sampling for recommendation. `AAAI (2020)` **[[PDF]()]**
- Efficient neural matrix factorization without sampling for recommendation. `TOIS (2020)` **[[PDF]()]**
- An Efficient Adaptive Transfer Neural Network for Social-Aware Recommendation. `SIGIR (2019)` **[[PDF]()]**
- Efficient non-sampling factorization machines for optimal context-aware recommendation. `WWW (2020)` **[[PDF]()]**
- Efficient non-sampling knowledge graph embedding. `WWW (2021)` **[[PDF]()]**


### Dynamic Negative Sampling
- Optimizing top-n collaborative filtering via dynamic negative item sampling. `WWW (2019)` **[[PDF]()]**
-  ` ()` **[[PDF]()]**


### Adversarial Negative Generation
- ` ()` **[[PDF]()]**

### Importance Re-weighting
- User response models to improve a reinforce recommender system. `WSDM (2021)` **[[PDF]()]**
- Transfer learning via contextual invariants for one-to-many cross-domain recommendation. `SIGIR (2020)` **[[PDF]()]**
- Influence function for unbiased recommendation. `SIGIR (2020)` **[[PDF]()]**
- BSPR: Basket-sensitive personalized ranking for product recommendation. `Information Sciences (2020)` **[[PDF]()]**
- Cross-domain Recommendation with Behavioral Importance Perception. `WWW (2023)` **[[PDF]()]**
- Learning explicit user interest boundary for recommendation. `WWW (2022)` **[[PDF]()]**
- SamWalker++: recommendation with informative sampling strategy. `TKDE (2021)` **[[PDF]()]**
- Fairly Adaptive Negative Sampling for Recommendations. `WWW (2023)` **[[PDF]()]**
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**


### Knowledge-enhanced Negative Sampling
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**

Scenarios
----
### Collaborative-guided Recommendation
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**

### Sequential Recommendation
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**

### Multi-modal Recommendation
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**

### Multi-behavior Recommendation
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**

### Cross-domain Recommendation
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**

### CL-enhanced Recommendation
- ` ()` **[[PDF]()]**
- ` ()` **[[PDF]()]**




## BibTeX
If you find this work useful for your research, please kindly cite NS4RS by:
```
@misc{NS4RS,
      title={Negative Sampling in Recommendation: A Survey and Future Directions}, 
      author={Haokai Ma and Ruobing Xie and Lei Meng and Fuli Feng and Xiaoyu Du and Xingwu Sun and Zhanhui Kang and Xiangxu Meng},
      year={2024},
      eprint={2409.07237},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2409.07237}, 
}
```
**
