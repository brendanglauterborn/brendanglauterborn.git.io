# Brendan Lauterborn

Data Scientist — Baltimore, MD

brendan.lauterborn@gmail.com | github.com/brendanglauterborn

## Education

M.S. Computer Science — Data Science, Towson University

B.S. Applied Mathematics, Texas A&M University

## Work Experience

### Data Science Intern — CPower (an NRG Company), 06/2026 – Present
- Ingested hourly utility meter data across 1,200 CAISO sites (~5M+ readings) via PySpark in Microsoft Fabric into a Delta Lake table
- Built statistical data quality scoring to detect outliers, flatlines, zeros, and nulls at scale, assigning account and monthly health grades
- Built a full-stack analytics dashboard (FastAPI + React) used by the Enrollment Settlements and Payments team
- Trained a multi-output XGBoost model on engineered time/lag features to predict kW usage and detect anomalies for 200 CAISO target accounts
- Combined ML predictions with rule-based logic to build an anomaly detection system for customer load curtailment, speeding up payout processing

### Business Intelligence Developer Intern — CPower, 06/2025 – 08/2025
- Built Python data quality scripts validating CRM data (Microsoft Dynamics 365), flagging ~8.5% of leads as low quality
- Implemented fuzzy matching to detect ~250 duplicate accounts across Dataverse and SharePoint

### Graduate Teaching Assistant — Towson University, Dept. of Computer Science, 01/2026 – 05/2026
- Assisted teaching Data Structures and Object-Oriented Programming (Java); evaluated and debugged student code

## Projects

### Brain Tumor Classification with Explainable AI

Brain tumors are typically diagnosed from MRI scans, a process that relies heavily on radiologist judgment and can benefit from automated, interpretable support tools. In this project, I built a custom convolutional neural network in PyTorch to classify tumor type from MRI images, then improved performance by applying DenseNet-121 transfer learning, raising accuracy from 79% to 88%. To make the model's decisions interpretable, I applied Grad-CAM to generate class activation heatmaps highlighting the MRI regions driving each prediction.

| Original MRI | Grad-CAM |
|:---:|:---:|
| ![Original MRI](images/xai2.1.png) | ![Grad-CAM](images/xai2.2.png) |
| ![Original MRI](images/xai3.1.png) | ![Grad-CAM](images/xai3.2.png) |
| ![Original MRI](images/xai4.1.png) | ![Grad-CAM](images/xai4.2.png) |

View code on GitHub.

### Humor Detection with Explainable AI

Humor is notoriously hard for models to detect because it depends on subtle context that simple keyword or sentiment approaches miss. In this project, I fine-tuned a RoBERTa transformer to classify text as humorous or not, improving accuracy from 56% to 98%. I then applied Captum's Integrated Gradients to attribute each prediction back to specific tokens, and built a full-stack app in FastAPI and React so users can submit their own text and see real-time predictions alongside the token-level explanations.

![Humor Detection](images/humor-detection.png)

View code on GitHub.

### CNN-Based Dog Breed Classification and Human-Dog Mapping

This project explores how well a CNN can generalize fine-grained visual classification, using dog breeds as a proxy for a task humans find intuitive but is genuinely hard at scale. I built a pipeline that detects whether an image contains a human or a dog, classifies the dog's breed across 133 categories, and maps human faces to their closest-matching breed. Replacing the baseline CNN with DenseNet-161 transfer learning took accuracy from roughly 13% to 86%.

![Dog Breed Classification](images/dog-breed.png)

View code on GitHub.

### MovieLens Recommender System

Recommender systems are core to how platforms surface relevant content, but naive similarity-based approaches often underperform once user behavior gets sparse. In this project, I built collaborative filtering and ALS matrix factorization recommenders on the MovieLens dataset to generate personalized movie suggestions. I evaluated them using ranking-focused metrics — Precision@K, Recall@K, MAP, and NDCG — and found the latent-factor models consistently outperformed baseline similarity approaches.

![MovieLens Recommender](images/movielens.png)

View code on GitHub.

### Subreddit Sentiment Classifier

Online comment quality is difficult to define, let alone classify automatically, since "quality" spans everything from controversy to genuine insight. In this project, I fine-tuned a RoBERTa transformer to classify Reddit comments into four categories — Controversial, Baseline, High-Quality, and Viral — building a full NLP pipeline from preprocessing through transformer tokenization and supervised training. The model reached an F1 score of approximately 0.79 on validation data.

![Subreddit Sentiment Classifier](images/subreddit-sentiment.png)

View code on GitHub.

### Natural Language to SQL — LLM Prompt Engineering

As LLMs get used more often to translate natural language questions into executable SQL, understanding where and why they fail becomes important for anyone relying on them in production. In this project, I used the OpenAI API to run controlled prompt-engineering experiments — zero-shot, few-shot, and chain-of-thought — measuring how each strategy affected query accuracy. The results quantified clear accuracy tradeoffs between approaches and surfaced common failure modes in LLM-generated SQL.

View code on GitHub.
