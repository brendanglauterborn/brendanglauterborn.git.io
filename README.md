# Brendan Lauterborn

Data Scientist — Baltimore, MD
brendan.lauterborn@gmail.com | [github.com/brendanglauterborn](https://github.com/brendanglauterborn)

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

### Graduate Teaching Assistant — Towson University, Dept. of Computer Science, 01/2026 – 05/2026
- Assisted teaching Data Structures and Object-Oriented Programming (Java); evaluated and debugged student code

### Business Intelligence Developer Intern — CPower, 06/2025 – 08/2025
- Built Python data quality scripts validating CRM data (Microsoft Dynamics 365), flagging ~8.5% of leads as low quality
- Implemented fuzzy matching to detect ~250 duplicate accounts across Dataverse and SharePoint

## Projects

### Brain Tumor Classification with Explainable AI

Brain tumors are typically diagnosed from MRI scans, a process that relies heavily on radiologist judgment and can benefit from automated, interpretable support tools. In this project, I built a custom convolutional neural network in PyTorch to classify tumor type from MRI images, then improved performance by applying DenseNet-121 transfer learning, raising accuracy from 79% to 88%. To make the model's decisions interpretable, I applied Grad-CAM to generate class activation heatmaps highlighting the MRI regions driving each prediction.

| Original MRI | Grad-CAM |
|---|---|
| <img src="images/xai2.1.png" width="220"> | <img src="images/xai2.2.png" width="220"> |
| <img src="images/xai3.1.png" width="220"> | <img src="images/xai3.2.png" width="220"> |
| <img src="images/xai4.1.png" width="220"> | <img src="images/xai4.2.png" width="220"> |


**Custom CNN**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 0.73 | 0.83 | 0.78 |
| Healthy | 0.94 | 0.85 | 0.89 |
| Meningioma | 0.64 | 0.53 | 0.58 |
| Pituitary | 0.83 | 0.92 | 0.87 |
| **Weighted Avg** | **0.80** | **0.80** | **0.79** |

**DenseNet121 Transfer Learning**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 0.88 | 0.87 | 0.88 |
| Healthy | 0.96 | 0.96 | 0.96 |
| Meningioma | 0.75 | 0.79 | 0.77 |
| Pituitary | 0.93 | 0.90 | 0.92 |
| **Weighted Avg** | **0.89** | **0.89** | **0.89** |

DenseNet121 improved meningioma classification the most, the class the custom CNN struggled with. The F1 score rose from 0.58 to 0.77 as recall improved from 0.53 to 0.79.

| Custom CNN | DenseNet121 Transfer Learning |
|---|---|
| <img src="images/scratchCM.png" width="300"> | <img src="images/transCM.png" width="300"> |

![Python](https://img.shields.io/badge/Python-eeeeee?style=flat&logo=python&logoColor=3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-eeeeee?style=flat&logo=pytorch&logoColor=EE4C2C) ![Grad-CAM](https://img.shields.io/badge/Grad--CAM-eeeeee?style=flat)

[View code on GitHub](https://github.com/brendanglauterborn/brain-tumor-mri-classification-xai).

<br>

---

<br>

### Humor Detection with Explainable AI

Humor is notoriously hard for models to detect because it depends on context, ambiguity, and wordplay that simple keyword or sentiment approaches miss. In this project, I fine-tuned a RoBERTa transformer to classify text as humorous or non-humorous, taking accuracy from a 56% baseline to 98% after fine-tuning. I then applied Captum's Integrated Gradients to attribute each prediction back to specific tokens, and built a full-stack app in FastAPI and React so users can submit their own text and see real-time predictions alongside the token-level explanations.

| Baseline RoBERTa | Fine-Tuned Model |
|---|---|
| <img src="images/cm-base.png" width="280"> | <img src="images/cm.png" width="280"> |

Integrated Gradients highlighted which tokens drove each prediction. ex: "I used to be a banker but I lost interest," words like "banker," "used," and "lost" pushed the model toward humorous, showing it leans on contextual interaction between tokens rather than any single word.

<img src="images/xai1.png" width="500">

The model also picked up on structural cues, not just semantics. ex: "The past, the present and the future all walk into a bar. It was tense," the abrupt period and the words following it carried strong positive attribution.

<img src="images/xai2.png" width="500">

![Python](https://img.shields.io/badge/Python-eeeeee?style=flat&logo=python&logoColor=3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-eeeeee?style=flat&logo=pytorch&logoColor=EE4C2C) ![Captum](https://img.shields.io/badge/Captum-eeeeee?style=flat) ![FastAPI](https://img.shields.io/badge/FastAPI-eeeeee?style=flat&logo=fastapi&logoColor=009688) ![React](https://img.shields.io/badge/React-eeeeee?style=flat&logo=react&logoColor=61DAFB)

[View code on GitHub](https://github.com/brendanglauterborn/Humor-Detection-with-XAI).

<br>

---

<br>

### CNN-Based Dog Breed Classification and Human-Dog Mapping

This project explores how well a CNN can generalize fine-grained visual classification, using dog breeds as a proxy for a task humans find intuitive but is genuinely hard at scale. I built a model that detects whether an image contains a human or a dog, classifies the dog's breed across 133 categories, and maps human faces to their closest-matching breed. Replacing the baseline CNN with DenseNet-161 transfer learning took accuracy from roughly 13% to 86%.

<img src="images/dog1.PNG" width="300">

<img src="images/dog3.PNG" width="300">

<img src="images/dog2.PNG" width="300">

![Python](https://img.shields.io/badge/Python-eeeeee?style=flat&logo=python&logoColor=3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-eeeeee?style=flat&logo=pytorch&logoColor=EE4C2C)

[View code on GitHub](https://github.com/brendanglauterborn/cnn-dog-breed-classifier).

<br>

---

<br>

### Netflix Customer Churn Prediction

*Predicting who cancels from 5,000 subscriber records.*

Netflix wants to know which subscribers are about to leave. I framed this as binary classification on the Kaggle Netflix Customer Churn dataset (5,000 customers, 13 features covering usage, plan, payment, and demographics) and compared Logistic Regression, SVM, and Random Forest on a 70/30 split. I also reviewed prior churn studies and found that both relied on self-reported surveys, which measure intent to leave and carry self-selection bias, rather than actual churn.

#### Results

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | .909 | .907 | .974 |
| SVM | .905 | .903 | .969 |
| Random Forest | .989 | .988 | .999 |

#### Key Finding

Random Forest reached 98.9% accuracy versus about 91% for the linear models, which was suspicious enough that I dug into feature importance. Inactivity signals (watch hours, avg watch time per day, last login days) dominated, along with number of profiles and payment method. Logistic regression agreed: those features were significant at p < .01. Activity was the strongest churn predictor, consistent with the business intuition that disengaged users leave.

![Random Forest feature importance](images/rfFI.png)

*Left: permutation importance (mean decrease in accuracy). Right: mean decrease in Gini impurity, how much each feature reduces class mixing across the forest's splits. Both put inactivity features and number of profiles at the top.*

![R](https://img.shields.io/badge/R-eeeeee?style=flat&logo=r&logoColor=276DC3) ![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-eeeeee?style=flat) ![SVM](https://img.shields.io/badge/SVM-eeeeee?style=flat) ![Random Forest](https://img.shields.io/badge/Random%20Forest-eeeeee?style=flat)

[View code on GitHub](https://github.com/brendanglauterborn/Big-Data-Classification).

<br>

---

<br>

### MovieLens Recommender System: User-Based CF vs. ALS Matrix Factorization

*Building two recommenders on MovieLens and testing whether latent factors beat neighbors on sparse data.*

Streaming platforms have far more content than anyone can browse, so personalization drives engagement. I built two recommenders on the MovieLens latest-small dataset (about 100,000 ratings, 600 users, 9,000 movies) and compared them with ranking metrics on an 80/20 train/test split.

- **User-based Collaborative Filtering (baseline):** cosine similarity between mean-centered user ratings, k = 50 neighbors, ratings predicted as the user's mean plus a similarity-weighted average of neighbors' deviations from their own means.
- **ALS Matrix Factorization:** the user-item matrix factorized into user and item latent factors, using the `implicit` library (20 factors, 20 iterations, regularization 0.1, confidence weight α = 15).

#### Results

| Metric | User-based CF (K=5 → 20) | ALS (K=5 → 20) |
|---|---|---|
| Precision@K | .0066 → .0043 | .142 → .133 |
| Recall@K | .0033 → .0063 | .077 → .239 |
| MAP | .0176 → .0207 | .2328 → .2331 |
| NDCG | .0078 → .0074 | .145 → .203 |

User-based CF also had an RMSE of 0.876 over 19,347 test ratings.

<table>
  <tr>
    <td align="center"><b>User-based CF (relevant = rating ≥ 3.0)</b></td>
    <td align="center"><b>ALS Matrix Factorization</b></td>
  </tr>
  <tr>
    <td><img src="images/cf_perf1.png" width="400"></td>
    <td><img src="images/ALS_perf1.png" width="400"></td>
  </tr>
</table>

#### Key Finding

ALS beat user-based CF on every ranking metric. At K=5, ALS precision was 14.2% versus 0.66% for the baseline. MovieLens is sparse, so most user pairs share very few rated movies, which weakens neighbor-based similarity. Latent factors can pick up structure that neighbors miss. Precision fell slightly as K grew for both models, while recall rose, which is the expected tradeoff. ALS recall reached 23.9% by K=20.

#### Limitations and Next Steps

- Random 80/20 split of ratings rather than a per-user or time-based split.
- Cold start (new users with no history) is a known weakness of both approaches and was not evaluated.
- Next: content-based filtering and deep learning approaches.

![Python](https://img.shields.io/badge/Python-eeeeee?style=flat&logo=python&logoColor=3776AB) ![implicit](https://img.shields.io/badge/implicit-eeeeee?style=flat) ![pandas](https://img.shields.io/badge/pandas-eeeeee?style=flat&logo=pandas&logoColor=150458) ![NumPy](https://img.shields.io/badge/NumPy-eeeeee?style=flat&logo=numpy&logoColor=013243)

[View code on GitHub](https://github.com/brendanglauterborn/Big-Data-Recommender-Sys).

<br>

---

<br>

### Natural Language to SQL — LLM Prompt Engineering

*Testing how prompting strategy changes the SQL an LLM writes, and where it breaks.*

Non-experts increasingly rely on LLMs to query databases, so it matters where the generated SQL fails. I wrote 15 questions of varying difficulty (5 easy, 5 medium, 5 hard) over the Chinook SQLite database and tested three prompting strategies with gpt-4.1-mini through the OpenAI API. That gave 45 queries in total. I executed each one and graded it correct, partially correct, or incorrect against a gold query.

#### Prompting Strategies

Every prompt included the full Chinook schema and the question, and told the model to return only a single SQL query. Only the extra guidance changed:

- **Zero-shot:** the schema and question only.
- **Few-shot:** the same, plus three worked examples: a single-table projection, a join with aggregation and GROUP BY, and a multi-table join with ORDER BY and LIMIT.
- **Chain-of-thought-style:** the model is told to reason step by step about tables, joins, filters, and aggregations before returning only the final query.

The schema is embedded directly in each prompt rather than retrieved with RAG, which is reasonable at this scale.

<details>
<summary> Schema used in the experiments (subset)</summary>

| Table | Key attributes |
|---|---|
| Album | AlbumId, Title, ArtistId |
| Artist | ArtistId, Name |
| Customer | CustomerId, FirstName, LastName, Country, Email |
| Employee | EmployeeId, FirstName, LastName, Title, ReportsTo |
| Genre | GenreId, Name |
| Invoice | InvoiceId, CustomerId, InvoiceDate, BillingCountry, Total |
| InvoiceLine | InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity |
| MediaType | MediaTypeId, Name |
| Playlist | PlaylistId, Name |
| PlaylistTrack | PlaylistId, TrackId |
| Track | TrackId, Name, AlbumId, MediaTypeId, GenreId, UnitPrice |

</details>

#### Results

| Strategy | Easy | Medium | Hard | Total correct |
|---|---|---|---|---|
| Zero-shot | 5/5 | 5/5 | 1/5 | 11/15 |
| Few-shot | 5/5 | 4/5 | 1/5 | 10/15 |
| Chain-of-thought | 5/5 | 5/5 | 2/5 | 12/15 |

#### Key Finding

Difficulty mattered more than prompting strategy. All three strategies handled easy and medium questions almost perfectly, then dropped sharply on hard questions that combined multiple joins, aggregation, and ordering. Chain-of-thought did slightly better overall, but the gap is only one or two queries out of 15. Failures were mostly missing GROUP BY, ORDER BY, or LIMIT clauses, or queries that stopped partway through a join.

#### Limitations and Next Steps

- One model and one schema. Next steps are more LLMs, a larger question set, and more varied schemas.
- The chain-of-thought prompt asked the model not to output its reasoning, so I never inspected whether it actually reasoned step by step.

![Python](https://img.shields.io/badge/Python-eeeeee?style=flat&logo=python&logoColor=3776AB) ![OpenAI API](https://img.shields.io/badge/OpenAI%20API-eeeeee?style=flat&logo=openai&logoColor=412991) ![SQLite](https://img.shields.io/badge/SQLite-eeeeee?style=flat&logo=sqlite&logoColor=003B57) ![pandas](https://img.shields.io/badge/pandas-eeeeee?style=flat&logo=pandas&logoColor=150458)

[View code on GitHub](#)
