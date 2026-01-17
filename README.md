# Automated Reviewer Assignment for Arabic Scientific Papers

An ML system that automates reviewer assignment for scientific papers using TF-IDF vectorization and a novel Sequential Disambiguation algorithm.


## Problem

Academic journals spend 2-4 weeks manually assigning reviewers to each submitted paper. This is time-consuming, prone to bias, and doesn't scale.

## Solution

Automated system that matches papers to reviewers in seconds based on content similarity and publication history.

## How It Works

```
Paper Submission → Text Extraction (OCR) → Arabic NLP Processing → 
TF-IDF Vectorization → Sequential Disambiguation → Reviewer Recommendation
```

**Sequential Disambiguation Algorithm**:
1. Find most similar article to the query
2. Initialize candidates with that article's authors
3. Iteratively check next similar articles
4. Narrow candidates through set intersection
5. Return the most consistent expert

## Tech Stack

- **Python 3.8+** with scikit-learn, pandas, numpy
- **Arabic NLP**: Tashaphyne (stemming), arabic-stopwords
- **OCR**: PaddleOCR for text extraction
- **Data**: 1,454 Arabic articles from 5 academic platforms

## Quick Start

```python
import pickle
from scipy.sparse import load_npz

# Load model
with open('3-modeling/03-tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
    
tfidf_matrix = load_npz('3-modeling/03-tfidf_matrix_train.npz')

# Preprocess new article
from stemmer import stem_text, remove_stopwords
processed_text = remove_stopwords(stem_text(article_text))

# Get recommendation
recommended_reviewer = recommend_reviewer_logic(processed_text)
```

## Project Structure

```
├── 1-scraping/          # Web scrapers for 5 academic platforms
├── 2-preprocessing/     # OCR, text cleaning, metadata extraction
├── 3-modeling/          # Feature engineering and ML pipeline
│   ├── 00-feature-engineering.py
│   ├── 01-fitting and inference.py
│   └── stemmer.py
├── data/                # Raw and processed datasets
└── docs/                # Project report (PDF)
```


**Method Distribution**:
- 59.8% Direct Match (single author)
- 36.8% Disambiguation needed
- 3.4% Random Fallback

## Dataset

- **Sources**: AJP, ARPD, Al Manara, AJSRP, AJSP
- **Total**: 1,454 articles, 227 unique authors
- **Split**: 80% train (1,163) / 20% test (291)
- **Challenge**: 97% of metadata was missing, solved via OCR

## Future Work

- AraBERT embeddings for semantic understanding
- Workload balancing algorithm
- Multi-reviewer recommendations
- Cross-lingual support (Arabic + English)

