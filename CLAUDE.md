# Movie Recommendation System - Technical Documentation

## Project Identity

**Repository**: https://github.com/srjyoussef/recommendation-system.git
**Local Directory**: `D:\Recommendation System`
**Dataset Location**: Same directory as project
**Python Command**: `py` (due to PATH configuration)

---

## Executive Summary

I'm building a production-ready hybrid movie recommendation system using the MovieLens 25M dataset to demonstrate end-to-end ML engineering capabilities for mid-to-senior level data science positions ($90K-180K range). This system combines collaborative filtering and content-based approaches to create Netflix-style personalized recommendations that balance accuracy, diversity, and business value.

**Key Differentiator**: This project emphasizes rigorous temporal validation, honest performance metrics, and production-grade code quality over inflated academic benchmarks. Every design decision is documented, every metric is contextualized, and every trade-off is justified with business reasoning.

---

## Business Context & Objectives

### Problem Statement
Streaming platforms like Netflix face a critical retention challenge: users who can't find content they enjoy cancel subscriptions. My recommendation system addresses this by:
1. Personalizing content discovery to reduce time-to-watch
2. Surfacing niche content that matches user taste profiles
3. Balancing popular hits with long-tail discovery
4. Handling cold-start scenarios for new users and content

### Success Metrics
- **Primary**: Precision@10 ≥ 10% (industry-realistic target for explicit feedback)
- **Secondary**: NDCG@10, Recall@50, Coverage, Diversity metrics
- **Business**: Demonstrate production-ready ML engineering skills, not academic perfection

### Target Audience
Technical recruiters and hiring managers evaluating ML engineering competency through:
- Clean, documented, maintainable code
- Rigorous evaluation methodology with confidence intervals
- Business-aware model selection and trade-off analysis
- Production deployment considerations

---

## Technical Architecture

### Technology Stack

**Core Libraries (Polars-First Approach)**:
```python
# Data manipulation - MANDATORY: Use Polars, not pandas
import polars as pl
import numpy as np

# ML frameworks
from surprise import Dataset, Reader, SVD, NMF, KNNBasic
from surprise.model_selection import cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Evaluation
from sklearn.metrics import precision_score, recall_score, ndcg_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
```

**Development Environment**:
- Python 3.12
- Jupyter Notebooks for iterative development
- VS Code with Python Data Science extension
- Git for version control

### Data Architecture

**Dataset Files**:
1. `ml-ratings.csv`: ~25M ratings (userId, movieId, rating, timestamp)
2. `ml-movies.csv`: ~62K movies (movieId, title, genres)
3. `ml-links.csv`: IMDb/TMDb mappings (movieId, imdbId, tmdbId)

**Key Constraint**: All timestamps must be used for temporal train/validation/test splits to prevent data leakage. This is non-negotiable for honest evaluation.

---

## Analytical Framework

### Phase 1: Exploratory Data Analysis (EDA)

**Objective**: Understand data characteristics, identify quality issues, and inform modeling decisions.

**Key Analyses**:
1. **Rating Distribution Analysis**
   - Overall rating distribution (expect positive bias)
   - Temporal trends in ratings (are people rating differently over time?)
   - User rating behavior (active vs. casual raters)
   - Movie popularity distribution (power law expected)

2. **Sparsity Analysis**
   - Matrix sparsity calculation (expect >99.5% sparse)
   - User-movie interaction density
   - Cold-start problem quantification (new users/movies)

3. **Temporal Patterns**
   - Rating volume over time
   - User cohort analysis (when did users join?)
   - Movie release patterns vs. rating patterns

4. **Genre Analysis**
   - Genre distribution and co-occurrence
   - Genre popularity trends
   - User genre affinity patterns

**Deliverables**:
- Statistical summary tables with confidence intervals
- Visualization suite (distributions, time series, heatmaps)
- Data quality report (missing values, duplicates, anomalies)
- Modeling implications document

### Phase 2: Data Preprocessing

**Objective**: Transform raw data into ML-ready formats while preserving temporal integrity.

**Critical Steps**:
1. **Temporal Split Strategy** (Non-Negotiable)
   ```
   Training Set: Timestamps ≤ 80th percentile
   Validation Set: 80th < Timestamps ≤ 90th percentile  
   Test Set: Timestamps > 90th percentile
   ```
   Rationale: Simulates production scenario where we predict future ratings based on past behavior.

2. **Data Cleaning**
   - Handle missing values (document approach)
   - Remove duplicates if any
   - Validate userId/movieId consistency across files
   - Check for rating range violations (should be 0.5-5.0)

3. **Feature Engineering**
   - Parse genres into binary/multi-hot encodings
   - Extract year from movie title
   - Create user activity features (num_ratings, avg_rating, rating_variance)
   - Create movie popularity features (num_ratings, avg_rating)

4. **Format Preparation**
   - Surprise library format (user, item, rating, timestamp)
   - Polars DataFrames for content-based features
   - Sparse matrix representations where appropriate

**Deliverables**:
- Cleaned datasets saved as Polars DataFrames
- Train/val/test split documentation with statistics
- Feature engineering pipeline code
- Data validation report

### Phase 3: Baseline Models

**Objective**: Establish performance benchmarks before complex models.

**Models to Implement**:
1. **Global Mean**: Predict average rating for all user-movie pairs
2. **User Mean**: Predict user's average rating across all movies
3. **Movie Mean**: Predict movie's average rating across all users
4. **User-Movie Mean**: Combined baseline using user and movie biases

**Evaluation Framework**:
```python
def evaluate_model(predictions, actuals, k=10):
    """
    Comprehensive evaluation with confidence intervals.
    
    Returns:
        dict: Precision@k, Recall@k, NDCG@k with 95% CI
    """
    # Bootstrap confidence intervals (1000 iterations)
    # Statistical significance testing vs. baseline
    # Business interpretation of metrics
```

**Deliverables**:
- Baseline performance metrics with confidence intervals
- Error analysis (where do baselines fail?)
- Computational cost documentation
- Insights for complex model development

### Phase 4: Collaborative Filtering

**Objective**: Leverage user-item interaction patterns for personalization.

**Algorithms**:
1. **Matrix Factorization (SVD)**
   - Latent factor dimensionality: Grid search [50, 100, 150, 200]
   - Regularization: Grid search [0.02, 0.05, 0.1]
   - Learning rate tuning

2. **Non-Negative Matrix Factorization (NMF)**
   - Interpretable latent factors
   - Genre-aware factorization potential

3. **K-Nearest Neighbors (User-Based & Item-Based)**
   - Similarity metrics: Cosine, Pearson
   - Neighborhood size optimization

**Hyperparameter Optimization**:
- 5-fold cross-validation on training set
- Validation set for final model selection
- Test set held out until final evaluation
- Document computational cost vs. performance trade-offs

**Deliverables**:
- Trained models with optimized hyperparameters
- Performance comparison table
- Computational cost analysis
- Error analysis by user segment and movie type

### Phase 5: Content-Based Filtering

**Objective**: Use movie features (genres, metadata) for recommendations, especially for cold-start scenarios.

**Approach**:
1. **TF-IDF Genre Vectors**
   - Multi-hot encode genres
   - Weight by genre rarity (TF-IDF logic)
   - Compute movie-movie similarity matrix

2. **User Profile Construction**
   - Aggregate rated movie features weighted by rating
   - Create user preference vectors in genre space

3. **Recommendation Generation**
   - Find movies similar to user's historical preferences
   - Filter out already-rated movies
   - Rank by similarity score

**Cold-Start Handling**:
- New users: Use popular items in predicted preferred genres
- New movies: Recommend to users who like similar content

**Deliverables**:
- Content-based model implementation
- Cold-start performance evaluation
- Genre-based recommendation quality analysis
- Comparison with collaborative filtering on cold-start scenarios

### Phase 6: Hybrid Ensemble System

**Objective**: Combine collaborative and content-based approaches for optimal performance.

**Ensemble Strategies**:
1. **Weighted Average**
   - Learn optimal weights on validation set
   - Weights may vary by user segment (cold vs. warm)

2. **Switching Hybrid**
   - Use content-based for cold-start users/items
   - Use collaborative filtering for warm-start scenarios

3. **Feature Augmentation**
   - Use content features as additional input to collaborative models

**Final Model Selection**:
- Evaluate all approaches on test set
- Select based on: Precision@10, NDCG@10, Coverage, Diversity
- Document business trade-offs (accuracy vs. diversity vs. computational cost)

**Deliverables**:
- Final hybrid model with documented architecture
- Comprehensive evaluation report with confidence intervals
- Business recommendation for production deployment
- Future improvement roadmap

---

## Evaluation Methodology

### Metrics Framework

**Ranking Metrics** (Primary):
- **Precision@K**: What fraction of top-K recommendations are relevant?
- **Recall@K**: What fraction of relevant items appear in top-K?
- **NDCG@K**: Normalized Discounted Cumulative Gain (position-aware)

**Beyond Accuracy**:
- **Coverage**: What percentage of catalog gets recommended?
- **Diversity**: Average dissimilarity in recommendation lists
- **Novelty**: How often do we recommend non-popular items?

**Statistical Rigor**:
- Bootstrap confidence intervals (95%) for all metrics
- Statistical significance testing (paired t-test)
- Segment-based analysis (user activity levels, movie popularity tiers)

### Business-Aware Evaluation

**Contextualize All Metrics**:
- "Precision@10 of 10.2% ± 0.3% means ~1 relevant movie per top-10 list, which aligns with Netflix's estimated 2-3 relevant items per row of 20 recommendations"
- "Coverage of 15% indicates long-tail discovery, important for content ROI"

**Never Report**:
- Metrics without confidence intervals
- Improvements without statistical significance tests
- Results without business interpretation

---

## Code Quality Standards

### Mandatory Principles

**No AI Indicators**:
- Never use words like "fixed", "updated", "corrected" in code/comments/commits
- Never include TODO comments with obvious AI patterns
- Never use emojis in code or documentation
- Never include placeholder text like "YOUR_CODE_HERE"

**Professional Style**:
- First-person narrative in documentation: "I chose SVD because..."
- Present tense for methodology: "The model processes ratings by..."
- Docstrings describe purpose, not implementation details
- Comments explain *why*, not *what* (code should be self-documenting)

**Code Organization**:
```python
# GOOD: Clear business purpose
def calculate_user_diversity_preference(user_ratings, genre_matrix):
    """
    Quantify how much a user explores across genres vs. staying in comfort zone.
    
    Higher scores indicate genre-diverse viewing habits, which informs
    whether to prioritize accuracy or diversity in recommendations.
    """
    
# BAD: Obvious AI-generated comment
def calculate_diversity(data):
    # Calculate the diversity metric
    # This function computes diversity
```

### File Structure

**Required Files**:
1. `01_eda.ipynb`: Exploratory data analysis
2. `02_preprocessing.ipynb`: Data cleaning and feature engineering
3. `03_baseline_models.ipynb`: Simple benchmark models
4. `04_collaborative_filtering.ipynb`: CF algorithms
5. `05_content_based.ipynb`: Content-based approach
6. `06_hybrid_system.ipynb`: Final ensemble model
7. `07_evaluation_report.ipynb`: Comprehensive results analysis
8. `README.md`: Project overview for GitHub
9. `requirements.txt`: Dependency list

**Forbidden Files**:
- Anything with "fixed", "v2", "backup", "old", "test" in filename
- Multiple versions of same notebook
- Scratch work or debugging notebooks

### Git Workflow

**Commit Message Format**:
```
Add temporal validation framework for rating splits

Implemented 80/10/10 temporal split to prevent data leakage.
Training set uses ratings before time t, validation uses t to t+delta,
test set uses ratings after t+delta. This simulates production scenario
where we predict future user behavior.
```

**Never Commit**:
- Messages like "fix bug" or "update code"
- Messages without context or business rationale
- Multiple small commits that should be squashed

---

## Execution Protocol

### Cell-by-Cell Development with Ultrathink

**Workflow**:
1. **Read CLAUDE.md**: Understand current phase objectives
2. **Generate Cell**: Write production-ready code for single atomic task
3. **Execute Cell**: Run code and capture outputs
4. **Ultrathink Analysis**: 
   - Did the cell accomplish its objective?
   - What insights emerged from the output?
   - What validation or error handling is needed?
   - What logical next step follows from these results?
   - Are there any business implications to document?
5. **Next Cell**: Based on ultrathink, write next atomic task

**First Cell Mandate**:
```python
"""
Movie Recommendation System - Setup
"""

# Data manipulation
import polars as pl
import numpy as np

# Collaborative filtering
from surprise import Dataset, Reader, SVD, NMF, KNNBasic, KNNWithMeans
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split
from surprise import accuracy

# Content-based filtering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# Evaluation metrics
from sklearn.metrics import precision_score, recall_score, ndcg_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
from collections import defaultdict, Counter
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Dataset paths
DATA_DIR = r"D:\Recommendation System"
RATINGS_PATH = os.path.join(DATA_DIR, "ml-ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "ml-movies.csv")
LINKS_PATH = os.path.join(DATA_DIR, "ml-links.csv")

print("Environment configured successfully")
print(f"Data directory: {DATA_DIR}")
```

### Analytical Depth Requirements

**Every Analysis Must Include**:
1. **Statistical Summary**: Mean, median, std dev, percentiles
2. **Confidence Intervals**: 95% CI for key metrics using bootstrap
3. **Visualization**: Appropriate plot type with clear labels
4. **Business Interpretation**: What does this mean for recommendations?

**Example - Rating Distribution Analysis**:
```python
# Calculate statistics
rating_stats = ratings_df.select([
    pl.col("rating").mean().alias("mean"),
    pl.col("rating").median().alias("median"),
    pl.col("rating").std().alias("std_dev"),
    pl.col("rating").quantile(0.25).alias("q25"),
    pl.col("rating").quantile(0.75).alias("q75")
])

# Visualize
plt.figure(figsize=(10, 6))
ratings_df.select("rating").to_pandas().plot(kind='hist', bins=10)
plt.title("Rating Distribution - Positive Bias Evident")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Business interpretation
# NOTE: I write business interpretation as a Markdown cell immediately after the code/plot,
# using 3–6 specific bullets under a "Key findings" heading. I do not print multi-line
# narrative summaries from code cells.
```

### Documentation Standards

**Markdown Cells Between Code**:
- Explain *why* this analysis matters for the business problem
- State hypotheses before analysis
- Interpret results in recommendation system context
- Connect findings to modeling decisions

**Example**:
```markdown
## User Activity Segmentation

Understanding user engagement levels helps us tailor recommendation strategies:
- **Power users** (>100 ratings): Can leverage collaborative filtering heavily
- **Casual users** (10-100 ratings): Need hybrid approach
- **Cold-start users** (<10 ratings): Rely on content-based and popularity

This segmentation will inform our ensemble weighting strategy in Phase 6.
```

---

## Model Development Guidelines

### Hyperparameter Tuning Protocol

**Grid Search Setup**:
```python
param_grid = {
    'n_factors': [50, 100, 150, 200],
    'reg_all': [0.02, 0.05, 0.1],
    'lr_all': [0.005, 0.01],
    'n_epochs': [20, 30]
}

# 5-fold CV on training set only
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)
gs.fit(trainset)

# Best params from CV
best_params = gs.best_params['rmse']

# Train final model on full training set with best params
final_model = SVD(**best_params)
final_model.fit(trainset)

# Evaluate on validation set to check for overfitting
val_predictions = final_model.test(valset)
```

**Document Trade-offs**:
- Computational cost vs. performance gain
- Model complexity vs. interpretability
- Accuracy vs. diversity vs. novelty

### Error Analysis Framework

**Segment-Based Evaluation**:
```python
def segment_evaluation(predictions, user_metadata, movie_metadata):
    """
    Evaluate model performance across user and movie segments.
    
    Segments:
    - User activity levels (cold/casual/power)
    - Movie popularity tiers (niche/mid/blockbuster)
    - Genre categories
    
    Returns detailed performance breakdown to identify model weaknesses.
    """
    # Group predictions by segment
    # Calculate metrics per segment
    # Statistical testing for significant differences
    # Visualize segment performance gaps
```

**Failure Case Analysis**:
- What types of users does the model serve poorly?
- What types of movies are hard to recommend?
- Are there systematic biases (e.g., only recommending popular items)?

---

## Production Considerations

### Scalability Analysis

**Computational Complexity**:
- Training time vs. dataset size
- Prediction latency (single user, batch)
- Memory footprint
- Storage requirements

**Optimization Strategies**:
- Approximate nearest neighbors for item-based CF
- Matrix factorization dimensionality trade-offs
- Incremental model updates vs. full retraining

### Monitoring & Maintenance

**Metrics to Track in Production**:
- Click-through rate on recommendations
- Watch time on recommended content
- User feedback (thumbs up/down)
- Coverage and diversity trends
- Model staleness (time since last update)

**Retraining Strategy**:
- Incremental updates: Daily for fast-changing user preferences
- Full retraining: Weekly/monthly for model architecture changes
- A/B testing framework for model comparison

---

## Deliverables Checklist

### Code Artifacts
- [ ] 7 Jupyter notebooks (01-07) with complete analysis
- [ ] Clean, documented, production-ready code
- [ ] No AI indicators, first-person narrative throughout
- [ ] Reproducible results with random seeds

### Documentation
- [ ] README.md with project overview and setup instructions
- [ ] requirements.txt with exact library versions
- [ ] Inline documentation explaining business context
- [ ] Methodology decisions documented with rationale

### Analysis & Results
- [ ] Comprehensive EDA with statistical summaries
- [ ] Baseline model performance with confidence intervals
- [ ] Multiple CF algorithms compared rigorously
- [ ] Content-based approach for cold-start handling
- [ ] Final hybrid model with business-justified architecture
- [ ] Evaluation report with segment-based analysis

### Business Value
- [ ] All metrics interpreted in business context
- [ ] Model trade-offs clearly articulated
- [ ] Production deployment considerations addressed
- [ ] Future improvement roadmap

---

## Quality Assurance

Before considering any phase complete:

1. **Code Review Checklist**:
   - [ ] No hardcoded values (use variables/constants)
   - [ ] All functions have docstrings
   - [ ] No print debugging statements
   - [ ] Appropriate error handling
   - [ ] Efficient algorithms (no O(n²) where O(n log n) exists)

2. **Analysis Validation**:
   - [ ] Statistical claims include confidence intervals
   - [ ] Visualizations have clear titles, labels, legends
   - [ ] Business interpretation provided for all findings
   - [ ] Results are reproducible (random seeds set)

3. **Documentation Check**:
   - [ ] First-person narrative, professional tone
   - [ ] No AI indicators in any text
   - [ ] Methodology decisions explained with business rationale
   - [ ] Code comments explain *why*, not *what*

---

## Success Criteria

This project demonstrates mid-to-senior level data science competency when:

1. **Technical Excellence**:
   - Temporal validation prevents data leakage
   - Multiple algorithms compared with rigorous evaluation
   - Ensemble system shows thoughtful architecture
   - Code is clean, efficient, maintainable

2. **Analytical Rigor**:
   - All metrics include confidence intervals
   - Statistical significance testing performed
   - Segment-based analysis reveals model behavior
   - Error analysis informs model improvements

3. **Business Acumen**:
   - Metrics interpreted in recommendation context
   - Model selection justified by business trade-offs
   - Production considerations addressed
   - Impact quantified in business terms

4. **Professional Presentation**:
   - Documentation reads like work of experienced DS
   - No AI indicators anywhere in codebase
   - GitHub repo is portfolio-ready
   - README effectively markets the project

---

## Execution Notes

**When Running Cells**:
- Execute sequentially, never skip cells
- After each cell, use ultrathink to analyze outputs and plan next step
- If unexpected results, investigate immediately (don't move forward)
- Document all anomalies and how they were addressed

**Common Pitfalls to Avoid**:
- Using pandas instead of Polars
- Forgetting temporal split for train/val/test
- Reporting metrics without confidence intervals
- Including emojis, AI-style comments, or TODO placeholders
- Creating multiple notebook versions (v1, v2, fixed, etc.)

**Remember**: This is a portfolio project for technical recruiters. Every line of code, every visualization, every markdown cell should demonstrate professional-level data science capabilities. Quality over quantity. Business context over academic perfection. Honest metrics over inflated benchmarks.

---

**Author**: Youssef  
**Project Start**: December 2025  
**Target Completion**: Production-ready recommendation system suitable for portfolio presentation

---

## Claude Code Notebook Style Guardrails

I use this section as the operational spec for Claude Code whenever it edits any notebook in this repo (especially `01_eda.ipynb`). These rules keep the notebook professional, consistent, and free of obvious AI fingerprints.

### Non-negotiables

- I keep **Polars-first** for data manipulation. I only convert to pandas when a library requires it (typically plotting).
- I preserve **temporal integrity** (no leakage). Any split logic uses timestamps and is documented.
- I keep **all existing visuals** unless a visualization is objectively redundant *and* I replace it with a better one that serves the same purpose.
- I keep outputs **compact and intentional**. If an output is long, I reduce it to top-k views and summary tables.

### Notebook composition rules

- I use Markdown cells for section headers, hypotheses (1–3 lines), interpretation, decisions, and **Key findings**.
- I use code cells for **one atomic task** (load, validate, compute one table, generate one figure, compute one CI).
- I avoid “mega-cells” that combine multiple tasks (load + clean + analyze + plot + summarize).

### Output hygiene

- I do **not** use decorative separators (e.g., `"="*80`, `"-"*100`, ASCII banners).
- I do **not** print long narratives or “key findings” from code cells.
- I prefer `display()` for tables and show only what is needed (small `head(n)`, top-k counts, percentiles tables).
- If diagnostics are useful but noisy, I gate them behind `DEBUG = False` (default).

### “No AI indicators” enforcement

- No emojis
- No placeholder text (e.g., `YOUR_CODE_HERE`)
- No “v2 / old / backup / test / fixed / updated / corrected” wording inside notebooks, filenames, or headings

### Required micro-structure per EDA section

For each EDA subsection, I follow this structure:

1) Markdown: why this matters for recommendations + expectation (1–3 lines)  
2) Code: summary statistics table (mean/median/std/percentiles)  
3) Code: 95% CI for one key metric (bootstrap) when applicable  
4) Code: one plot with clear title/labels  
5) Markdown: **Key findings** (3–6 bullets: numbers + implication for the next modeling step)

### Mandatory style scan after edits

After edits, I scan notebook JSON to ensure banned patterns are absent:

- `print("\n" + "="*`
- `"="*` or `"-"*` used as separators
- large triple-quoted `print(f"""...""")` narrative blocks
- repeated `.head()` / `.describe()` spam
- the strings: `fixed`, `updated`, `corrected`, `v2`, `backup`, `old`, `test`

If any appear, I refactor immediately by moving narrative to Markdown, splitting cells, and reducing outputs to top-k (while keeping figures intact).

### Definition of done

A notebook is portfolio-ready when each section reads like a concise report, outputs are compact, plots are labeled, and every section ends with Key findings + modeling implications with no AI-indicator patterns.
