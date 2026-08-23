# Titanic Survival Prediction — Neural Network & Random Forest

This project explores passenger survival prediction on the Titanic dataset using both a **PyTorch Neural Network** and a traditional **Random Forest Classifier**. It focuses on understanding how feature engineering, network architecture, activation functions, and optimizer choice affect performance on a small tabular dataset.

---

## Table of Contents

- [Objectives](#objectives)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Neural Network](#neural-network)
  - [Baseline Architecture](#baseline-architecture)
  - [Neural Network Experiments](#neural-network-experiments)
  - [Optimizer Experiment](#optimizer-experiment)
- [Random Forest](#random-forest)
  - [Baseline](#random-forest-baseline)
  - [Tuning](#random-forest-tuning)
- [Model Comparison](#model-comparison)
- [Final Selected Model](#final-selected-model)
- [Final Test Evaluation](#final-test-evaluation)
- [Key Findings](#key-findings)

---

## Objectives

- Perform exploratory data analysis on the Titanic dataset.
- Identify relationships between passenger characteristics and survival.
- Engineer meaningful features from the raw passenger information.
- Build a baseline neural network using PyTorch.
- Systematically experiment with different features and network architectures.
- Compare the neural network with a traditional Random Forest model.
- Select the best-performing configuration using validation data.
- Evaluate the final selected model on an unseen test set.

---

## Dataset

The project uses the **Titanic dataset**, which contains passenger-level information including:

- Passenger class (`Pclass`)
- Age
- Sex
- Number of siblings/spouses aboard (`SibSp`)
- Number of parents/children aboard (`Parch`)
- Fare
- Ticket
- Cabin
- Port of embarkation (`Embarked`)

### Target Variable

`Survived`

| Value | Meaning |
|:---:|---|
| `0` | Did not survive |
| `1` | Survived |

---

## Exploratory Data Analysis

The complete exploratory data analysis is available in:

> `notebooks/01_EDA.ipynb`

The EDA includes:

- Dataset structure and descriptive statistics
- Missing-value analysis
- Target/class distribution
- Feature distributions
- Relationships between important features and survival
- Outlier analysis
- Feature engineering
- Feature-selection decisions

---

## Feature Engineering

The following engineered features were investigated:

| Feature | Description |
|---|---|
| `Title` | Extracted from passenger names |
| `FamilySize` | `SibSp + Parch + 1` |
| `TicketGroupSize` | Number of passengers associated with the same ticket |
| `CabinKnown` | Whether cabin information is available |

The raw `Name`, `Ticket`, and `Cabin` columns were not directly used in their original high-cardinality/sparse forms.

---

## Neural Network

A fully connected feed-forward neural network was implemented using **PyTorch**.

### Baseline Architecture

```text
Input → 32 → 16 → 1
```

| Setting | Value |
|---|---|
| Hidden-layer activation | ReLU |
| Output activation | Sigmoid |
| Loss function | Binary Cross-Entropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Early stopping | Yes |
| Random seed | 42 |

> The input layer size is determined by the number of preprocessed features.

### Neural Network Experiments

All experiments were performed using the same train/validation/test split and controlled conditions wherever applicable.

| Experiment | Modification | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Baseline | Original feature set | 85.07% | 80.77% | 80.77% | 80.77% |
| Exp 1 | Add `FamilySize` | 86.57% | 85.42% | 78.85% | 82.00% |
| Exp 2 | Add `CabinKnown` | **88.81%** | 84.91% | **86.54%** | **85.71%** |
| Exp 3 | One hidden layer | 86.57% | 82.69% | 82.69% | 82.69% |
| Exp 4 | Replace ReLU with Tanh | 83.58% | 82.61% | 73.08% | 77.55% |
| Exp 5 | Larger network | 87.31% | 85.71% | 80.77% | 83.17% |

**Experiment Summary**

- Adding `FamilySize` produced a modest improvement over the baseline.
- Adding `CabinKnown` produced the strongest improvement.
- Reducing the network to one hidden layer decreased performance.
- Replacing ReLU with Tanh decreased performance.
- Increasing network size did not improve performance over the selected architecture.
- The original two-hidden-layer ReLU architecture was therefore retained.

### Optimizer Experiment

An additional experiment compared Adam with SGD.

| Optimizer | Learning Rate | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Adam | 0.001 | 85.07% | 80.77% | 80.77% | 80.77% |
| SGD | 0.05 | 85.82% | 83.67% | 78.85% | 81.19% |

SGD produced a small improvement in F1-score over the Adam baseline. However, **Adam was retained** for the subsequent controlled feature and architecture experiments.

---

## Random Forest

A Random Forest Classifier was implemented as a traditional machine-learning comparison model.

### Random Forest Baseline

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 83.58% | 84.09% | 71.15% | 77.08% |

### Random Forest Tuning

`GridSearchCV` with 5-fold cross-validation was used to search for a better Random Forest configuration. F1-score was used as the primary scoring metric during tuning.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Tuned Random Forest | 86.57% | **88.64%** | 75.00% | 81.25% |
| Neural Network | **88.81%** | 84.91% | **86.54%** | **85.71%** |

---

## Model Comparison

The selected neural network achieved higher validation accuracy, recall, and F1-score than the tuned Random Forest. The Random Forest achieved higher precision, but its lower recall resulted in a lower overall F1-score.

---

## Final Selected Model

Based on validation performance, **Experiment 2 — CabinKnown** was selected as the final neural-network configuration.

### Architecture

```text
Input → 32 → 16 → Output
```

### Configuration

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Hidden activation | ReLU |
| Output activation | Sigmoid |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Early stopping | Yes |
| Random seed | 42 |

### Best Validation Performance

| Metric | Validation Result |
|---|---:|
| Accuracy | 88.81% |
| Precision | 84.91% |
| Recall | 86.54% |
| F1-score | 85.71% |

### Validation Confusion Matrix

```text
[[74,  8],
 [ 7, 45]]
```

---

## Final Test Evaluation

The test set was kept separate from the experiment-selection process and was used for the final evaluation of the selected neural-network model.

### Test Results

| Metric | Test Result |
|---|---:|
| Accuracy | 79.10% |
| Precision | 76.74% |
| Recall | 64.71% |
| F1-score | 70.21% |

### Test Confusion Matrix

```text
[[73, 10],
 [18, 33]]
```

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Did not survive | 0.80 | 0.88 | 0.84 |
| Survived | 0.77 | 0.65 | 0.70 |
| **Overall** | **0.79** | **0.79** | **0.79** |

---

## Key Findings

1. Feature engineering had a measurable effect on neural-network performance.
2. `CabinKnown` was the most effective engineered feature among the tested feature experiments.
3. The two-hidden-layer ReLU architecture performed better than the tested one-hidden-layer architecture.
4. Replacing ReLU with Tanh reduced validation performance.
5. Increasing the network size did not provide an improvement over the selected architecture.
6. The tuned Random Forest was competitive but achieved a lower validation F1-score than the neural network.
7. The selected neural network achieved a validation F1-score of 85.71%.
8. The final test F1-score was 70.21%, showing a noticeable difference between validation and unseen-test performance on this relatively small dataset.
