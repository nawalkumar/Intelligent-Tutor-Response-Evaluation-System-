For more details how my model is work:
check Intelligent..../pdf file...


# Intelligent Tutor Response Evaluation System

**Hybrid DistilBERT + Symbolic AI System for Automatic Tutor Feedback Classification**

Achieved **90.5% accuracy** (from 76% baseline) by combining **DistilBERT**, **SymPy symbolic math**, and domain-specific rules with **LIME/SHAP explainability**.

---

## Problem

Automatically classify tutor responses into three categories:
- **0 - Correct**
- **1 - Mistake**
- **2 - Unclear**

Dataset consists of **4 tracks** (math, tax/finance, logic, vague responses) — **1980 training** and **496 test** samples. Pure ML models fail on symbolic math and domain logic.

---

## Solution Overview

Developed an **iterative hybrid AI system** that combines:
- **Rule-based symbolic reasoning** (SymPy) for math & arithmetic validation
- **DistilBERT** (fine-tuned transformer) for semantic understanding
- **Custom domain rules** for tax, logic, and hedging detection
- **LIME & SHAP** for local & global explainability

**Key Innovation**: One unified rule engine that covers **all 4 tracks** with high precision.

---

## Key Features & Improvements

- Started with TF-IDF + Logistic Regression → **76.0% accuracy**
- Added SymPy + basic rules → **82.1% accuracy** (+6.1%)
- Integrated DistilBERT + hybrid rules → **87.3% accuracy** (+5.2%)
- Final system with improved parsing + explainability → **90.5% accuracy** (+14.5% overall)
- F1-score for minority "Unclear" class improved by **720%** (0.11 → 0.82)
- Achieved **100% accuracy** on all critical math test cases

---

## Architecture

- **Input**: Combined tutor + question + response text
- **Stage 1**: Rule Engine (SymPy math solver + tax/logic/unclear rules)
- **Stage 2**: DistilBERT (10 epochs, batch inference)
- **Stage 3**: LIME (local) + SHAP (global) explanations
- **Output**: Final label (0/1/2) + confidence + explanation

---

## Results

| Model                      | Accuracy | F1(Correct) | F1(Mistake) | F1(Unclear) |
|---------------------------|----------|-------------|-------------|-------------|
| Baseline (TF-IDF + LR)    | 76.0%    | 0.10        | 0.86        | 0.11        |
| SymPy + Rules             | 82.1%    | 0.68        | 0.89        | 0.25        |
| Hybrid + DistilBERT       | 87.3%    | 0.80        | 0.91        | 0.52        |
| **Final System**          | **90.5%**| **0.88**    | **0.95**    | **0.82**    |

---

## Tech Stack

- **Core**: Python, PyTorch, Hugging Face Transformers
- **Model**: DistilBERT (fine-tuned)
- **Symbolic AI**: SymPy
- **Explainability**: LIME, SHAP
- **Data**: Pandas, NumPy
- **NLP**: NLTK, TF-IDF (baseline)
- **Visualization**: Matplotlib, Seaborn

---

## How to Run

```bash
# Clone repo
git clone <your-repo-url>
cd tutor-evaluator

# Install dependencies
pip install -r requirements.txt

# Run inference
python inference.py --input "Solve 2y + 3 = 11" --response "y = 4"
