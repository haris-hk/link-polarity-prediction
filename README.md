# Predicting Link Polarity and Structural Balance in Pakistani Political Networks using Graph Neural Networks

## Project Overview

This project investigates **Structural Balance Theory** in social networks by predicting link polarity (positive/negative ties) using Graph Neural Networks (GNNs). We train a foundational GNN model on large-scale signed datasets (Slashdot) and then fine-tune it on a localized dataset of Pakistani political figures to analyze whether regional political rivalries settle into mathematically balanced network states.

**Key Research Question:** Does Pakistan's political network exhibit structural balance properties predicted by balance theory?

---

## Architecture & Methodology

### Approach
1. **Transfer Learning Pipeline:**
   - Pretrain on large external signed benchmark (Slashdot: 82,140 nodes)
   - Fine-tune on localized Pakistan political network (43,085 nodes)
   - Shared feature space across datasets

2. **Model Architecture: SignedLinkPredictor**
   - **Two-stream GCN design** (separate positive/negative message passing)
   - **Hidden dimension:** 128
   - **Dropout:** 0.35
   - **Optimizer:** Adam
   - **Loss:** Binary cross-entropy
   - **Early stopping:** Validation AUC-based (patience=20)

3. **Baseline Comparison:**
   - Logistic Regression on node degree & community features
   - Provides performance reference for GNN improvement

---

## Dataset Summaries

### 🌐 External Benchmark: Slashdot Dataset

| Metric | Value |
|--------|-------|
| **Nodes** | 82,140 |
| **Total Edges** | 500,481 |
| **Positive Edges (allies)** | 382,167 (76.4%) |
| **Negative Edges (rivals)** | 118,314 (23.6%) |
| **Triangles Sampled** | 9,896 |
| **Balanced Triangles** | 8,615 |
| **Balance Ratio** | **0.8706** (87.06% balanced) |

**Interpretation:** Slashdot exhibits strong structural balance with 87% of triangles obeying balance theory (no three-way conflicts).

---

### 🇵🇰 Pakistan Political Network

| Metric | Value |
|--------|-------|
| **Nodes** | 43,085 |
| **Total Edges** | 68,644 |
| **Positive Edges (allies)** | 48,769 (71.0%) |
| **Negative Edges (rivals)** | 19,875 (29.0%) |
| **Triangles Sampled** | 9,900 |
| **Balanced Triangles** | 3,991 |
| **Balance Ratio** | **0.4031** (40.31% balanced) |

**Interpretation:** Pakistan's network is **significantly less balanced** than Slashdot (40% vs 87%), suggesting greater structural complexity, multi-sided conflicts, or coalition instability in political dynamics.

---

## Results & Metrics

### 📊 Benchmark (Slashdot) Pretraining Results

#### Baseline: Logistic Regression
| Metric | Value |
|--------|-------|
| Accuracy | 0.7627 |
| F1-Score | 0.8653 |
| AUC-ROC | 0.5618 |

#### GNN (PyG GCN) - Pretrained Model
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.8816** ✅ (+15.5% vs baseline) |
| **F1-Score** | **0.9225** ✅ (+6.6% vs baseline) |
| **AUC-ROC** | **0.9426** ✅ (+67.7% vs baseline) |
| **Best Epoch** | 120 |
| **Validation AUC (Best)** | 0.9400 |
| **Final Loss** | 0.2375 |

**Key Finding:** GNN dramatically outperforms logistic regression on Slashdot, achieving ~94% AUC.

---

### 🇵🇰 Pakistan Fine-tuning Results (Transfer Learning)

#### Baseline: Logistic Regression
| Metric | Value |
|--------|-------|
| Accuracy | 0.6714 |
| F1-Score | 0.8030 |
| AUC-ROC | 0.6057 |

#### GNN (Fine-tuned from Benchmark)
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.6370** ⚠️ (-5.1% vs baseline) |
| **F1-Score** | **0.6315** ⚠️ (-21.4% vs baseline) |
| **AUC-ROC** | **0.9312** ✅ (+53.7% vs baseline) |
| **Early Stopping Epoch** | 48 |
| **Validation AUC (Best)** | 0.9300 |
| **Training Epochs** | 120 (pretrain) + 200 (finetune) |

**Key Finding:** GNN achieves strong AUC discrimination (93%) but lower accuracy/F1, suggesting:
- Model learns high-confidence predictions but calibration differs from baseline
- Transfer learning successfully adapts benchmark patterns to Pakistan domain
- Possible label imbalance or domain distribution shift

---

## Comparative Analysis

### Model Performance Comparison

```
                 Slashdot (Benchmark)    Pakistan (Localized)
                 Baseline    GNN          Baseline    GNN
Accuracy:        76.27%      88.16%       67.14%      63.70%
F1-Score:        86.53%      92.25%       80.30%      63.15%
AUC-ROC:         56.18%      94.26%       60.57%      93.12%
```

### Key Observations

1. **Transfer Learning Success:** 
   - GNN trained on Slashdot (87% balanced) successfully transfers to Pakistan (40% balanced)
   - AUC remains ~93% on Pakistan despite domain difference, showing robust feature learning

2. **Structural Balance Mismatch:**
   - Slashdot: 87% balanced → Healthy consensus-building network
   - Pakistan: 40% balanced → Fragmented, multi-sided conflicts
   - **Political Interpretation:** Pakistani political network exhibits greater polarization and unstable coalitions

3. **Baseline vs GNN Trade-off:**
   - Baseline: High accuracy (67%) but poor ranking (AUC 61%)
   - GNN: Strong ranking (AUC 93%) but lower accuracy suggests miscalibrated confidence
   - **Implication:** GNN learns polarity patterns well but may overfit to noisy labels

---

## Structural Balance Theory Validation

### What is Structural Balance?

Balance theory states that triads (3-node triangles) prefer one of two configurations:
- **Balanced:** All positive edges, or 1 positive + 2 negative (transitivity holds)
- **Unbalanced:** 2 positive + 1 negative, or all negative (tension/instability)

### Empirical Findings

| Network | Balance Ratio | Interpretation |
|---------|---------------|-----------------|
| **Slashdot** | 87.06% | Highly stable; users form clear coalitions with minimal conflict |
| **Pakistan** | 40.31% | Unstable; lacks dominant coalitions; complex multi-party dynamics |

**Conclusion:** Pakistan's political network **violates structural balance theory**, suggesting:
- Multi-party politics with shifting alliances
- No single dominant coalition
- Temporal dynamics (elections, scandals) create transient imbalance
- Mathematical model may be too simplistic for real political systems

---

## Visualizations Generated

All plots automatically saved to `plots/` folder:

1. **benchmark_slashdot_graph_stats.png**
   - Network density, degree distribution, community structure
   - Confirms 87% balance on Slashdot

2. **pakistan_graph_stats.png**
   - Network density, degree distribution, community structure
   - Shows 40% balance on Pakistan

3. **pakistan_metric_comparison.png**
   - AUC/F1/Accuracy comparison: Baseline vs GNN
   - Visualizes trade-off between ranking and calibration

---

## Usage & Reproducibility

### Environment Setup
```powershell
python -m venv .venv
& ".venv\Scripts\Activate.ps1"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric pandas scikit-learn networkx matplotlib
```

### Run Full Pipeline (Transfer Learning)
```powershell
& ".venv\Scripts\python.exe" main.py `
  --use-external-benchmark `
  --benchmark-dataset slashdot `
  --use-gnn `
  --hidden-dim 128 `
  --lr 0.005 `
  --dropout 0.35 `
  --pretrain-epochs 120 `
  --finetune-epochs 200
```

### Run Hyperparameter Sweep
```powershell
& ".venv\Scripts\python.exe" main.py `
  --use-external-benchmark `
  --benchmark-dataset slashdot `
  --use-gnn `
  --sweep `
  --pretrain-epochs 80 `
  --finetune-epochs 120
```

### Run Baseline Only (Faster)
```powershell
& ".venv\Scripts\python.exe" main.py
```

---

## Technical Details

### Party Rivalry Assumptions (party_rivalries.json)
```json
{
  "PTI": ["PML(N)", "PPP"],
  "PML(N)": ["PTI", "PPP"],
  "PPP": ["PTI", "MQM-P"],
  "JUI-F": ["MQM-P"],
  "MQM-P": ["PPP", "JUI-F"]
}
```

**Note:** Rivalries based on historical coalitions and electoral competition. Refined from 2024 election dynamics.

### Edge Construction Rules

**Positive Edges (Allies):**
- Same party membership
- Co-mentioned positively in tweets (keywords: "support", "alliance")

**Negative Edges (Rivals):**
- Party rivalry from config
- Co-mentioned negatively in tweets (keywords: "against", "oppose")

---

## Limitations & Future Work

### Current Limitations
1. **Static snapshot:** Network represents 2024 election period; lacks temporal evolution
2. **Rule-based labels:** Party rivalries manually curated; tweet sentiment from keyword matching, not NLP
3. **Domain shift:** Benchmark (tech community) differs from politics (institutional constraints)
4. **Label noise:** ~10% estimated labeling error from heuristics

### Future Extensions
- **Temporal dynamics:** Model time-evolving alliances (elections, corruption scandals)
- **NLP-based sentiment:** Train BiLSTM or BERT on political discourse for sign inference
- **Cross-domain transfer:** Test on other political networks (India, Turkey) to validate generalization
- **Ablation studies:** Measure impact of each edge source (party membership vs. tweets)
- **Robustness testing:** Label noise injection, edge sparsity sensitivity

---

## Project Team

- **Moiz Zulfiqar** (Data & Preprocessing)
- **Hamza Ansari** (Modeling & Implementation)
- **Haris Hussain Khan** (Evaluation & Analysis)

---

## References

- Heider, F. (1946). "Attitudes and Cognitive Organization." *Journal of Psychology*
- Leskovec, L., Huttenlocher, D., & Kleinberg, J. (2010). "Signed networks in social media." *CHI 2010*
- Kipf, T. & Welling, M. (2016). "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR*
- PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/

---

## License

See LICENSE file for details.
