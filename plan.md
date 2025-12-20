# Interpretable Deep Learning for Galaxy Morphology Classification

## 1. Background & Motivation

Galaxy morphology is a fundamental observable in astrophysics, closely related to galaxy formation and evolution. With the growing volume of astronomical imaging data, manual classification becomes inefficient and subjective. This project explores the use of convolutional neural networks (CNNs) for galaxy morphology classification, with interpretability as a core objective to align machine learning predictions with astrophysical understanding. Understanding *why* a model makes specific predictions is crucial for scientific trust and for discovering new astrophysical insights.

## 2. Research Objectives

- Build a CNN-based classifier to distinguish galaxy morphologies (e.g., spiral vs elliptical)
- Establish baseline comparison with traditional morphological methods (CAS parameters, Gini-M20)
- Evaluate model performance beyond accuracy using class-wise metrics and calibration
- Apply interpretability techniques (Grad-CAM, Integrated Gradients) to analyze which visual features influence predictions
- Quantitatively assess whether CNN attention aligns with astrophysical expectations
- Establish baseline for future multi-survey transfer learning studies

## 2.1 Domain-Specific Challenges

- **Projection effects**: Edge-on spirals can resemble ellipticals
- **Redshift evolution**: Morphology appearance changes with lookback time
- **Surface brightness dimming**: Affects feature visibility at higher redshifts
- **Label noise**: Citizen science disagreement in classifications
- **Class imbalance**: Fewer mergers/irregulars than spirals and ellipticals

## 3. Dataset

### 3.1 Source Data
- Galaxy Zoo 2 (SDSS images)
- Expected dataset size: ~240,000 galaxies
- Image specifications: SDSS g-r-i composite RGB cutouts

### 3.2 Label Processing
- Labels derived from aggregated citizen science annotations
- Strategy: Use vote fractions as soft labels or apply threshold (e.g., >80% agreement) for hard labels
- Address label uncertainty through:
  - Probabilistic label retention (vote fractions)
  - Uncertainty-aware loss functions (optional: evidential deep learning)
  - Document consensus metrics (entropy, agreement scores)

### 3.3 Preprocessing Pipeline
- Resizing to consistent dimensions (e.g., 224×224 for ResNet compatibility)
- Normalization (zero mean, unit variance per channel)
- Optional augmentation: rotation (physically motivated), horizontal/vertical flips
- Brightness/contrast normalization to handle surface brightness variations

### 3.4 Data Splits
- Train/validation/test split: 70/15/15 or 60/20/20
- Stratify by:
  - Morphology class
  - Redshift bins (if available)
  - Brightness distribution
- Ensure no data leakage

### 3.5 Data Quality & Preprocessing
- Remove artifacts (satellite trails, diffraction spikes, edge effects)
- Handle redshift-dependent resolution degradation
- Document selection criteria:
  - Magnitude limits
  - Redshift range (if applicable)
  - Signal-to-noise ratio thresholds
  - Minimum angular size requirements

## 4. Methodology

### 4.1 Exploratory Data Analysis
- Visual inspection of galaxy samples across morphology classes
- Class distribution analysis and imbalance quantification
- Resolution and noise characteristics per class
- Correlation analysis between labels and known parameters (if available)
- Statistical analysis of vote distributions and label confidence

### 4.2 Baseline Models

#### Traditional Method Baseline
- CAS parameters (Concentration, Asymmetry, Smoothness)
- Gini-M20 coefficients
- Simple decision tree or logistic regression on morphological parameters
- Purpose: Compare deep learning against established astrophysical methods

#### Simple CNN Baseline
- 3-4 convolutional layers with max pooling
- 1-2 fully connected layers
- Purpose: Establish performance reference for shallow architecture

### 4.3 Main Model Architecture

#### Primary Architecture
- **ResNet18** pretrained on ImageNet
- Alternative considerations:
  - EfficientNet-B0 (parameter efficiency)
  - Vision Transformer (ViT-Small) for native attention mechanisms
  - AstroCLIP or astronomy-pretrained models (if available)

#### Architecture Modifications
- Replace final classification layer for galaxy morphology classes
- Optional: Add auxiliary outputs for morphological parameters
- Optional: Multi-scale feature pyramid for capturing structures at different scales

### 4.3.1 Architecture Ablations
- Compare ImageNet pretraining vs. random initialization vs. astronomical pretraining
- Evaluate depth vs. width tradeoffs
- Test multi-scale feature fusion (FPN-style architecture)
- Document impact of each architectural choice

### 4.4 Training Strategy

#### Loss Functions
- Primary: Cross-entropy loss (or weighted cross-entropy for class imbalance)
- Alternative: Focal loss for hard example mining
- Optional: Soft label loss if using vote fractions

#### Optimization
- Optimizer: Adam or AdamW
- Learning rate: 1e-4 with cosine annealing or reduce-on-plateau
- Batch size: 32-64 (adjust based on GPU memory)
- Epochs: 50-100 with early stopping (patience: 10-15 epochs)
- Regularization: Dropout (0.3-0.5), weight decay (1e-4)

#### Data Augmentation
- Rotation: 0-360° (galaxies are rotationally invariant)
- Horizontal/vertical flips
- Slight zoom/scale variations
- Avoid: heavy color jittering (preserves astrophysical color information)

### 4.5 Interpretability Framework

#### Gradient-Based Methods
- **Grad-CAM**: Applied to final convolutional layers
- **Integrated Gradients**: For pixel-level attribution
- **Guided Grad-CAM**: Combining Grad-CAM with guided backpropagation

#### Alternative Methods
- **SHAP (SHapley Additive exPlanations)**: Model-agnostic explanations
- **Attention rollout**: If using Vision Transformers
- **Layer-wise relevance propagation (LRP)**: For deep attribution

#### Visualization Strategy
- Generate saliency maps for:
  - Correctly classified samples (typical cases)
  - High-confidence predictions
  - Misclassified samples (failure modes)
  - Ambiguous cases (low-confidence predictions)
- Overlay saliency maps on original images
- Compare attention across morphology classes

### 4.6 Feature Attribution Validation

#### Quantitative Interpretability Metrics
- Extract morphological parameters from images:
  - Sérsic profile fitting
  - Concentration index
  - Asymmetry measures
  - Spiral arm tracings (if applicable)

#### Alignment Analysis
- Correlate saliency map peak locations with known structures:
  - Bulge regions for ellipticals
  - Spiral arms for spirals
  - Disk vs. bulge for classification boundaries
- Quantify overlap using:
  - Intersection over Union (IoU) with morphological masks
  - Pixel-wise correlation coefficients
  - Statistical tests for spatial alignment

#### Human-in-the-Loop Validation
- Have domain experts (or self-assess if experienced) rate whether highlighted regions make astrophysical sense
- Scale: 1-5 rating for physical interpretability
- Document agreement metrics between CNN attention and expert expectations

## 5. Evaluation

### 5.1 Classification Metrics

#### Standard Metrics
- Overall accuracy
- Precision, recall, F1-score per class
- Confusion matrix with class-wise error analysis
- Cohen's kappa (accounts for chance agreement)

#### Advanced Metrics
- Class-weighted accuracy (handles imbalance)
- ROC-AUC per class (one-vs-rest)
- Precision-Recall curves for each class
- Top-2 and Top-3 accuracy (for ambiguous cases)

#### Calibration Assessment
- Reliability diagrams (calibration plots)
- Expected Calibration Error (ECE)
- Brier score
- Confidence histogram analysis

### 5.2 Astrophysical Validation

#### Performance Stratification
Analyze model performance across:
- **Redshift bins**: Does performance degrade with distance?
- **Apparent magnitude**: Effect of brightness on classification
- **Signal-to-noise ratio**: Noise sensitivity analysis
- **Galaxy size**: Angular diameter impact on accuracy
- **Surface brightness**: Dimming effects on features

#### Morphological Consistency
- Correlation with morphological parameters (CAS, Gini-M20)
- Comparison with Hubble classification scheme
- Cross-validation with other survey classifications (if available)
- Physical plausibility checks (e.g., spiral arms in "spiral" predictions)

#### Uncertainty Analysis
- Prediction confidence vs. Galaxy Zoo 2 vote entropy
- Calibration of uncertainty estimates
- Out-of-distribution detection:
  - Test on galaxies from other surveys (e.g., HST, DECaLS samples)
  - Anomaly detection for unusual morphologies

### 5.3 Interpretability Assessment

#### Qualitative Analysis
- Visual inspection of saliency maps across classes
- Case studies of representative examples
- Analysis of failure modes through interpretability lens

#### Quantitative Analysis
- IoU scores between saliency maps and morphological masks
- Statistical significance of feature alignment
- Consistency of attention patterns within classes
- Diversity of attention patterns across classes

#### Scientific Insight Extraction
- Do models learn astrophysically meaningful features?
- Are there unexpected patterns that suggest new morphological indicators?
- How do misclassifications correlate with ambiguous astrophysical features?

### 5.4 Failure Analysis

#### Systematic Error Identification
- Which morphology transitions are most confused?
- Are edge-on galaxies systematically misclassified?
- Do low signal-to-noise images show specific failure patterns?
- Are there biases related to galaxy size, brightness, or color?

#### Comparison with Human Performance
- Where do humans and models disagree?
- Are model failures aligned with low human consensus?
- Do models make "non-human" errors?

#### Root Cause Analysis
- Label noise vs. model limitations vs. data quality
- Insufficient training examples for rare classes
- Limitations of 2D projection (missing 3D information)

## 6. Results

### 6.1 Model Performance
- Report all metrics from Section 5.1
- Compare baseline vs. main model
- Compare traditional methods vs. deep learning
- Include tables and plots (confusion matrix, ROC curves, calibration plots)

### 6.2 Interpretability Findings
- Present representative saliency maps
- Quantitative alignment metrics with morphological features
- Class-specific attention pattern analysis
- Highlight astrophysically meaningful discoveries

### 6.3 Astrophysical Validation
- Performance across physical parameter ranges
- Comparison with known morphological indicators
- Out-of-distribution performance

## 7. Discussion

### 7.1 Model Strengths and Limitations
- Where does the model excel?
- What are the failure modes?
- How does interpretability enhance understanding?

### 7.2 Sources of Misclassification
- Noise and low signal-to-noise ratios
- Edge-on vs. face-on projection effects
- Ambiguous morphologies (transition types)
- Label noise and citizen science disagreement
- Resolution limitations

### 7.3 Interpretability Insights
- Do learned features align with astrophysical expectations?
- What features drive classification decisions?
- Are there unexpected attention patterns?
- How do attention patterns differ between correct and incorrect predictions?

### 7.4 Astrophysical Relevance
- Can interpretability guide new morphological indicators?
- How does this approach compare to traditional methods?
- What are implications for large-scale surveys?

### 7.5 Computational Efficiency
- Training time and resource requirements
- Inference speed for large datasets
- Tradeoffs between accuracy and efficiency

### 7.6 Comparison with Human Classifiers
- Agreement with Galaxy Zoo 2 consensus
- Model confidence vs. human uncertainty
- Complementary strengths and weaknesses

## 8. Limitations

### 8.1 Data Limitations
- Single survey (SDSS) - generalization concerns
- 2D imaging only (no 3D kinematics)
- Limited wavelength coverage
- Label noise from citizen science
- Class imbalance issues

### 8.2 Model Limitations
- Interpretability methods show correlation, not causation
- Saliency maps can be noisy and unstable
- Pretrained models may have ImageNet biases
- Limited to classification (no generative understanding)

### 8.3 Scope Limitations
- Binary or few-class classification (not full Hubble sequence)
- No temporal evolution modeling
- No multi-band simultaneous analysis
- No integration with spectroscopic data

## 9. Future Work

### Priority 1: Enhanced Interpretability
- Physics-informed neural networks (PINNs) with morphological priors
- Concept bottleneck models (map to explicit astrophysical concepts)
- Causal interpretability methods
- Mechanistic interpretability for understanding learned representations

### Priority 2: Scientific Extensions
- **Multi-label classification**: Use probabilistic GZ2 labels for nuanced morphology
- **Redshift-aware models**: Include photometric/spectroscopic redshift as input feature
- **Multi-band modeling**: Simultaneous u, g, r, i, z analysis
- **Time-domain analysis**: Morphological change detection in repeat imaging
- **3D information integration**: Combine imaging with kinematic data

### Priority 3: Scaling & Generalization
- **Self-supervised pretraining** on unlabeled galaxy images
- **Transfer learning** across surveys:
  - Test on DESI, Euclid, LSST data
  - Test on HST high-resolution images
- **Active learning** for efficient labeling of new data
- **Federated learning** across survey datasets
- **Domain adaptation** techniques for cross-survey robustness

### Priority 4: Advanced Architectures
- Vision Transformers with explicit attention mechanisms
- Equivariant neural networks (rotation/scale invariance)
- Graph neural networks for structural analysis
- Diffusion models for morphology generation and understanding

## 10. Implementation Details

### 10.1 Tools & Frameworks

#### Data Handling
- **Astropy**: FITS file handling, coordinate systems
- **AstroML**: Statistical tools for astronomy
- **astroquery**: Querying astronomical databases
- **Pandas/NumPy**: Data manipulation

#### Machine Learning
- **PyTorch** or **TensorFlow/Keras**: Deep learning framework
- **timm** (PyTorch Image Models): Pretrained model zoo
- **scikit-learn**: Baseline models and metrics
- **imbalanced-learn**: Handling class imbalance

#### Interpretability
- **Captum** (PyTorch): Integrated Gradients, SHAP, attribution methods
- **SHAP**: Model-agnostic explanations
- **pytorch-grad-cam**: Grad-CAM implementations
- **tf-explain** (TensorFlow): TensorFlow interpretability tools

#### Visualization
- **matplotlib**: Static plots
- **seaborn**: Statistical visualizations
- **astropy.visualization**: Astronomy-specific plotting
- **Plotly**: Interactive visualizations
- **WandB** or **TensorBoard**: Training monitoring

#### Morphological Analysis
- **Photutils**: Source detection and photometry
- **Statmorph**: Morphological parameter calculation
- **GalSim**: Galaxy image simulation (for testing)

### 10.2 Development Environment

#### Agentic IDE Setup
- **Primary IDE**: Cursor (or similar agentic IDE like Windsurf)
- **AI pair programming**: Leverage for:
  - Boilerplate code generation
  - Debugging assistance
  - Documentation generation
  - Code refactoring
  - Literature integration

#### Computing Infrastructure
- **GPU requirements**: 
  - Minimum: 1× NVIDIA GPU with 8GB VRAM (GTX 1080, RTX 2070, or better)
  - Recommended: 1× V100 (16GB) or A100 (40GB)
  - Cloud alternatives: Google Colab Pro, AWS SageMaker, Lambda Labs
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB for dataset and experiments

#### Estimated Compute Time
- Data preprocessing: 2-4 hours
- EDA and visualization: 2-3 hours
- Baseline model training: 2-4 hours
- Main model training: 10-20 hours (depends on GPU and hyperparameters)
- Interpretability analysis: 5-10 hours
- **Total pipeline**: ~25-40 hours of compute time

### 10.3 Reproducibility

#### Version Control
- Git repository with structured organization
- Branch strategy: main, develop, feature branches
- Regular commits with descriptive messages

#### Environment Management
- **requirements.txt** or **environment.yml** for dependencies
- Pin exact versions for reproducibility
- Docker container (optional but recommended):
  ```dockerfile
  FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
  # Add astronomical packages
  ```

#### Experiment Tracking
- **Weights & Biases** or **MLflow**:
  - Log hyperparameters
  - Track metrics across runs
  - Save model checkpoints
  - Store artifacts (plots, interpretability maps)
- **Random seed fixing**:
  ```python
  import random, numpy, torch
  random.seed(42)
  numpy.random.seed(42)
  torch.manual_seed(42)
  ```

#### Documentation
- README.md with project overview
- INSTALL.md with setup instructions
- API documentation for custom modules
- Jupyter notebooks for EDA and results visualization

### 10.4 Project Structure
```
galaxy-morphology-classification/
├── data/
│   ├── raw/                 # Original GZ2 data
│   ├── processed/           # Preprocessed images
│   └── splits/              # Train/val/test indices
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_main_model.ipynb
│   └── 04_interpretability.ipynb
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── resnet.py
│   │   └── interpretability.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── configs/
│   ├── baseline_config.yaml
│   └── main_config.yaml
├── outputs/
│   ├── models/              # Saved checkpoints
│   ├── figures/             # Plots and visualizations
│   └── results/             # Metrics and reports
├── tests/
│   └── test_*.py
├── requirements.txt
├── README.md
└── plan.md                  # This document
```

## 11. Expected Outcomes & Timeline

### 11.1 Deliverables
1. **Trained model**: Interpretable galaxy morphology classifier
2. **Research report**: Comprehensive analysis document (20-30 pages)
3. **Code repository**: Well-documented, reproducible implementation
4. **Visualizations**: Publication-quality figures and interpretability maps
5. **Presentation**: Conference-style talk or poster

### 11.2 Timeline (Semester Project: 12-16 weeks)

#### Weeks 1-2: Setup & Data Preparation
- Environment setup and literature review
- Data download and exploration
- Preprocessing pipeline implementation

#### Weeks 3-4: Baseline Models
- Implement traditional morphological methods
- Train simple CNN baseline
- Initial evaluation and metrics

#### Weeks 5-8: Main Model Development
- Implement and train ResNet/other architectures
- Hyperparameter tuning
- Architecture ablations
- Performance optimization

#### Weeks 9-11: Interpretability Analysis
- Implement Grad-CAM, Integrated Gradients, SHAP
- Generate saliency maps for representative samples
- Quantitative feature attribution validation
- Astrophysical alignment analysis

#### Weeks 12-14: Evaluation & Analysis
- Comprehensive performance evaluation
- Failure mode analysis
- Astrophysical validation
- Comparison with baselines

#### Weeks 15-16: Documentation & Presentation
- Write final report
- Prepare visualizations
- Create presentation
- Clean and document code

### 11.3 Success Criteria
- **Minimum viable**: 
  - Accuracy ≥ 85% on test set
  - Clear interpretability visualizations
  - Documented, reproducible code
- **Target**:
  - Accuracy ≥ 90% with balanced class performance
  - Quantitative alignment between CNN attention and astrophysical features
  - Publishable-quality analysis
- **Stretch goal**:
  - Graduate-level publication submission
  - Novel astrophysical insights from interpretability analysis
  - Generalization to other survey data

### 11.4 Potential for Publication
This project has strong potential for publication in:
- **Machine Learning venues**: NeurIPS Workshop on ML4PS, ICML Workshop
- **Astronomy venues**: MNRAS, ApJ, A&A (with sufficient novelty)
- **Interdisciplinary**: Nature Astronomy (if major discovery)

**Key requirements for publication**:
- Rigorous quantitative interpretability validation
- Novel astrophysical insights or methodology
- Comprehensive comparison with existing methods
- Clear demonstration of scientific value

---

## Appendix A: Key References

### Machine Learning for Galaxy Morphology
- Dieleman et al. (2015) - Winning Galaxy Challenge solution
- Huertas-Company et al. (2015) - CNN for CANDELS galaxies
- Walmsley et al. (2022) - Galaxy Zoo DECaLS deep learning

### Interpretability in Astronomy ML
- Ribeiro et al. (2020) - XAI in astrophysics survey
- Stein et al. (2022) - Interpretability for stellar parameter estimation
- Cranmer et al. (2020) - Discovering symbolic physics with deep learning

### Galaxy Morphology Background
- Galaxy Zoo papers (Lintott et al. 2008, 2011)
- Conselice (2003) - CAS system
- Lotz et al. (2004) - Gini-M20 system

---

**Note**: This plan is a living document and should be updated as the project progresses. Adjust timelines, methods, and scope based on preliminary results and available resources.