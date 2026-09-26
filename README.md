# BootstrapMD

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React 19](https://img.shields.io/badge/React-19.2-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.2-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4.2-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BootstrapMD** is a clinical-grade medical and tabular data augmentation tool designed to generate statistically sound synthetic datasets from small sample cohorts. It couples **smoothed bootstrap resampling** and **k-nearest neighbors (k-NN) conditional sampling** with **early data type evaluation** and a comprehensive **statistical validation engine**.

---

## 🌟 Key Features

### 1. Early Data Type Evaluation & Manual Overrides
Small datasets (e.g., $N < 50$) often confuse automated type detectors when discrete numerical codes represent categorical states (e.g., `treatment_group`: `1, 2, 3` or `nyha_class`: `I, II, III, IV`).
- **Heuristic Type Profiling**: Evaluates uniqueness, cardinality ratios, sample ranges, and missing rates immediately upon upload.
- **Tri-Type Classification**: Explicitly supports **Continuous**, **Categorical**, and **Ordinal** data types.
- **Interactive Override Workspace**: Provides an intuitive column-by-column inspection grid to reclassify variables with instant feedback before running synthesis.
- **Safety Net**: Changing data types dynamically recalculates all statistical tests without corrupting the raw data.

### 2. Statistically Sound Data Augmentation
- **Continuous Variables**:
  - **Smoothed Bootstrap (Gaussian Jitter)**: Introduces controlled perturbation scaled to feature variance ($\mathcal{N}(0, 0.05 \times \sigma)$).
  - **Range & Plausibility Clamping**: Constrains generated values to biological or empirical $[\min, \max]$ boundaries to prevent physiologically impossible figures.
  - **Precision Preservation**: Retains natural precision and decimal places (e.g., rounding discrete integers vs. high-precision lab assays).
- **Categorical & Ordinal Variables**:
  - **k-NN Conditional Modeling ($k=5$)**: Preserves multidimensional feature correlations by finding the nearest neighbors in normalized Euclidean space.
  - **Categorical Mismatch Penalty**: Accounts for discrete distance penalties to maintain realistic feature co-occurrence.
  - **Stochastic Weighted Voting**: Preserves class balance and conditional distribution frequencies.

### 3. Dual-Validation Statistical Suite
Every augmented dataset is rigorously compared against the ground truth across multiple parametric and non-parametric tests:

| Variable Type | Statistical Test | Purpose & Metric |
| :--- | :--- | :--- |
| **Continuous** | **Two-Sample Student's t-Test** | Evaluates equality of means ($\alpha = 0.05$) |
| **Continuous** | **Mann-Whitney U Test** | Non-parametric rank-sum test for stochastic equivalence |
| **Continuous** | **Kolmogorov-Smirnov (KS) Test** | Compares overall empirical cumulative distribution functions |
| **Continuous** | **95% Confidence Intervals** | Quantifies absolute mean shift between datasets |
| **Categorical / Ordinal** | **Pearson's Chi-Square Test** | Assesses goodness-of-fit across category proportions |
| **Categorical / Ordinal** | **Cramér's V** | Measures association strength ($V < 0.1$ denotes strong alignment) |
| **Categorical / Ordinal** | **Total Variation Distance (TVD)** | Quantifies total probability difference ($\text{TVD} < 0.1$) |

### 4. Interactive Visualizations & Summary Reports
- **Interactive Histograms**: Side-by-side distribution plots comparing original vs. augmented datasets for any selected variable.
- **Comprehensive CSV Summary**: Downloadable report compiling descriptive statistics, p-values, test statistics, effect sizes, and pass/fail similarity indicators.
- **Text Report Export**: Human-readable executive statistical summary.
- **AI Statistical Assistant**: Embedded Gemini-powered consultant to answer questions and interpret complex statistical outcomes.

---

## 🚀 Quick Start

### Prerequisites
- [Node.js](https://nodejs.org/) (version 18.0.0 or higher recommended)
- `npm` or `bun`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/smoothed-bootstrap-data-augmenter.git
   cd smoothed-bootstrap-data-augmenter
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure Environment Variables (Optional):**
   To enable the AI Statistical Assistant, set your Gemini API key:
   ```bash
   cp .env.example .env
   # Add your Google Gemini API key:
   # API_KEY=your_gemini_api_key_here
   ```

4. **Start the local development server:**
   ```bash
   npm run dev
   ```
   Open your browser and navigate to `http://localhost:3000`.

---

## 🛠️ Usage Workflow

1. **Upload Dataset**: Upload any tabular `.csv` file containing clinical, biological, or experimental data.
2. **Review & Adjust Data Types**:
   - Inspect the **Data Type Evaluation** view.
   - For any column where integer codes represent categories (e.g., `0` and `1` for binary status, or `1, 2, 3` for stage), switch the dropdown to **Categorical** or **Ordinal**.
3. **Configure Target Sample Size**: Choose your target augmented sample size (e.g., $N = 500$, $1000$, or $5000$).
4. **Generate Augmented Data**: Click **Synthesize Data**.
5. **Inspect Validation Tabs**:
   - **Summary**: Quick glance at similarity metrics across all tests.
   - **Histograms**: Visual density checks of original vs. synthetic distributions.
   - **T-Test & Mann-Whitney**: Review central tendency and rank-sum p-values.
   - **KS-Test**: Ensure distribution shapes match empirical observations.
   - **Categorical Tests**: Review Chi-Square, Cramér's V, and TVD scores.
6. **Export**: Download the synthetic dataset (`.csv`) and the statistical validation audit trail (`.csv` or `.txt`).

---

## 📁 Project Architecture

```
├── components/
│   └── DataTypeEvaluationWorkspace.tsx # Early data type evaluation and override UI
├── services/
│   ├── dataAugmentation.ts             # Smoothed bootstrap & k-NN augmentation engine
│   └── statisticalAnalysis.ts          # Parametric, non-parametric, & distance tests
├── App.tsx                             # Main workflow coordinator and UI tabs
├── types.ts                            # Core TypeScript interfaces and type definitions
├── index.html                          # Entry HTML document
├── index.tsx                           # React DOM root entry
├── package.json                        # Scripts and dependencies
└── vite.config.ts                      # Vite build configuration
```

---

## 🔬 Mathematical Methodology

### Smoothed Bootstrap
For a continuous feature $X_j$ with empirical standard deviation $s_j$:
$$X_j^* = X_{j, \text{seed}} + \epsilon_j, \quad \epsilon_j \sim \mathcal{N}\left(0, (0.05 \cdot s_j)^2\right)$$
Subject to physiological boundary preservation:
$$X_j^* = \max\left(\min(X_j), \min\left(\max(X_j), X_j^*\right)\right)$$

### Conditional Categorical Assignment
For categorical attribute $Y_k$, assignment is conditioned on normalized continuous distance $D(x^*, x_i)$ and discrete matching:
$$D(x^*, x_i)^2 = \sum_{j \in \text{continuous}} \left(\frac{x_j^* - x_{ij}}{s_j}\right)^2 + \sum_{m \neq k} \mathbb{I}(y_m^* \neq y_{im}) \cdot \omega$$
The value is sampled proportionally across the $k$-nearest empirical neighbors ($k=5$).

---

## 📜 Scripts

| Command | Action |
| :--- | :--- |
| `npm run dev` | Starts the Vite development server on `http://localhost:3000` |
| `npm run build` | Compiles TypeScript and builds production-ready static assets |
| `npm run lint` | Type-checks code using the TypeScript compiler (`tsc --noEmit`) |
| `npm run preview` | Previews the production build locally |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
