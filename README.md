# ExtendedSustainaML

Extended SustainaML is an interactive research framework for **sustainable, explainable, and human-centered Automated Machine Learning (AutoML)**.

It extends the original SustainaML system beyond performance and sustainability visualization by introducing **dataset-aware recommendations, classification and clustering workflows, audience-adaptive explanations, interactive explainability, sustainability-aware model comparison, and recommendation validation**.

The overall objective is to help users not only identify effective machine-learning solutions, but also understand model behavior, examine performance–sustainability trade-offs, and interactively evaluate alternative configurations.

---

## Key Features

### Multi-Framework AutoML

Extended SustainaML provides a unified interface for experimenting with multiple AutoML frameworks, including:

- FLAML
- H2O
- MLJAR

Users can configure experiments, compare candidate solutions, and examine their predictive and sustainability-related characteristics through a common interface.

---

### Classification Workflow

The classification environment supports interactive training, comparison, and analysis of candidate models.

Users can explore:

- predictive performance,
- alternative algorithms and configurations,
- feature-level model behavior,
- sustainability measurements,
- model explanations,
- performance–sustainability trade-offs, and
- recommendations for subsequent experiments.

The workflow is designed to expose information that is normally hidden inside an automated model-selection process and provide users with greater control over AutoML experimentation.

---

### Clustering Workflow

Extended SustainaML also provides a dedicated **unsupervised clustering workflow**.

The clustering environment supports the configuration and comparison of alternative clustering solutions using clustering-specific evaluation measures rather than classification accuracy.

The interface combines clustering-quality evaluation with sustainability measurements and visual analysis, allowing users to inspect both the discovered data structure and the computational implications of alternative configurations.

The clustering workflow includes:

- configuration of clustering experiments,
- comparison of alternative clustering solutions,
- internal clustering-quality evaluation,
- sustainability measurement,
- visual exploration of cluster structure,
- performance–sustainability comparison,
- audience-adaptive interpretation,
- configuration recommendations, and
- validation of recommended changes.

This provides a parallel decision-support workflow for unsupervised learning while retaining evaluation criteria appropriate to clustering.

---

### Dataset-Aware Meta-Learning Recommendations

Extended SustainaML incorporates a **meta-learning layer** to provide dataset-aware recommendations.

Instead of selecting configurations solely from predefined defaults, the system characterizes an uploaded dataset using dataset-level meta-features and uses knowledge obtained from previously benchmarked datasets to recommend suitable configurations for the supported AutoML frameworks.

The general process is:

```text
New Dataset
    ↓
Dataset Meta-Feature Extraction
    ↓
Meta-Learning Models
    ↓
Framework-Specific Recommendation
    ↓
Suggested Algorithm / Configuration
```

The meta-learning component is intended to provide an informed starting point for AutoML experimentation rather than replacing subsequent empirical evaluation.

---

### Sustainability-Aware Evaluation

Extended SustainaML evaluates machine-learning experiments from both **model-quality and sustainability perspectives**.

Alongside conventional machine-learning metrics, the system incorporates estimates of:

- energy consumption,
- carbon emissions, and
- computational resource implications.

CodeCarbon is integrated into the experimental pipeline to support energy and carbon estimation.

These measurements are intended to complement predictive or clustering-quality metrics rather than replace them. Sustainability measurements should be interpreted as estimates whose precision depends on the available hardware-monitoring mechanisms and execution environment.

---

### Performance–Sustainability Trade-off Exploration

Selecting the model with the highest predictive score does not necessarily result in the most computationally efficient solution.

Extended SustainaML therefore provides visual mechanisms for examining relationships between **model quality and sustainability cost**.

The interface helps users investigate whether a relatively small change in predictive or clustering quality is associated with a meaningful change in energy consumption or carbon emissions.

This enables model selection to be treated as a multi-dimensional decision rather than a single-metric optimization problem.

---

### Explainable and Interactive Model Analysis

Extended SustainaML incorporates explainability mechanisms to help users understand model behavior rather than treating AutoML results as opaque outputs.

The interface provides model- and feature-level information together with interactive visualizations that support investigation of:

- important features,
- prediction behavior,
- model confidence,
- feature influence, and
- relationships between model behavior and selected configurations.

---

### Counterfactual and What-If Exploration

For classification experiments, Extended SustainaML provides interactive **counterfactual and what-if exploration**.

Users can modify feature values and observe how the selected trained model responds to those changes.

This supports questions such as:

> What happens to the model prediction if this feature changes?

and

> Which changes can alter the current prediction?

The component is intended as an interactive perturbation and sensitivity mechanism. It should not be interpreted as guaranteeing a globally minimal or causally valid counterfactual.

---

## Audience-Adaptive Explanations

A central extension of SustainaML is the **audience-adaptive explanation layer**.

The same machine-learning result may need to be communicated differently depending on the user's technical expertise. Extended SustainaML therefore supports multiple audience perspectives, including:

- **Normal User**
- **Domain Expert**
- **Data Scientist**

The explanation layer adapts the terminology, technical depth, and interpretation of the available evidence according to the selected audience.

For example, a normal user can receive a concise interpretation of the result without unnecessary machine-learning terminology, while a data scientist can receive a more technical description involving evaluation metrics, model characteristics, feature information, and sustainability measurements.

The objective is not to change the underlying experimental evidence, but to change **how that evidence is communicated**.

---

## LLM-Supported Explanation and Recommendation

Extended SustainaML supports locally hosted language models through **Ollama** for audience-adaptive explanation and constrained recommendation generation.

The default implementation uses **Gemma 3 4B (`gemma3:4b`)** through a local Ollama instance.

The LLM acts primarily as a **communication and decision-support layer**. Experimental metrics and validation outcomes are calculated by the application rather than generated by the language model.

Prompt constraints are used to keep explanations grounded in the supplied experimental evidence. The LLM is instructed not to invent unavailable feature meanings, experimental results, or unsupported optimization claims.

Recommendations are similarly constrained to actions that can actually be performed through the Extended SustainaML interface.

---

## Recommendation and Validation Loop

Extended SustainaML moves beyond passive explanation by introducing an interactive **recommendation–validation loop**.

After examining the current experiment, the system can propose an actionable configuration change based on controls available within the dashboard.

A recommendation is treated as an **experimental hypothesis**, not as a guaranteed improvement.

The user can apply the recommendation and rerun the corresponding experiment. The new result is then compared with the original baseline.

The workflow is:

```text
AutoML / Clustering Result
          ↓
Audience-Adaptive Explanation
          ↓
Actionable Recommendation
          ↓
Experimental Hypothesis
          ↓
Apply Recommendation
          ↓
Rerun Experiment
          ↓
Before vs. After Comparison
          ↓
Repeated Validation
          ↓
Audience-Adaptive Interpretation
```

This allows users to experimentally challenge a recommendation instead of simply accepting it.

---

## Before–After Comparison

When a recommendation is validated, Extended SustainaML preserves the original configuration as a **baseline**.

The recommended configuration is then evaluated and compared with the baseline using relevant model-quality and sustainability measures.

Depending on the observed changes, the interface can communicate whether the new configuration represents:

- an improvement,
- a sustainability improvement,
- a performance–sustainability trade-off,
- a deterioration, or
- no clear change.

This distinction is important because a recommended configuration is not considered successful merely because it was suggested by the system.

---

## Repeated Recommendation Validation

A single experimental rerun can be affected by stochastic variation.

Extended SustainaML therefore supports **repeated paired validation**, in which the baseline and recommended configurations are evaluated repeatedly under matched experimental conditions.

The resulting distributions can be compared using summary statistics and practical-effect criteria.

Repeated validation provides stronger empirical evidence about whether an observed change is reasonably consistent across runs.

The current decision rules represent **practical-effect thresholds rather than statistical-significance tests**.

---

## Classification and Clustering as Parallel Workflows

Extended SustainaML treats classification and clustering as related but distinct experimental workflows.

```text
                         Dataset
                            │
             ┌──────────────┴──────────────┐
             │                             │
      Classification                  Clustering
             │                             │
     Model Configuration           Cluster Configuration
             │                             │
    Predictive Evaluation          Cluster Evaluation
             │                             │
       Explainability              Visual Exploration
             │                             │
             └──────────────┬──────────────┘
                            │
                 Sustainability Analysis
                            │
                Audience-Adaptive Explanation
                            │
                     Recommendation
                            │
                    Apply & Validate
                            │
                  Repeated Validation
```

This design allows the interface to retain a consistent human-centered workflow while using evaluation mechanisms appropriate to each machine-learning task.

---

## Typical Workflow

A typical Extended SustainaML experiment consists of:

1. Uploading and inspecting a dataset.
2. Selecting classification or clustering.
3. Configuring the corresponding experiment.
4. Obtaining dataset-aware recommendations where applicable.
5. Running candidate models or clustering configurations.
6. Comparing model/cluster quality.
7. Inspecting energy and carbon estimates.
8. Exploring visual and explainability information.
9. Selecting an audience-specific interpretation.
10. Reviewing an actionable experimental recommendation.
11. Applying the recommendation.
12. Comparing the new result with the original baseline.
13. Optionally performing repeated validation.
14. Interpreting the resulting performance–sustainability relationship.

---

## High-Level Architecture

```text
                    ┌─────────────────────┐
                    │       Dataset       │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
           Meta-Learning              User Configuration
          Recommendation                    │
                  │                         │
                  └────────────┬────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  ML Experiment     │
                    │ Classification /   │
                    │    Clustering      │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
        Model Quality     Sustainability      XAI / Visual
          Metrics           Metrics             Analysis
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                    Audience-Adaptive
                       Explanation
                               │
                       Recommendation
                               │
                       Apply & Validate
                               │
                    Repeated Validation
```

---

## Technology

Extended SustainaML integrates several tools and libraries across its experimental pipeline, including:

- Python
- Streamlit
- FLAML
- H2O
- MLJAR-related model experimentation
- scikit-learn
- TabPFN
- PyMFE
- CodeCarbon
- Ollama
- Gemma 3

Additional dependencies are listed in the project environment or requirements configuration.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

Create and activate an appropriate Python environment and install the project dependencies.

```bash
pip install -r requirements.txt
```

Depending on the enabled functionality, additional local services or dependencies may be required, including H2O and Ollama.

---

## Local LLM Setup

Audience-adaptive LLM functionality can be run locally using Ollama.

After installing Ollama, obtain the required model:

```bash
ollama pull gemma3:4b
```

Verify that the model is available:

```bash
ollama list
```

The application can then communicate with the local Ollama service for audience-adaptive explanation and recommendation generation.

---

## Running the Application

Launch the required backend components according to the repository configuration and start the Streamlit interface.

For example:

```bash
streamlit run frontend.py
```

The exact startup procedure may depend on the current repository configuration and enabled AutoML frameworks.

---

## Research Context

Extended SustainaML builds upon the original **SustainaML** research prototype.

The extended work investigates how sustainable AutoML interfaces can evolve from passive visualization toward an interactive, human-centered decision-support process combining:

- multi-framework AutoML,
- classification and clustering,
- dataset-aware meta-learning,
- sustainability-aware evaluation,
- explainable machine learning,
- interactive what-if analysis,
- audience-adaptive communication,
- actionable recommendations, and
- empirical recommendation validation.

The central research direction is to support users in understanding not only **which machine-learning solution performs well**, but also **why it behaves as it does, what computational and environmental costs are associated with it, and whether proposed alternatives actually improve the observed result**.

---





