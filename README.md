<!-- ABOUT THE PROJECT -->
## KANs for clinical classification

Kolmogorov–Arnold Networks (KANs) are a novel neural architecture that enable interpretable learning by construction. In this project, we explore their use for clinical classification tasks using tabular medical data. We introduce two models:
- **Logistic-KAN**, a flexible generalization of logistic regression.
- **KAAM**, an additive KAN-based model for inherently interpretable predictions.

These models offer not only competitive accuracy but also **built-in explainability**, including symbolic formulas, personalized explanation plots, and retrieval of similar patients.


This repository provides:
- Scripts to train and evaluate various models: MLP, Logistic Regression (LR), Random Forest (RF), Neural Additive Models (NAM), Logistic-KAN, and KAAM.
- Pre-processed, ready-to-use datasets included.
- Built-in validation metrics using `scikit-learn`.
- Visualization scripts for individual and global interpretability.
- Pre-trained models to save training time.
- Scripts to replicate results from the paper.

For more details, see full paper [here](https://doi.org/10.48550/arXiv.2509.16750).


<!-- GETTING STARTED -->
## Getting Started
Follow these simple steps to make this project work on your local machine.

### Prerequisites
You should have the following installed on your machine:

* Python 3.9.0
* Required packages
  ```sh
  pip install -r requirements.txt
  ```

### Installation

Download the repo manually (as a .zip file) or clone it using Git.
   ```sh
   git clone https://github.com/Patricia-A-Apellaniz/classification_with_kans.git
   ```

Already trained models and dictionaries with results can be found in /classification_with_kans/results_metrics/.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

You can modify training and model configurations in `utils.py`.

### 1. Train and Evaluate Models

To train, test, and evaluate all models (MLP, LR, RF, NAM, Logistic-KAN, KAAM), run:
   ```sh
   python src/main_performance.py
   ```
This script outputs standard classification metrics such as accuracy, AUC, F1-score, precision, and recall.

### 2. Explore Explainability Tools
To generate visual explanation tools (e.g., partial dependence plots, radar plots, nearest-patient comparisons), run:
   ```sh
   python src/main_interpretability.py
   ```

These plots illustrate the explainability capabilities of the proposed models and are saved in the /results/ directory, grouped by dataset and patient.

### 3. Launch the Interactive Interface

For the heart clinical dataset use case, an interactive interface is available on Hugging Face Spaces. You can launch it here:
   ```
https://huggingface.co/spaces/marElizo/classification_with_kans
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



[//]: # (<!-- LICENSE -->)

[//]: # (## License)

[//]: # ()
[//]: # (Distributed under the XXX License. See `LICENSE.txt` for more information.)

[//]: # ()
[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



<!-- CONTACT -->
## Contact

Patricia A. Apellániz - patricia.alonsod@upm.es

Mar Elizo - mar.ealonso@upm.es

<p align="right">(<a href="#readme-top">back to top</a>)</p>


[//]: # (<!-- ACKNOWLEDGMENTS -->)

[//]: # (## Acknowledgments)

[//]: # ()
[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)

