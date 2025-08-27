# mini_Supervised_Unsupervised_Deep_Learning_GenAI_projects

A collection of compact hands-on projects covering supervised, unsupervised, deep learning, and generative AI topics. Each project is self-contained and demonstrates typical data preprocessing, model building, training, and evaluation workflows for small real-world or educational datasets.

## Projects (one- to two-sentence summaries)

- decision_tree_project.py  
  Uses the UCI Flags dataset to train decision tree classifiers (comparing depths), prints summary statistics for selected landmasses, and visualizes accuracy vs. tree depth and the best tree.

- Word2Vec.ipynb  
  Demonstrates text preprocessing and training of Word2Vec embeddings (example uses "Alice in Wonderland"), with tokenization and gensim-based embedding experiments.

- admissions_DL.ipynb  
  Builds a neural network to predict university admission chances from features like GRE, TOEFL, CGPA, and other application metrics, including preprocessing, scaling, and evaluation.

- Covid19_prediction.ipynb  
  Implements an image-classification pipeline using Keras ImageDataGenerator and a CNN to classify chest X-ray images for COVID-19/related classes, with data augmentation and evaluation utilities.

- heart_failure_prediction.py  
  Loads a heart-failure dataset, preprocesses features (scaling and encoding), trains a small Keras neural network to predict death events, evaluates performance, and includes an example inference snippet.

- Machine_translation.ipynb  
  Experiments with transformer-based sequence-to-sequence translation (using Hugging Face / Helsinki-NLP model checkpoints), dataset loading, tokenization, and BLEU-style evaluation for machine translation tasks.

- life_expectancy_DL.ipynb  
  Trains a regression neural network to predict life expectancy from WHO-style data, covering data cleaning, feature engineering, scaling, model training, and regression evaluation.

- Hyperparamater_tuning.ipynb  
  Shows hyperparameter exploration and tuning (example uses a raisins/classification dataset), including model selection experiments and visualization of tuning results.

- cancer_classifier_with_KNeighborsClassifier.py  
  Trains a K-Nearest Neighbors classifier on the scikit-learn breast cancer dataset, evaluates validation accuracy for many k values, and plots accuracy vs k to help choose a neighborhood size.

- PCA Classification Particles/  
  Demonstrates dimensionality reduction with PCA on synthetic or real datasets and follows with classification and visualizations (particle-style scatter/cluster visualizations and decision-boundary illustration) to show how PCA affects separability and classifier performance.

- requirements.txt  
  Lists core Python package dependencies (pandas, tensorflow, scikit-learn, matplotlib, seaborn, numpy) required to run many of the notebooks and scripts.

## Getting started

1. Clone the repository:
   git clone https://github.com/deathvadeR-afk/mini_Supervised_Unsupervised_Deep_Learning_GenAI_projects.git

2. Create and activate a virtual environment (recommended) and install dependencies:
   pip install -r requirements.txt

3. Open the notebooks in Jupyter / JupyterLab or Google Colab (many notebooks include a Colab badge and are ready to run in Colab).

4. Inspect individual project README or notebook cells for dataset download instructions; several notebooks download or expect CSV/image datasets to be present.

## Notes and tips

- Many notebooks are designed for educational exploration; check top cells for dataset paths and adjust as needed.  
- For transformer or heavier models, use a GPU runtime (e.g., Colab GPU) to speed up training.  
- Some notebooks may require extra packages (e.g., `transformers`, `datasets`, `sacrebleu`) beyond requirements.txt — install them as needed per notebook.

## Contributing

Contributions, improvements, and bug fixes are welcome. Please open issues or PRs with short descriptions and reproducible steps.

## License

No license file is included in the repository; treat the code as "as-is" until an explicit license is added. Consider adding an OSI-compatible license (e.g., MIT) if you want others to reuse the work.
