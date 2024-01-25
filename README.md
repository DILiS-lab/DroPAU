# Identifying Drivers of Predictive Aleatoric Uncertainty

Appendices, Supplements, and Code for IJCAI 2024 submission "Identifying Drivers of Predictive Aleatoric Uncertainty".

This README details how to reproduce the results from our manuscript **Identifying Drivers of Predictive Aleatory Uncertainty**. The README is divided into three main sections. The first section is about our benchmarks metric experiments using data with synthetically created uncertainty and real world benchmark datasets. The second section is about out global localization experiments using only synthetic data. The third section is on our illustrative use case of uncertainty in age detection.


### Global Synthetic Benchmark

We provide a command-line interface for running an synthetic uncertainty explanation experiments.
The script will train a heteroscedastic Gaussian neural network on the train data.
We subsequently explain the variance estimates on a test set using variance feature attribution (VFA flavors), infoshap, and Counterfactual Latent Uncertainty Explanations(CLUE). By default, we will run the explainers on 200 test instances with the highest predicted uncertainty "highU", highest predicted confidence "lowU", and random intances "randomU".
For CLUE we adapted code from https://github.com/cambridge-mlg/CLUE .
We use Python 3.11.5.


### How to Use

To run the global_synthetic_benchmark uncertainty explanation experiment, follow these steps:

1. Open a terminal or command prompt.

2. Install the requirements.txt file using `pip install -r requirements.txt`

3. Navigate to the directory `global_synthetic_benchmark`.

4. Use the following command to execute the script:

   ```bash
   python synthetic_uc_epl_experiment.py [options]
   ```

Options:

- `--n_instances_to_explain` (Default: 512): The number of instances to explain in the experiment.

- `--noise_scaler` (Default: 2.0): The noise scaler value used in the experiment. (In the paper this is alpha)

- `--n` (Default: 40000): The number of training instances (20% of these will be used for early stopping.)

- `--n_test` (Default: 1500): The number of data instances.

- `--remake_data`: An optional flag. If specified, it will resample data, if not it will look for an existing dataset with the parameters specified

- `--beta_gaussian`: An optional flag. If specified, beta gaussian loss will be used instead of vanilla gaussian NLL.


### Example Usage

```bash
python synthetic_uc_epl_experiment.py --n_instances_to_explain 256 --explainer_repeats 1 --noise_scaler 3.0 --n 30000 --n_test 2048 --remake_data
```

### Output

The script creates (directed and undirected) feature importances as results. The feature importances can be analyzed using
a) `shap_summaries.ipynb` for Figure 2 of the paper.
b) `plotting/plotting.R` for Figure 3 of the paper.


##  Metrics Benchmark 

Go to the folder `metrics_benchmark`

1. Global Perturbation metrics (will run for all methods and datasets):
```bash
python run_perturbation_exp_global.py
```

2. Local Accuracy Metrics (will run for all methods and the specified datasets):
```bash
python run_localization_exp.py --dataset="<dataset>"
```
Dataset is a selection of `synthetic`, `red_wine`, `ailerons`, and `lsat`. `synthetic` is the default.

3. Local Lipschitz Continuity Metrics (will run for all methods and the specified datasets):
```bash
python run_robustness_lipschitz_exp.py --dataset="<dataset>"
```
Dataset is a selection of `synthetic`, `red_wine`, `ailerons`, and `lsat`. `synthetic` is the default.

## Finding Potential Drivers of Uncertainty in Age Detection
If not specified, all paths refer to files in the `age_detection` directory.

Install the requirements.txt file using `pip install -r requirements.txt`.

### Download the IMBD-clean dataset
Follow the instructions on the [IMBD-clean GitHub repository](https://github.com/yiminglin-ai/imdb-clean) to download the images. The downloaded images (in the numbered directories) need to be stored in a directory called images:
```bash
mivolo/data/dataset/images
```

Additionally, the [MiVOLO specific annotations](https://drive.google.com/file/d/17uEqyU3uQ5trWZ5vRJKzh41yeuDe5hyL/view?usp=sharing)  need to be downloaded. The CSV files for the train, validation, and test sets need to be stored at:
```bash
mivolo/data/dataset/annotations
```
We provide a separate example set, `imdb_example_new.csv`, that contains the example images used in our manuscript.

### Download a MiVOLO checkpoint and the YOLO checkpoint
Refer to the [MiVOLO GitHub](https://github.com/WildChlamydia/MiVOLO) to download an IMBD-clean checkpoint of choice and the [YOLO checkpoint](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view) also provided by the authors of MiVOLO and store them at:
```bash
models/
```
**Note:** We provide the variance fine-tuned checkpoint used in the manuscript: `models/variance_feature_attribution_mivolo_checkpoint.pth.tar`.

**Note:** To use a downloaded checkpoint, you must extend it with a variance output!

### Expanding the model with a variance output
The `variance_output_injection.ipynb` notebook shows how to inject an additional weight vector into the head and auxiliary head and extend the bias.

### Fine-tuning the pre-trained model with Gaussian negative log-likelihood loss
To fine-tune the extended model, use `train_mivolo.py`. To run the training with the parameters from the manuscript, run the following:
```bash
python train_mivolo.py --name="example_run" --learning-rate=0.00001 --weight-decay=0.01 --epochs=50
```

#### Running the model on the test set 
To run the trained model on the test set, refer to the `test_mivolo.ipynb`. The default checkpoint selected in the notebook is the one we trained and used to produce the results in the manuscript.

To reproduce Figure 8 from the Appendix, run the data on the test set and then use `generate_data_for_uncertainty_evaluation.ipynb` to generate the necessary files used for the plot in `plotting/plotting.R`.

### Creating variance explanations
There are two options to create variance explanations:
#### Option 1: Use our example images notebook
To reproduce the explanations for the variance from our age detection example, refer to the `example_explanations.ipynb` notebook. Similar to the testing notebook, it has the checkpoint used to create the figures from the manuscript pre-selected. 
#### Option 2: Create explanations for the whole test set
If you want to create examples for the whole test set, use `create_explanations.py`. The configuration used in the manuscript is:
```bash
python create_explanations.py --checkpoint="variance_feature_attribution_mivolo_checkpoint" --method="hiresCAM"
```

### Calibration
The `age_detection/calibration` folder contains a notebook to recalibrate the uncertainties using std-scaling.

# Licences
We use code from different projects that we build on. In the following, we list the appropriate licenses for different sections of the code that we have used from other work.

#### MiVOLO
We use and adapt various code from [MiVOLO](https://github.com/WildChlamydia/MiVOLO) which is under [this licenses](/age_detection/mivolo_license/en_us.pdf). This involves code in the following files:
- `mivolo/predictor.py` and ``mivolo/predictor_orig.py``
- `mivolo/structures.py`
- All code in `mivolo/data/`
- `mivolo/model/create_timm_model.py`
- `mivolo/model/cross_bottleneck_attn.py`
- `mivolo/model/mi_volo.py` and `mivolo/model/mi_volo_orig.py`
- `mivolo/model/mivolo_model.py`
- `mivolo/model/yolo_detector.py`


#### hugging face PyTorch Image Models timm VOLO
We use and adapt the code in `mivolo/model/volo.py` from [huggingface PyTorch Image Models](https://github.com/huggingface/pytorch-image-models), which is under the [Apache 2.0 license](/age_detection/pytorch_image_models_licence/LICENSE.txt).

#### Transformer-Explanations
We use and adapt the code in `mivolo/model/explanation_generator.py` from [Transformer-Explanations](https://github.com/hila-chefer/Transformer-Explainability), which is under this [MIT Licence](/age_detection/transformer_explainability_licence/LICENSE.txt).

#### Original implementaiton of Lipschitz metric
We use and adapt code in `metrics_benchmark/lipschitz_metric.py` from [Robustness-of-Interpretability-Methods](https://github.com/viggotw/Robustness-of-Interpretability-Methods), which is under this [MIT Licence](metrics_benchmark/Robustness-of-Interpretability-Methods_licence/LICENSE.txt).

#### CLUE

We apadt code in `synthetic_experiments/CLUE` and in the `/synthetic_experiments/synthetic_experiment_utils.py` `explain_clue()` function from [CLUE](https://github.com/cambridge-mlg/CLUE), which is under this [MIT Licence](/synthetic_experiments/CLUE_LICENSE.txt).


