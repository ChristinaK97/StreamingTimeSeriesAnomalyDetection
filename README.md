# StreamingTimeSeriesAnomalyDetection
Mining from Massive Datasets Assignment


### Project Directory Structure:
StreamingTimeSeriesAnomalyDetection (base directory) <br>
|-- __StreamingAnomalyDetectionNotebook.ipynb__: Project notebook containing functions for training and evaluating the AD models, as well as results analysis and plots. <br>
|-- __StreamingAnomalyDetectionNotebook.html__: HTML version of the notebook <br>
|-- __Streaming Time Series Anomaly Detection Project 2 Report__ in pdf and docx formats <br>
|-- __utils__: Package containing functions for reading datasets, handling and plotting results. <br>
|-- __TSB-UAD__: Evaluation benchmark cloned from its original [github repo](https://github.com/TheDatumOrg/TSB-UAD/) <br>
|-- __TSB-UAD-Public__:[Public datasets](https://www.thedatum.org/datasets/TSB-UAD-Public.zip) from the TSB benchmark <br>
|-- __EncDec-AD__: Files copied from the [original repo](https://github.com/KDD-OpenSource/DeepADoTS/tree/master) (modified) <br>
|-- __redsofa_online_ae_lstm__: AE-LSTM model cloned from its [original repo](https://github.com/redsofa/streaming_anomaly_detection) <br>
|-- __OUTPUTS__: Contains json files with the results of AD models (results are grouped by dataset normality)


### TSB-UAD
- Follow the [instructions](https://github.com/ChristinaK97/StreamingTimeSeriesAnomalyDetection/tree/main/TSB-UAD) provided in the original github repo.
to create a conda environment (TSB) with the listed requirements.
- To use the resulting TSB environment in jupyter notebooks run the following commands in anaconda prompt and
then select the TSB kernel:
```
conda activate TSB
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=TSB
```
- Install additional requirements for other frameworks

### TSB-UAD-Public
- Download the [public datasets](https://www.thedatum.org/datasets/TSB-UAD-Public.zip) provided by the TSB-UAD benchmark


### EncDec-AD
- Offline baseline
- Paper: [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148)
- Original Repository: [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS/tree/master)
- Requirements: <br>
  ```
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  ```
  
### EncDec-AD-Batch & OnlineEncDec-AD
- Streaming variants of the baseline EncDec-AD model, originally proposed for offline settings
- The code of these models can be found in the __StreamingAnomalyDetectionNotebook.ipynb__ notebook in the base directory of this project.

    
### Online AE-LSTM
- Additional online baseline
- Paper [An LSTM Encoder-Decoder Approach for Unsupervised Online Anomaly Detection in Machine Learning Packages for Streaming Data](https://ieeexplore.ieee.org/document/10020872)
- Original Repository: [streaming_anomaly_detection](https://github.com/redsofa/streaming_anomaly_detection)
- Requirements: <br>
  ```
  # GPU support for tensorflow
  conda install -c conda-forge cudatoolkit=11.6 cudnn=8.1.0
  pip install mplcursors==0.4
  pip install pysad==0.1.1
  pip install tdigest
  pip install scikit-multiflow
  pip install seaborn
  pip install combo
  pip install rrcf
  # pip uninstall scikit-learn
  # Found existing installation: scikit-learn 1.3.2
  pip install scikit-learn==0.23.0
  ```
  
### Additional Requirements
  ```
  pip install openpyxl
  ```
