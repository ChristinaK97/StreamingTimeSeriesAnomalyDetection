# StreamingTimeSeriesAnomalyDetection
Mining from Massive Datasets Assignment

### REPORT DOCUMENT LINK
[Report](https://aristotleuniversity-my.sharepoint.com/:w:/g/personal/kchristi_office365_auth_gr/ETiycbkhKl5PrxVD0GtVZS8BLHUS7wqk7581xY9O2Ul1uA?e=7KmeR6)

### Project Directory Structure:
StreamingTimeSeriesAnomalyDetection (base directory) <br>
|-- TSB-UAD (cloned from the original [github repo](https://github.com/TheDatumOrg/TSB-UAD/)) <br>
|-- TSB-UAD-Public ([public datasets](https://www.thedatum.org/datasets/TSB-UAD-Public.zip) from the TSB benchmark) <br>
|-- EncDec-AD (contains files copied from the [original repo](https://github.com/KDD-OpenSource/DeepADoTS/tree/master) - modified) <br>
|-- redsofa_online_ae_lstm (cloned from the [original repo](https://github.com/redsofa/streaming_anomaly_detection)) <br>
|-- StreamingAnomalyDetectionNotebook.ipynb (draft project notebook) <br>


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
- Paper: [Long short term memory networks for anomaly detection in time series](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56), ESANN 2015 
- Original Repository: [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS/tree/master)
- Requirements: <br>
  ```
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  ```
    
### Online AE-LSTM
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
