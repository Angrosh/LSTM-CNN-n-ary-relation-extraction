## LSTM-CNN-n-ary-relation-extraction

This project provides Tensorflow based implementation of LSTM-CNN model for n-ary relation extraction for the paper, [Combining Long Short Term Memory and Convolutional Neural Network for Cross-Sentence n-ary Relation Extraction](https://openreview.net/forum?id=Sye0lZqp6Q).

### Requirements

* Python 2.7
* Tensorflow >= 1.6

### Data

This paper uses [dataset](https://drive.google.com/drive/folders/1Jgw6A08nh-4umCV7tfqQ6HFg7mtDwo67) developed by [Peng et al., 2016](https://transacl.org/ojs/index.php/tacl/article/view/1028).

### Usage

The original dataset is preprocessed for the LSTM-CNN model and is provided in data.zip. Unzip data.zip to obtain 'data' folder consisting preprocessed data for different relations.

The model uses Glove pre-trained embeddings. Download glove.6B.300d.txt into 'glove' floder.

Five-fold cross-validation is conducted. The model is run on different folds using the following and the average is taken across five folds:

* python train_lstm_cnn.py data/drug_var_cross_sents/fold_0/ ./training_config.json
