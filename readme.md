# KANN
This repository is the implementation of The Knowledge-aware Attentional Neural Network for Movie Recommendation (KANN), which has been accepted by Neural Computing and Applications (NCAA).

KANN is proposed to predict ratings and provide knowledge-level explanations by capturing understandable interactions between users and items at the knowledge level.
## Environments
- python 3.8.11
- tensorflow 2.6
- numpy
- scipy
## Files in the folder
- data/
  - amazon/: the dataset of Amazon movies & TV;
    - amazon.csv: the raw data of amazon dataset;
    - entity_emb50_concat_mean.bin: the context knowledge embeddings of amazon review entities;
    - entity_oriemb50.bin: the knowledge embeddings of amazon review entities;
    - entity_index: the indices of review entities.
    - 10core/: the preprocessed dataset with 10-core;
      - amazon.csv: the raw data of amazon's 10core dataset.
      - amazon.para: the parameters of the dataset (eg., the number of users/items);
      - amazon.test: randomly selected test dataset;
      - amazon.train: randomly selected train dataset;
      - amazon.valid: randomly selected valid dataset;
      - amazon_test.csv: raw test rating dataset;
      - amazon_train.csv: raw train rating dataset;
      - amazon_valid.csv: raw valid rating dataset;
      - entity_index: the indecies of review entities;
      - item_review: the dict of item reviews;
      - item_rid: item dict, key:item, value:users (who interact with the item);
      - user_review: the dict of user reviews;
      - user_rid: user dict, key:user, value:items (those reviewed by the user);
  - imdb/: the dataset of IMDb dataset;
    - imdb.csv: the raw data of imdb dataset;
    - entity_emb50_concat_mean.bin: the contex knowledge embeddings of imdb review entities;
    - entity_oriemb50.bin: the knowledge embeddings of imdb review entities;
    - 10core/: the preprocessed dataset with 10-core.
    
- output/
  - 10core/
    - checkpoints/: the trained models;
    - new_2layers_50d_4heads_1024dff_128batch_size/: the name is the hyper-parameters of KANN;
      - ckpt, etc.: the saving models of last 5 epoches;
    - logs/new_2layers_50d_4heads_1024dff_128batch_size: the saving results of KANN by recording into tensorboard.
- src
  - CustomSchedule.py: the customed parameter of attention mechanism;
  - Interacter.py: the encoding based on InteracterLayer;
  - InteracterLayer.py: the interaction layer for implementing inner and outer-attention mechanisms;
  - KANN.py: the implementation of KANN;
  - KANN_main.py: the main implementation by using one GPU;
  - KANN_multi_main.py: the main implementation by using multiple GPUs based on tf.distribute;
  - MultiHeadAttention.py: KANN's multi-head parallel processing;
  - build_dataset.py: the second step of preprocessing the raw dataset;
  - read_ratings.py: the first step of preprocessing the raw dataset.
## The description of datasets
- amazon.csv
  - user_id: user ids.
  - item_id: item ids.
  - ratings: ratings from users to items.
  - reviews: ids of review entities.
- imdb.csv
  - user_id: user ids.
  - item_id: item ids.
  - ratings: ratings from users to items.
  - reviews: ids of review entities.
## Running the code
### For one GPU:
- cd src
- python read_rating.py
- python build_dataset.py
- python KANN_main.py
### For multiple GPUs:
- cd src
- python read_rating.py
- python build_dataset.py
- python KANN_multi_main.py
