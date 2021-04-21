# Wiki2Prop
The companion material for the [WWW2021 Wiki2Prop Paper](https://doi.org/10.1145/3442381.3450082)


## Pipeline

### Preparing the Wikidata Graph
The pipeline starts with [wd-graph](https://github.com/eXascaleInfolab/wd-graph) transformed version of Wikidata.

* `01_extract.py` : Extract the properties per article.
* `01_extract_classes.py` : Extract the classes for Recoin.

### Preprocess Wikipedia and Images
Join with the Wikipedia embdings gained with [Wikipedia2Vec](https://wikipedia2vec.github.io/wikipedia2vec/).

* `02_preprocess_w2v.py` : Join Wikidata Properties with Embeddings.
* `02_preprocess_image.py` : Embed the images with Densenet.

### Training or Building Baselins
For recoin collect the statistics, for BPMLL train the network.

* `03_build_recoin.py` : For recoin we build on the same input data the statistics.
* `03_bpmll_train.py` : Train the BPMLL network.

### Training Wiki2Prop
Training first independently and in parallel the languages, then the fusion layer.

* `04_wiki2Prop_train_partial.py`: Training per Language.
* `04_wiki2Prop_train_final.py`: Train the Fusion.

### Evaluation
First automatically pick the thr, then output the different evaluations.

* `05_model_pick_thr.py`: Automatically pick the prediction threshold for the non-ranking metrics.
* `05_model_eval_per_property.py` : Output per property.
* `05_model_eval_per_property_recoin.py` : Output per property Recoin.
* `05_model_test.py` : Output the predictions.


### Misc
`data_loader.py` : Help with manipulatiog files.
`tools.py`: The definition of the metrics.


