# Automlizer
A Realtime Application to analyze the performance of AutoML Pipelines

### Documentation

Refer to the [AutoMLizer Blog]("https://0bsidiansteel.com/posts/automlizer"){:target="_blank"}

* `assets` include the required css, images and other required js files
* `aml_config.py` is the config file for TPOT's hyperparameter search. For more information refer to [TPOT documentation]("https://epistasislab.github.io/tpot/using/#customizing-tpots-operators-and-parameters"){:target="_blank"}
* `app.py`: The main application
* `aml.py`: Runs TPOT
* `util.py`: Set of utility functions required by automlizer


### Installation and Execution

1. Clone the repository
2. `pip install -r requirements.txt`
3. `python3 app.py`

### Limitations

1. Currently, the relationship between the selection of the label and the problem type is not coded - hence selecting the correct problem type and metric is essential
2. This is a non-database lite version of the actual application and hence the state is not managed. 