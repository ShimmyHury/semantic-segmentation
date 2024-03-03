
# Semantic Segmentation

Semantic segmentation using Pytorch Unet implementation

## Features

- Train Unet model for semantic segmentation tasks
- Choose from different available loss functions (Tversky, Focal and Cross-Entropy)
- Data augmentation, early stopper for training, adaptive learning rate.

## Instructions

Please note: the data used for this project is the 12 classes CamVid dataset.
each of the 12 classes has a mask value [0-11] (order of classes is specified in the Dataset class)
To train using different data, revisions to the list of classes and mapping of the masks in the Dataset class are required


1. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
2. Make sure the data (images and masks) for the training, validation and test is structured relative to the directory supplied as follows: 

```bash
data_dir:.
        ├───test
        ├───testannot
        ├───train
        ├───trainannot
        ├───val
        └───valannot
```
3. Train your model using the following command line input:
```bash
python main_tmp.py -d DATA_DIR -m MODEL_PATH -l LOSS -n NUM_CLASSES -e EPOCHS -b BATCH -r LEARNING_RATE -f LR_DECREASE_FACTOR -p LR_PATIENCE -s EARLY_STOPPER_PATIENCE -c DELTA
```
where the arguments are as follows:

```bash
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        Path to training data directory (default: ./data/CamVid/)
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to model (save/load) (default: ./trained_model)
  -l LOSS, --loss LOSS  Loss function, one of [Focal,Tversky,CrossEntropy] (default: Tversky)
  -n CLASSES, --classes CLASSES
                        classes for detection. subset of the classes specified in the Dataset class (default: ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car',      
                        'pedestrian', 'bicyclist', 'unlabelled'])
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs for training (default: 1)
  -b BATCH, --batch BATCH
                        Batch size (default: 4)
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Initial learning rate (default: 0.0001)
  -f LR_DECREASE_FACTOR, --lr_decrease_factor LR_DECREASE_FACTOR
                        Learning-rate-decrease-factor (default: 0.1)
  -p LR_PATIENCE, --lr_patience LR_PATIENCE
                        Learning rate patience (default: 10)
  -s EARLY_STOPPER_PATIENCE, --early_stopper_patience EARLY_STOPPER_PATIENCE
                        Early stopper patience (default: 15)
  -c DELTA, --delta DELTA
                        Early stopper minimum delta (default: 15)
  -t PLOT, --plot PLOT  Plot learning trends (default: True)
  -i SAVE_TEST_OUTPUT, --save_test_output SAVE_TEST_OUTPUT
                        Save predicted masks of test dataset (default: True)

```
The classes argument is a subset of the classes specified in the Dataset class. 
The default values are derived from the CamVid dataset. to train on different datasets make sure to change the list of classes in Dataset.py as well as the subset of classes for classification passed on as arguments.
## Project Structure

- Train.py: train and evaluate model
- Unet.py: pytorch model implementation 
- requirements.txt: A list of python packages needed for the project.
- Dataset.py: data retrival and augmentation class used for training. contains the list of classes available for classification
- Losses.py: loss function available for training
- Utils.py: additional functions required for training
