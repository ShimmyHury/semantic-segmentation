from Dataset import *
from Utils import *
from Losses import FocalLoss, TverskyLoss
import numpy as np
from torch.nn import CrossEntropyLoss
from Unet import Unet
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def run_epoch(data_loader, model, optimizer, loss_func, n_classes, description, training=True, save_img_output=False):
    batch_loss = []
    batch_acc = []
    activation = nn.Softmax2d()
    for step, (batch_x, batch_y) in enumerate(
            tqdm(data_loader, ncols=100, desc=description, leave=True)):  # for each training step
        targets = one_hot_to_class(batch_y).type(torch.LongTensor).to(device)
        if training:
            optimizer.zero_grad()  # clear gradients for next train
        prediction = model(batch_x.float().to(device))  # input x and predict based on x
        loss = loss_func(prediction, targets)  # must be (1. nn output, 2. target)
        if training:
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        batch_loss.append(float(loss))
        y_true = targets.cpu().numpy()
        y_pred = torch.argmax(activation(prediction), dim=1).cpu().detach().numpy()

        if save_img_output:
            save_images(target=y_true.squeeze(), prediction=y_pred.squeeze(), n_classes=n_classes,
                        folder_name=description, img_name=step)

        iou, wiou = miou(targets=y_true, predictions=y_pred, num_classes=n_classes)
        batch_acc.append(wiou)
    return sum(batch_loss) / len(batch_loss), sum(batch_acc) / len(batch_acc)


def save_images(target, prediction, n_classes, folder_name, img_name):
    # save test output for visual comparison with true test masks
    im_v = cv2.vconcat([target, prediction])
    filename = str(img_name) + '.png'
    output_dir = folder_name + '_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, filename), im_v * (255 / (n_classes - 1)))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_dir", default='./data/CamVid/', help="Path to training data directory")
    parser.add_argument("-m", "--model_path", default='./trained_model', help="Path to model (save/load)")
    parser.add_argument("-l", "--loss", default="Tversky", help="Loss function, one of [Focal,Tversky,CrossEntropy]")
    parser.add_argument("-n", "--classes", default=['sky', 'building', 'pole', 'road', 'pavement',
                                                    'tree', 'signsymbol', 'fence', 'car',
                                                    'pedestrian', 'bicyclist', 'unlabelled'],
                        type=dict, help="classes for detection. "
                                        "subset of the classes specified in the "
                                        "Dataset class")

    parser.add_argument("-e", "--epochs", default=1, type=int, help="Number of epochs for training")
    parser.add_argument("-b", "--batch", default=4, type=int, help="Batch size")

    parser.add_argument("-r", "--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("-f", "--lr_decrease_factor", default=0.1, type=float, help="Learning-rate-decrease-factor")
    parser.add_argument("-p", "--lr_patience", default=10, type=int, help="Learning rate patience")
    parser.add_argument("-s", "--early_stopper_patience", default=15, type=int, help="Early stopper patience")
    parser.add_argument("-c", "--delta", default=15, type=float, help="Early stopper minimum delta")

    parser.add_argument("-t", "--plot", default=True, type=bool, help="Plot learning trends")
    parser.add_argument("-i", "--save_test_output", default=True, type=bool,
                        help="Save predicted masks of test dataset")

    args = vars(parser.parse_args())

    print(args)

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # Paths to data and model
    PATH = args['model_path']  # './trained_model'
    DATA_DIR = args['data_dir']  # './data/CamVid/'
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    # mean, std = calc_mean_std(x_train_dir) #calc mean & std manually
    mean = [0.485, 0.456, 0.406]  # imagenet mean values
    std = [0.229, 0.224, 0.225]  # imagenet std values
    weights = calc_class_weights(y_train_dir)
    num_classes = len(args['classes'])
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(mean, std),
        classes=args['classes'],
    )
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(mean, std),
        classes=args['classes'],
    )

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(mean, std),
        classes=args['classes'],
    )

    train_loader = DataLoader(train_dataset, batch_size=args['batch'], shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    unet = Unet(numberClasses=num_classes).float().to(device)
    optim = torch.optim.Adam(unet.parameters(), lr=args['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                           factor=args['lr_decrease_factor'],
                                                           patience=args['lr_patience'],
                                                           threshold_mode='abs', verbose=True)
    if args['loss'] == 'Tversky':
        loss_function = TverskyLoss(num_classes=num_classes).to(device)
    elif args['loss'] == 'Focal':
        loss_function = FocalLoss(weight=torch.tensor(weights)).to(device)
    elif args['loss'] == 'CrossEntropy':
        loss_function = CrossEntropyLoss(weight=torch.tensor(weights)).to(device)
    else:
        loss_function = TverskyLoss(num_classes=num_classes).to(device)

    best_loss = np.Inf
    try:
        unet.load_state_dict(torch.load(PATH))
        print(f"{PATH} valid. state dict loaded")
    except Exception as e:
        print(e)
        print(f"{PATH} not valid. state dict not loaded. training from scratch.")

    early_stopper = EarlyStopper(patience=args['early_stopper_patience'], min_delta=args['delta'])
    epoch_train_loss = []
    epoch_train_iou = []
    epoch_val_loss = []
    epoch_val_iou = []

    # start training
    for epoch in range(args['epochs']):
        unet.train()
        desc = "Epoch " + str(epoch + 1) + "/" + str(args['epochs'])
        epoch_loss, epoch_acc = run_epoch(data_loader=train_loader, model=unet, optimizer=optim,
                                          loss_func=loss_function, n_classes=num_classes,
                                          description=desc, training=True)
        epoch_train_loss.append(epoch_loss)
        epoch_train_iou.append(epoch_acc)

        unet.eval()
        with torch.no_grad():
            val_loss, val_acc = run_epoch(data_loader=valid_loader, model=unet, optimizer=optim,
                                          loss_func=loss_function, n_classes=num_classes,
                                          description='Validation', training=False)
        epoch_val_loss.append(val_loss)
        epoch_val_iou.append(val_acc)
        scheduler.step(val_loss)

        print("Epoch: ", epoch + 1, " | Train loss: ", "{:.2f}".format(epoch_train_loss[-1]), " | Train IoU: ",
              "{:.2f}".format(epoch_train_iou[-1]),
              " | Validation loss: ", "{:.2f}".format(epoch_val_loss[-1]), " | Validation IoU: ",
              "{:.2f}".format(epoch_val_iou[-1]))

        if epoch_val_loss[-1] <= best_loss:
            torch.save(unet.state_dict(), PATH)
            best_loss = epoch_val_loss[-1]

        if early_stopper.early_stop(epoch_val_loss[-1]):
            break

    unet = Unet(numberClasses=num_classes)
    unet.to(device)
    unet.load_state_dict(torch.load(PATH))
    unet.eval()
    with torch.no_grad():
        test_loss, test_acc = run_epoch(data_loader=test_loader, model=unet, optimizer=optim, loss_func=loss_function,
                                        n_classes=num_classes,
                                        description='Test', training=False,
                                        save_img_output=args['save_test_output'])
    print("Test loss: ", np.mean(test_loss), " | Test IoU: ", np.mean(test_acc))

    if args['plot']:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(np.arange(1, len(epoch_train_iou) + 1), epoch_train_loss, label='Train')
        ax1.plot(np.arange(1, len(epoch_train_iou) + 1), epoch_val_loss, label='Validation')
        ax1.set_title('Loss')
        ax1.set(xlabel='Epoch', ylabel='Loss')
        ax1.legend()
        ax2.plot(np.arange(1, len(epoch_train_iou) + 1), epoch_train_iou, label='Train')
        ax2.plot(np.arange(1, len(epoch_train_iou) + 1), epoch_val_iou, label='Validation')
        ax2.set_title('IoU')
        ax2.set(xlabel='Epoch', ylabel='IoU')
        ax2.legend()
        plt.show()
