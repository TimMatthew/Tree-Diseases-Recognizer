from torch.utils.data import DataLoader
from torchsummary import summary
from CNN import deploy_cnn, train, test
from dataset_preprocessing import create_dataset
from draft import numbers_of_files

from constants import (PATH_TRAIN, PATH_TEST, PATH_VALID, AUGMENT_TRANSFORM as aug,
                       NO_AUGMENT_TRANSFORM as no_aug)

import os

if __name__ == '__main__':
    # os.rename(r"C:\Users\tymop\OneDrive\Робочий стіл\Курсова\EfficientNet results\B3\thunder-124463.wav",
    # r"C:\Users\tymop\OneDrive\Робочий стіл\Reaper projects\There is no tomorrow\thunder-124463.wav")

    # PREPROCESSING
    print("TRAIN")
    numbers_of_files(PATH_TRAIN)
    print("VALID")
    numbers_of_files(PATH_VALID)
    print("TEST")
    numbers_of_files(PATH_TEST)

    # Initializing a training, validation and testing set
    train_set = create_dataset(PATH_TRAIN, no_aug)
    valid_set = create_dataset(PATH_VALID, no_aug)
    test_set = create_dataset(PATH_TEST, no_aug)

    print("\nTraining part: ", len(train_set.samples))
    print("Validating part: ", len(valid_set.samples))
    print("Testing part: ", len(test_set.samples))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=6)

    print(test_set.classes)

    # CNN DEPLOYING
    conv_net = deploy_cnn(train_set)

    print(conv_net)
    summary(conv_net, (3, 300, 300))

    # TRAINING AND VALIDATING
    conv_net = train(conv_net, train_loader, valid_loader)

    # TESTING
    test(conv_net, test_loader, test_set.classes)
