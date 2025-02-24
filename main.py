import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets

from constants import (PATH_TRAIN, PATH_TEST, PATH_VALID, AUGMENT_TRANSFORM as aug,
                       NO_AUGMENT_TRANSFORM as no_aug)
from extra import numbers_of_files
from training import deploy_cnn, deploy_custom_cnn, train, test, deploy_fine_tuned_cnn

if __name__ == '__main__':
    # PREPROCESSING
    print("TRAIN")
    numbers_of_files(PATH_TRAIN)
    print("VALID")
    numbers_of_files(PATH_VALID)
    print("TEST")
    numbers_of_files(PATH_TEST)

    # Initializing a training, validation and testing set
    train_set = datasets.ImageFolder(PATH_TRAIN, aug)
    valid_set = datasets.ImageFolder(PATH_VALID, no_aug)
    test_set = datasets.ImageFolder(PATH_TEST, no_aug)

    print("\nTraining part: ", len(train_set.samples))
    print("Validating part: ", len(valid_set.samples))
    print("Testing part: ", len(test_set.samples))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=6)

    print(test_set.classes)

    # EFFICIENT NET DEPLOYING
    eff_cnn = deploy_cnn(train_set, models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
    differ_rate_cnn = deploy_cnn(train_set, models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
    # fine_tuned_cnn = deploy_fine_tuned_cnn(train_set, models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
    # print(conv_net)
    # summary(conv_net, (3, 300, 300))

    # SCRATCH CNN DEPLOYING
    # scratch_cnn = deploy_custom_cnn()

    # TRAINING AND VALIDATING EFFICIENT NET
    # eff_net = train(eff_cnn, train_loader, valid_loader, False)
    differ_rate_net = train(differ_rate_cnn, train_loader, valid_loader, True)
    # TRAINING AND VALIDATING SCRATCH NETWORK
    # scratch_net = train(scratch_cnn, train_loader, valid_loader)

    # TESTING EFFICIENT NET
    # test(eff_net, test_loader, test_set.classes)
    test(differ_rate_net, test_loader, test_set.classes)
    # TESTING SCRATCH NETWORK
    # test(scratch_net, test_loader, test_set.classes)


