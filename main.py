import torchinfo
import torchvision.models as models
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets

from constants import (PATH_TRAIN, PATH_TEST, PATH_VALID, AUGMENT_TRANSFORM as aug,
                       NO_AUGMENT_TRANSFORM as no_aug)
from extra import numbers_of_files
from training import (deploy_cnn, deploy_custom_cnn, train, test, deploy_fine_tuned_cnn_2_layers,
                      deploy_fine_tuned_cnn_4_layers,
                      deploy_feature_extractor)


def deploy_simple_effnetb3():
    eff_cnn = deploy_cnn(train_set, models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
    print(eff_cnn)
    summary(eff_cnn, (3, 300, 300), 32)
    eff_net = train(eff_cnn, train_loader, valid_loader, False)
    test(eff_net, test_loader, test_set.classes)


def deploy_extractor_cnn():
    eff_net = deploy_feature_extractor(train_set, models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
    print(eff_net)
    summary(eff_net, (3, 300, 300), 32)
    eff_net = train(eff_net, train_loader, valid_loader, False)
    test(eff_net, test_loader, test_set.classes)


def deploy_effnetb3_with_diff_rates():
    differ_rate_cnn = deploy_cnn(train_set, models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
    print(differ_rate_cnn)
    summary(differ_rate_cnn, (3, 300, 300))
    differ_rate_net = train(differ_rate_cnn, train_loader, valid_loader, True)
    test(differ_rate_net, test_loader, test_set.classes)


def deploy_effnetb3_fine_tuned_2_layers():
    fine_tuned_cnn = deploy_fine_tuned_cnn_2_layers(train_set, models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT)
    print(fine_tuned_cnn)
    torchinfo.summary(fine_tuned_cnn, (1, 3, 300, 300))
    fine_tuned_net = train(fine_tuned_cnn, train_loader, valid_loader, False)
    test(fine_tuned_net, test_loader, test_set.classes)


def deploy_effnetb3_fine_tuned_4_layers():
    fine_tuned_cnn = deploy_fine_tuned_cnn_4_layers(train_set, models.efficientnet_b3,
                                                    models.EfficientNet_B3_Weights.DEFAULT)
    print(fine_tuned_cnn)
    torchinfo.summary(fine_tuned_cnn, (1, 3, 300, 300))
    fine_tuned_net = train(fine_tuned_cnn, train_loader, valid_loader, False)
    test(fine_tuned_net, test_loader, test_set.classes)


def deploy_scratch_cnn():
    scratch_cnn = deploy_custom_cnn()
    print(scratch_cnn)
    summary(scratch_cnn, (3, 300, 300))
    scratch_net = train(scratch_cnn, train_loader, valid_loader, False)
    test(scratch_net, test_loader, test_set.classes)


if __name__ == '__main__':

    print("TRAIN")
    numbers_of_files(PATH_TRAIN)
    print("VALID")
    numbers_of_files(PATH_VALID)
    print("TEST")
    numbers_of_files(PATH_TEST)

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
