import time

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import confusion_matrix
from torch import cuda

from constants import LR, DECAY, NUM_EPOCHS, last_trainable_layers

matplotlib.use('TkAgg')


# Fine tuning


def deploy_cnn(train_set):
    efficient_net = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)


    print("-------LAST ", last_trainable_layers, " LAYERS BEFORE FC-------\n",
          efficient_net.features[-last_trainable_layers:])
    print("-------AVG POOLING-------\n", efficient_net.avgpool)
    print("-------FULLY-CONNECTED-------\n", efficient_net.classifier)

    # # Трейн - 95% Валід - 93%
    # # Freeze all feature layers except for 2 last
    # for param in efficient_net.features[0:-last_trainable_layers].parameters():
    #     param.requires_grad = False
    #
    # # Keep 2 last layers trainable
    # for param in efficient_net.features[-last_trainable_layers:].parameters():
    #     param.requires_grad = True
    #
    # # Keep AdaptiveAvgPool2d trainable
    # for param in efficient_net.avgpool.parameters():
    #     param.requires_grad = True

    classes_num = len(train_set.classes)
    # print(classes_num)

    efficient_net.classifier[1] = nn.Linear(efficient_net.classifier[1].in_features, classes_num)

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

    return efficient_net.to(device)


def train(model, train_loader, valid_loader):
    device = 'cuda' if cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)

    FC_optimizer = optim.Adam(model.classifier.parameters(), lr=LR, weight_decay=DECAY)
    backbone_optimizer = optim.Adam(model.features.parameters(), lr=1e-5, weight_decay=1e-6)

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    start_time = time.time()

    results = []
    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        process = 0

        for inputs, labels in train_loader:
            process += 1
            print(f"{epoch + 1} epoch processing training... {process}")
            inputs, labels = inputs.to(device), labels.to(device)

            backbone_optimizer.zero_grad()
            FC_optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            backbone_optimizer.step()
            FC_optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)

        train_results = (f"Epoch #{epoch + 1}/{NUM_EPOCHS} ------- "
                         f"Training Accuracy: {epoch_acc:.4f}%, "
                         f"Loss: {epoch_loss:.4f}")

        # print(train_results)

        train_accs.append(epoch_acc)
        train_losses.append(epoch_loss)

        # VALIDATION

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()

            epoch_acc = 100 * valid_correct / valid_total
            epoch_loss = valid_loss / len(valid_loader)

            valid_results = (f"\n                     Validation Accuracy: {epoch_acc:.4f}%, "
                             f"Loss: {epoch_loss:.4f}")

            # print(valid_results)

            results.append(train_results + valid_results)

            valid_accs.append(epoch_acc)
            valid_losses.append(epoch_loss)

    end_time = time.time()
    summary = end_time - start_time

    print("Time for network processing:", summary)

    for elem in results:
        print(elem)

    show_stats(train_accs, train_losses, valid_accs, valid_losses)
    return model


# def test(model, test_set, tensor_num):
#     device = 'cuda' if cuda.is_available() else 'cpu'
#
#     test_model = model.to(device)
#
#     img_tensor, actual_label = test_set[tensor_num]
#     to_pil = transforms.ToPILImage()
#     image_to_test = to_pil(img_tensor)
#     img_tensor = img_tensor.unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         outputs = test_model(img_tensor)
#
#     _, predicted_class = torch.max(outputs.data, 1)
#     predicted_class_name = test_set.classes[predicted_class.item()]
#
#     print(f"Actual: {test_set.classes[actual_label]}")
#     print(f"Predicted: {predicted_class_name}\n")
#     image_to_test.show()

def test(model, test_loader, class_names):
    true_labels = []
    predicted_labels = []

    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0, va='center')

    plt.subplots_adjust(left=0.21, right=0.886, top=0.9, bottom=0.233)

    plt.xlabel("Predicted labels")
    plt.xlabel("True labels")
    plt.xlabel("Confusion matrix test")
    plt.show()


def show_stats(train_accs, train_losses, valid_accs, valid_losses):
    epochs = range(1, NUM_EPOCHS + 1)

    plt.plot(epochs, train_accs, 'y', label='Training accuracy')
    plt.plot(epochs, valid_accs, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

    plt.plot(epochs, train_losses, 'y', label='Training loss')
    plt.plot(epochs, valid_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
