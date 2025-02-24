import math
import os
import shutil


# 'C:\Users\tymop\OneDrive\Робочий стіл\Курсова\dataset\train'
def numbers_of_files(directory):
    subdirs = os.listdir(directory)

    for disease in subdirs:
        n = 0
        files = os.listdir(os.path.join(directory, disease))
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                n += 1
        print(f"{disease}: ", n)


def replace_underlines_in_files(parent_dir):
    child_dirs = os.listdir(parent_dir)

    for child_dir in child_dirs:
        full_directory = os.path.join(parent_dir, child_dir)

        if len(os.listdir(full_directory)) != 0:
            dir_files = os.listdir(full_directory)

            for file in dir_files:
                filepath = os.path.join(full_directory, file)
                new_name = filepath.replace("_", " ")
                os.rename(filepath, new_name)

                print("File renamed to: ", new_name)


def transfer_images_from_train_valid(train_dir, valid_dir, is_to_valid):
    diseases = os.listdir(train_dir)  # can be valid - directories are same in both
    images_to_transfer = []

    for disease_dir in diseases:

        full_disease_valid_dir = os.path.join(valid_dir, disease_dir)
        full_disease_train_dir = os.path.join(train_dir, disease_dir)
        i = 0

        if is_to_valid:
            while i < len(os.listdir(full_disease_valid_dir)):
                image = os.listdir(full_disease_train_dir)[i]
                images_to_transfer.append(os.path.join(full_disease_train_dir, image))
                print(f"{i} ", images_to_transfer[-1])

                shutil.move(images_to_transfer[-1], full_disease_valid_dir)
                i += 1
        else:
            half_of_images = math.floor(len(os.listdir(full_disease_valid_dir)) / 2)
            while i < half_of_images:
                image = os.listdir(full_disease_valid_dir)[i]
                images_to_transfer.append(os.path.join(full_disease_valid_dir, image))
                print(f"{i} ", images_to_transfer[-1])

                shutil.move(images_to_transfer[-1], full_disease_train_dir)
                i += 1
