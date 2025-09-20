#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import shutil
import matplotlib.pyplot as plt
import os
import torch
import pickle
from datasets import load_dataset, Dataset

from PIL import Image
import random
import math
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from env import DATA_PATH
from tqdm import tqdm

# set the directory
PENDULUM_DIR = f'{DATA_PATH}/pendulum'

# pendulum's attributes
IMG_COLUMN = 'img_dir'

# CONCEPT_NAMES = ['theta', 'phi','suspension_x','suspension_y','pendulum_length',
#                  'pendulum_x', 'pendulum_y','ball_x','ball_y','light_x', 'light_y',
#                  'shade_plane','shade_length']

CONCEPT_NAMES = ['theta', 'phi']

TASK_NAMES = ['pendulum_x']

def projection(phi, x_0, y_0, base = -0.5): # calculate x intersection between y - y_0 = tan(phi) (x- x_0) and y = base
    b = y_0-x_0*math.tan(phi)
    shade = (base - b)/math.tan(phi)
    return shade

def create_dataframe(CONCEPT_NAMES):
    #scale = np.array([[0,44],[100,40],[7,7.5],[10,10]]) # [min_value, max_value] for each concept
    count = 0
    train_df = pd.DataFrame(columns=[IMG_COLUMN]+CONCEPT_NAMES+TASK_NAMES)
    val_df = pd.DataFrame(columns=[IMG_COLUMN]+CONCEPT_NAMES+TASK_NAMES)
    test_df = pd.DataFrame(columns=[IMG_COLUMN]+CONCEPT_NAMES+TASK_NAMES)
    for theta in tqdm(np.linspace(-200,200,100)):# angle formed by the pendulum
        for phi in np.linspace(60,140,1000):# angle formed by the light
            if phi == 100:
                continue
            plt.rcParams['figure.figsize'] = (1.0, 1.0)
            theta_rad = theta*math.pi/200.0 # convert the original values into radians (I think they use centesimal degrees)
            phi_rad = phi*math.pi/200.0

            # Calculate the coordinates of the center of the pendulum’s ball
            # (10, 10.5) are the coordinates of the pendulum’s point of suspension, and 8 is the length of the pendulum
            x = 10 + 8*math.sin(theta_rad) 
            y = 10.5 - 8*math.cos(theta_rad)

            # Draw the pendulum
            ball = plt.Circle((x,y), 1.5, color = 'firebrick')
            gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)

            light = projection(phi_rad, 10, 10.5, 20.5) # x-coordinate of the center of the lights' ball, y = 20.5 is its fixed height
            sun = plt.Circle((light,20.5), 3, color = 'orange')


            # Calculate an approximation of the border of the pendulum's ball, starting point for the shade
            ball_x = 10+9.5*math.sin(theta_rad)
            ball_y = 10.5-9.5*math.cos(theta_rad)

            # Calculate mid point of the shade and the length of the shade (min 3)
            mid = (projection(phi_rad, 10.0, 10.5)+projection(phi_rad, ball_x, ball_y))/2
            shade = max(3,abs(projection(phi_rad, 10.0, 10.5)-projection(phi_rad, ball_x, ball_y)))

            shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)
            
            ax = plt.gca()
            ax.add_artist(gun)
            ax.add_artist(ball)
            ax.add_artist(sun)
            ax.add_artist(shadow)
            ax.set_xlim((0, 20))
            ax.set_ylim((-1, 21))

            # normalize values
            values_dict = {
            'img_dir': "",
            'theta': theta_rad, #(theta - scale[0][0]) / (scale[0][1] - 0),
                'phi': phi_rad, #(phi - scale[1][0]) / (scale[1][1] - 0)
                'suspension_x': 10,
                'suspension_y': 10.5,
                'pendulum_length': 8,
                'pendulum_x': x,
                'pendulum_y': y,
                'ball_x': ball_x,
                'ball_y': ball_y,
                'light_x': light,
                'light_y': 20.5,
                'shade_plane': -0.5,
                'shade_length': shade, #(shade - scale[2][0]) / (scale[2][1] - 0),
                'shade_mid': mid #(mid - scale[2][0]) / (scale[2][1] - 0)
            }

            new = pd.DataFrame([values_dict], columns=values_dict.keys(), index=[1])
            

            plt.axis('off')
            if count == 0 or count ==1:
                # save in a pickle both the images and the different concepts
                new['img_dir'] = f'{PENDULUM_DIR}/test/a_' + str(round(float(theta),4)) + '_' + str(round(float(phi),4)) + '_' + str(round(float(shade),4)) + '_' + str(round(float(mid),4)) +'.png'
                test_df=pd.concat([test_df, new], ignore_index=True)
                plt.savefig(new['img_dir'].iloc[0], dpi=96, transparent=False)
            elif count == 2:
                new['img_dir'] = f'{PENDULUM_DIR}/val/a_' + str(round(float(theta),4)) + '_' + str(round(float(phi),4)) + '_' + str(round(float(shade),4)) + '_' + str(round(float(mid),4)) +'.png'
                val_df=pd.concat([val_df, new], ignore_index=True)
                plt.savefig(new['img_dir'].iloc[0],dpi=96, transparent=False)
            else:
                new['img_dir'] = f'{PENDULUM_DIR}/train/a_' + str(round(float(theta),4)) + '_' + str(round(float(phi),4)) + '_' + str(round(float(shade),4)) + '_' + str(round(float(mid),4)) +'.png'
                train_df=pd.concat([train_df, new], ignore_index=True)
                plt.savefig(new['img_dir'].iloc[0],dpi=96, transparent=False)
                if count == 9:
                    count = -1
            plt.clf()
            count += 1

            # save dataframes
    test_df.to_pickle(f'{PENDULUM_DIR}/test/test_df.pkl')
    val_df.to_pickle(f'{PENDULUM_DIR}/val/val_df.pkl')
    train_df.to_pickle(f'{PENDULUM_DIR}/train/train_df.pkl')
    return None


class PendulumDataset:
    def __init__(self,
                    already_created: bool,
                    batch_size: int = 32
                 ):

        if not already_created:
            print('Generating dataset...')
            if os.path.exists(PENDULUM_DIR):
                shutil.rmtree(PENDULUM_DIR)

            os.makedirs(f'{PENDULUM_DIR}/train/')
            os.makedirs(f'{PENDULUM_DIR}/val/')
            os.makedirs(f'{PENDULUM_DIR}/test/')
            create_dataframe(CONCEPT_NAMES)
        else:
            # check if there is some data in the directory
            if not os.path.exists(PENDULUM_DIR) or not os.path.exists(f'{PENDULUM_DIR}/train/train_df.pkl')\
               or not os.path.exists(f'{PENDULUM_DIR}/val/val_df.pkl')\
               or not os.path.exists(f'{PENDULUM_DIR}/test/test_df.pkl'):
                raise ValueError(f"Dataset not found in {PENDULUM_DIR}. Please set already_created to False to create the dataset.")
            else:   
                print('Using already created dataset at', PENDULUM_DIR)

        self.name = 'Pendulum'
        self.root = PENDULUM_DIR
        self.batch_size = batch_size

        self.train_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(self.root, 'train/train_df.pkl')))
        self.train_dataset.set_format(type='python')
        self.val_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(self.root, 'val/val_df.pkl')))
        self.val_dataset.set_format(type='python')
        self.test_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(self.root, 'test/test_df.pkl')))
        self.test_dataset.set_format(type='python')

    def collator(self,
                 num_workers: int = 0,
                 persistent_workers: bool = False,
                 pin_memory: bool = True,):
        
        data_collator = CustomDataCollator()
        loaded_train = DataLoader(
            self.train_dataset, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=True
            )

        loaded_val = DataLoader(
            self.val_dataset, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        
        loaded_test = DataLoader(
            self.test_dataset, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        
        return loaded_train, loaded_val, loaded_test


class CustomDataCollator:
    def __init__(self):
        self.concept_names = [concept for concept in CONCEPT_NAMES]
        self.task_names = TASK_NAMES

    def __call__(self, batch):

        # transform the batch into a tensor
        labels = torch.Tensor([[example[concept] for concept in self.task_names] for example in batch])
        if len(self.task_names) == 1:
            labels = labels.squeeze(1)
        concepts = torch.tensor(
            [[example[concept] for concept in self.concept_names] for example in batch]
        )
        #food = torch.Tensor([example['food'] for example in batch])
        #ambiance = torch.Tensor([example['ambiance'] for example in batch])
        #service = torch.Tensor([example['service'] for example in batch])
        #noise = torch.Tensor([example['noise'] for example in batch])


        #extract the images in img_dir
        images = [Image.open(example[IMG_COLUMN]).convert('RGB') for example in batch]
        # 2. Convert to NumPy array (uint8 → float32 in [0,1])
        images = np.array([np.array(img, dtype=np.float32)/255.0 for img in images])
        # permute the dimensions to have channels first, from (batch_size, height, width, channels) to (batch_size, channels, height, width)
        images = np.transpose(images, (0, 3, 1, 2))
        images = torch.tensor(images)

        #test: print one of the images and put it in dir= '/home/admin/example_image.png'
        #plt.imsave('/home/admin/example_image.png', images[0].numpy())


        return {
            'x': images,
            'c': concepts,
            'y': labels
        }
    

# Test
def main(): 
    loader = PendulumDataset(already_created=True, batch_size=32)
    train_loader, _, _ = loader.collator()

    for batch in train_loader:
        print(batch['x'])
        print(batch['c'])
        print(batch['y'])
        break

if __name__=="__main__":
    main()