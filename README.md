# CE4041_Kaggle
ML project on Kaggle

## Members 
Sankar Samiksha <br>
Jia Min <br>
Xing Kun <br>
Yu Pei <br>
Tabu <br>

## Documents, train, test, CSVs etc. 
[For training images](https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/data?select=test-public-faces.zip) <br><br>
[For training relationships CSV:](https://github.com/S-Samiksha/CE4041_Kaggle/blob/main/test-public-relationships.csv) <br><br>
[For test images:](https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/data?select=test.zip) <br><br>
[For test sample submission csv:](https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/data?select=sample_submission.csv) <br><br>
[For monitoring all other submissions:](https://docs.google.com/spreadsheets/u/0/d/1gLLzor08xsw7wZcxoJIVv4xDkaSnzP8LzTi0noMx8vA/edit?pli=1#gid=0) <br><br>

## Proposed Solution and Explanation of code 

*Fully Connected Layers* <br>
Changing the fully connected layers allows us to have more flexibility and control over the model whilst still using a pre trained model. The fully connected layers that are important are the DropOut Layer and the BatchNormId. 
<br>
```python
model.classifier = nn.Sequential(
            nn.Linear(3584,2048),
            nn.ReLU(),
            nn.Dropout(0.55),  # Add dropout for regularization
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256), # Apply batch normalization
            nn.Linear(256, 2)
        )
```
*Pre-trained Model, Learning Rate, Adam Optimizer and Loss Criterion* <br>
A pre-trained model was used. [Facenet](https://github.com/timesler/facenet-pytorch/tree/master) has been trained on the vggface2 image dataset. 
<br>
```python

# Create the Siamese network
net = SiameseNetwork(InceptionResnetV1(pretrained='vggface2', classify=False)).cuda()
# Define the contrastive loss
criterion = nn.CrossEntropyLoss()

# Define the optimizer (e.g., Adam)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

```
*Constantly changing data augmentation* <br>
A constantly changing data augmentation was implemented by the team. It is `unique` and has not been used by those in the competition. 
<br>
```python
    if epoch % 10 == 0 or epoch %10 == 1 or epoch %10 == 2 or epoch %10 == 3:
      print("Data Augmentation: None")
      trainloader = createTrain([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor()])
    elif epoch %10 == 4 or epoch %10 == 5:
       print("Data Augmentation: RandomGrayScale(0.5)")
       trainloader = createTrain([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomGrayscale(p=0.5),transforms.ToTensor()])
    elif epoch %10 == 6 or epoch %10 == 7:
      print("Data Augmentation: RandomCrop((90,90)),RandomGrayScale(0.8), RandomHorizontalFlip, GaussianBlur(kernel_size = 5, sigma=(0.1, 3.0)")
      trainloader = createTrain([transforms.RandomCrop((80,80)),transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomGrayscale(p=0.8),transforms.RandomHorizontalFlip(),transforms.GaussianBlur(kernel_size = 5, sigma=(0.1, 3.0)),transforms.ToTensor()])
    elif epoch %10 == 8 or epoch %10 == 9:
      print("Data Augmentation: RandomGrayScale(0.8), RandomHorizontalFlip, ColorJitter(brightness=0.7, contrast=0.3),")
      trainloader = createTrain([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomGrayscale(p=0.5),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=0.7, contrast=0.3),transforms.ToTensor()])
    else:
      print("Data Augmentation: None")
      trainloader = createTrain([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor()])
```


## Significant Milestones Achieved through Experiments 

1. Facenet Version 4 (0.807 @ 30 Epochs)
   - Add a lot more data to become 59K
   - The data added is more within the same family. Meaning, the images are family but not kin (this data predominates the whole thing)
   - The fully connected layers after the FaceNet convolutional layer has a dropout layer at 0.7 to prevent overfitting and batchnorm as well
   - Every epoch, data augmentation is changed to introduce more variation to prevent overfitting
   - Grayscale is important because some of the test data is grayscale. blur is important because some of the data is highly blurred. random horizontal flip to increase variation
   - learning rate is set at 0.005 batch size is 64

2. Facenet Version 5 (0.867 @ 30 Epochs)
   - Updated version from version 4
   - Randomized data set 

3. Facenet Version 7 (0.907 @ 30 Epochs)
   - https://colab.research.google.com/drive/1FM-ls2q-9VKl3Ny2cOQjxbP5MXSQN2Wk?usp=sharing
   - Changing dropout layer from 0.7 to 0.55
   - Adding data augmentation of cropping as there are images with two faces in them, some faces are obscured by sunglasses, or other accessories.
   - Adding data augmentation of color jittering as the images are of different brightness and contrast
   - Randomizing the non-relationship dataset
   - All relationships in the [CSV](https://github.com/S-Samiksha/CE4041_Kaggle/blob/main/test-public-relationships.csv) however the pictures selected was randomized
   - [Submitted CSV](https://github.com/S-Samiksha/CE4041_Kaggle/blob/main/test_Results_SAM_NO_MTCNN_30Epochs_BatchSize_64_new_csv_probability_new_data_new_data_augment_05Drouput.csv)
