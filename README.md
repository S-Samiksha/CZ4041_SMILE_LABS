# CE4041_Kaggle
ML project on Kaggle

## Documents, train, test, CSVs etc. 
For training images: https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/data?select=test-public-faces.zip <br><br>
For training relationships CSV: test-public-relationships.csv (from github) <br><br>
For test images: https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/data?select=test.zip <br><br>
For test sample submission csv: https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/data?select=sample_submission.csv <br><br>
For monitoring all other submissions: https://docs.google.com/spreadsheets/u/0/d/1gLLzor08xsw7wZcxoJIVv4xDkaSnzP8LzTi0noMx8vA/edit?pli=1#gid=0<br><br>

## What has been done so far 

1. Facenet Version 4 (0.807 @ 30 Epochs)
   - Add a lot more data to become 59K
   - The data added is more within the same family. Meaning, we are family but not kin (this data predominates the whole thing)
   - The fully connected layers after the facenet convolutional layer has dropout at 0.7 to prevent overfitting and batchnorm as well
   - Every other epoch, we are changing data augmentation to introduce more variation to prevent overfitting
   - grayscale is important because some of the test data is greyscale. blur is important because some of the data is highly blurred. random horizontal flip to increase variation
   - we can try randomcrop because some of the test data is cropped, color jitter as well
   - some images in the test cases have two faces —> so i feel random crop might help. OR stacking (there is a pytorch function iirc)
   - learning rate is set at 0.005 batch size is 64

2. Facenet Version 5 (0.867 @ 30 Epochs)
   - Updated version from version 4
   - Randomized data set 

3. Facenet Version 7 (0.907 @ 30 Epochs)
   - https://colab.research.google.com/drive/1FM-ls2q-9VKl3Ny2cOQjxbP5MXSQN2Wk?usp=sharing
   - Changing drop out from 0.7 to 0.55
   - Adding data augmentation of cropping and color jittering
   - Randomizing the non-relationship
   - Using all relationships available but randomizing the pictures selected
   - Submitted CSV: test_Results_SAM_NO_MTCNN_30Epochs_BatchSize_64_new_csv_probability_new_data_new_data_augment_05Drouput.csv
