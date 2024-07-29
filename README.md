# Towards General Text-guided Universal Image Synthesis for Customized Multimodal Brain MRI Generation

Welcome! This is the official implementation of TUMSyn, which is a Text-guided Universal MR image Synthesis framework. It can flexibly generate brain MR images with demanded image contrast and spatial resolution from routinely-acquired scans guided by imaging metadata as text prompt. 
The model is trained and evaluated on a brain MR database comprising 31,407 3D images with 7 structural MRI modalities from 13 centers.

## What does TUMSyn achieve?
 1. **Multi-modal data processing:** By incorporating metadata of MR data (demographic and imaging parameter information) as text prompts to guide the image synthesis, TUMSyn enables customized cross-sequence synthesis and flexible data domain transfer.
 2. **Accuracy and versatility:** TUMSyn achieves clinically acceptable precision in image generation and fulfills diverse real-world application scenarios by performing a wide array of cross-sequence synthesis tasks using a unified model.
 3. **Zero-shot Generalizability:** TUMSyn's zero-shot synthesis performance matches or even surpasses the performance of models trained on each individual external dataset.
 4. **Clinical impact:** TUMSyn can produce clinically meaningful sequences to assist in the diagnosis of Alzheimer's disease and cerebral small vessel diseases.

## Usage
### Preparing the dataset
For the MR image synthesis model, the relative storage path of each training/testing image path should be put in .txt files. The example format is shown [here](https://github.com/Wangyulin-user/TUMSyn/blob/main/train_img_M1.txt).
The format of training/testing images are .npy and .nii.gz, respectively. 
We provide the English template for compiling text prompts. An example of the text prompt is shown below:
> "train_HCPD_T1w.npy": "Age: 12; Gender: M; Scanner: 3.0 Siemens; Modality: T1w; Voxel size: (0.8, 0.8, 0.8); Imaging parameter TR(ms), TE(ms), TI(ms), and FA(degree):  (2500.0, 2.2, 1000.0, 8.0)"

The content within the first double quotation mark is the image name, while the content within the second double quotation mark contains the corresponding demographic information and imaging parameters for that image. **When compiling your text prompts, you just need to place the metadata after each corresponding colon in our format**.

The full names of abbreviations as below:
 - M: Male (If that is a female, use the F)
 - TR: Repetition time
 - TE: Echo time
 - TI: Inversion time
 - FA: Flip angle

If the metadata does not contain the required information, use “None” as the placeholder.

Examples of training and testing images for the image synthesis model are also provided in [Experimental_data](https://github.com/Wangyulin-user/TUMSyn/tree/main/Experimental_data/image) directory, and their corresponding text prompts are provided [here](https://github.com/Wangyulin-user/TUMSyn/tree/main/Experimental_data).

### Training the image synthesis model
![image](https://github.com/Wangyulin-user/TUMSyn/blob/main/img/github_framework.jpg)
The overview of TUMSyn training

To effectively align and fuse image-text pairs, TUMSyn is built upon a two-stage training strategy. In the first stage (stage 1), we pre-trained a text encoder using contrastive learning to extract textual semantic features that are aligned with the corresponding image features from metadata. Built on the pre-trained text encoder, in the second stage (stage 2), the text encoder is frozen and used to extract prompt features to steer the cross-sequence synthesis.

The following steps can help you to train your own network for text-guided MR image synthesis.

#### Step 1. Set the hyper-parameters of image synthesis model in the [config](https://github.com/Wangyulin-user/TUMSyn/blob/main/configs/train_lccd_sr.yaml) file. you can only modify the file roots of training/validation datasets and the batch sizes of model training . 
#### Step 2. Download the pre-trained weight ([checkpoint_CLIP.pt](https://zenodo.org/records/13119176)) of the text encoder in stage 1, and save it in the [main directory](https://github.com/Wangyulin-user/TUMSyn). 
#### Step 3. Train the model by simply running the following command:
     
     python train.py
     
The trained network parameters will be saved in the [save](https://github.com/Wangyulin-user/TUMSyn/tree/main/save) folder when the training process finishes.

### Start using TUMSyn to synthesize desired MR images
For an easy evaluation of TUMSyn, we also provide [demo](https://github.com/Wangyulin-user/TUMSyn/blob/main/demo.py) code to perform the model inference, along with our trained weight ([checkpoint.pth](https://zenodo.org/records/13119176)) of the image synthesis model. The trained weight should be downloaded and put in the [save](https://github.com/Wangyulin-user/TUMSyn/tree/main/save) directory. **Note: The model should be run on a Linux machine with GPU.**

## Content

 - [TUMSyn](https://github.com/Wangyulin-user/TUMSyn/tree/main): This is the main folder containing all model architecture and training/inference code.
   - [CLIP](https://github.com/Wangyulin-user/TUMSyn/tree/main/CLIP): This folder contains the model architectures and hyper-parameter configs of image encoder and text encoder in stage 1.
   - [utils_clip](https://github.com/Wangyulin-user/TUMSyn/tree/main/utils_clip): This folder contains the utility functions required to run the models in stage 1.
   - [Experimental_data](https://github.com/Wangyulin-user/TUMSyn/tree/main/Experimental_data): This folder provides example images and text prompts for model training and inference.
   - [datasets](https://github.com/Wangyulin-user/TUMSyn/tree/main/datasets): This folder contains codes for the image synthesis model to process text and image data.
   - [configs](https://github.com/Wangyulin-user/TUMSyn/tree/main/configs): This folder contains `train_lccd_sr.yaml`, which is used to set all training/validation parameters.
   - [models](https://github.com/Wangyulin-user/TUMSyn/tree/main/models): This folder contains `lccd.py`, which is a wrapper around the model. It also contains the network architectures of all the modules used in the image synthesis model.
   - [save](https://github.com/Wangyulin-user/TUMSyn/tree/main/save): This folder saves all the trained checkpoints of the image synthesis model.
   - [demo.py](https://github.com/Wangyulin-user/TUMSyn/blob/main/demo.py): contains the function to inference the network.
   - [train.py](https://github.com/Wangyulin-user/TUMSyn/blob/main/train.py): contains the function to train the network.
   - [train_img_M1.txt](https://github.com/Wangyulin-user/TUMSyn/blob/main/train_img_M1.txt): is the image path file of modality 1. 
   - [train_img_M2.txt](https://github.com/Wangyulin-user/TUMSyn/blob/main/train_img_M2.txt): is the image path file of modality 2. 
   - [utils.py](https://github.com/Wangyulin-user/TUMSyn/blob/main/utils.py): includes the utility functions required to run the image synthesis model.
