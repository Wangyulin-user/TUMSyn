# Towards General Text-guided Universal Image Synthesis for Customized Multimodal Brain MRI Generation

Welcome! This is the official implementation of TUMSyn, which is a Text-guided Universal MR image Synthesis framework. It can flexibly generate brain MR images with demanded image contrast and spatial resolution from routinely-acquired scans guided by imaging metadata as text prompt. 
The model is trained and evaluated on a brain MR database comprising 31,407 3D images with 7 structural MRI modalities from 13 centers.

## What does TUMSyn achieves?
 1. **Multi-modal data processing:** By incorporating metadata of MR data (demographic and imaging parameter information) as text prompts to guide the image synthesis, TUMSyn enables customized cross-modal synthesis and flexible data domain transfer.
 2. **Accuracy and versatility:** TUMSyn achieves clinically acceptable precision in image generation and fulfills diverse real-world application scenarios by performing a wide array of cross-sequence synthesis tasks using a unified model.
 3. **Zero-shot Generalizability:** TUMSyn's zero-shot synthesis performance matches or even surpasses the performance of models trained on each individual external dataset.
 4. **Clinical impact:** TUMSyn can produce clinically meaningful sequences to assist in the diagnosis of Alzheimer's disease and cerebral small vessel diseases.

## Usage
### Preparing the dataset
For the MR image synthesis model, the relative storage path of each training/testing image path should be put in .txt files. The example format is shown [here](https://github.com/Wangyulin-user/TUMSyn/blob/main/train_img_M1.txt).
The format of training/testing images are .npy and .nii.gz, respectively. 
For compiling text prompt, we provide the English template. An examples of text prompt is shown below:
> "train_HCPD_T1w.npy": "Age: 12; Gender: M; Scanner: 3.0 Siemens; Modality: T1w; Voxel size: (0.8, 0.8, 0.8); Imaging parameter TR(ms), TE(ms), TI(ms), and FA(degree):  (2500.0, 2.2, 1000.0, 8.0)"

The content within the first double quotation marks is the image name, while the content within the second double quotation marks contains the corresponding demographic information and imaging parameters for that image. **When compiling your text prompts, you just need to place the metadata after each corresponding colon in our format**.

The full names of abbreviations as below:
 - M: Male (If it's a female, use the F)
 - TR: Repetition time
 - TE: Echo time
 - TI: Inversion time
 - FA: Flip angle

If the metadata does not contain the required information, use “None” as the placeholder.
Examples of training and testing image for the image synthesis model are also also provided in [Experimental_data](https://github.com/Wangyulin-user/TUMSyn/tree/main/Experimental_data/image), and their corresponding text prompts are provided [here](https://github.com/Wangyulin-user/TUMSyn/tree/main/Experimental_data).

### Training the image synthesis model
you need to set the [config](https://github.com/Wangyulin-user/TUMSyn/blob/main/configs/train_lccd_sr.yaml) file. We recommend that you only modify the file roots of training/validation datasets and Batch size. 

Once the config file setting is done, you can train the model by simply running the following command:

```
python train.py
```
When the training process finished, the trained network parameters will be saved in the [save]() directory.

### Start using TUMSyn to synthesize desired MR images

