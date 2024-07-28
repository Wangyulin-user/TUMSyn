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
For the MR image synthesis model, the relative storage path of each training/testing image path should be put in .txt files. The example format is shown [here](https://www.csdn.net/).
For the training/testing images, the format are .npy and .nii.gz, respectively. 
An examples of text prompt is shown below:
> "train_HCPD_T1w.npy": "Age: 12; Gender: M; Scanner: 3.0 Siemens; Modality: T1w; Voxel size: (0.8, 0.8, 0.8); Imaging parameter TR(ms), TE(ms), TI(ms), and FA(degree):  (2500.0, 2.2, 1000.0, 8.0)"


