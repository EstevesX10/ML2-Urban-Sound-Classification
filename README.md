<div align="center">

# ML2 | Urban Sound Classification

</div>

<p align="center" width="100%">
    <img src="./Urban-Sound-Classification/Assets/NoisePolution.jpeg" width="55%" height="55%" />
</p>

<div align="center">
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Python-fec8bf?style=for-the-badge&logo=Python&logoColor=fec8bf">
    </a>
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Tensorflow-fec8bf?style=for-the-badge&logo=tensorflow&logoColor=fec8bf">
    </a>
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Jupyter-fec8bf?style=for-the-badge&logo=Jupyter&logoColor=fec8bf">
    </a>
</div>

<br/>

<div align="center">
    <a href="https://github.com/EstevesX10/_REPO_NAME_/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/EstevesX10/_REPO_NAME_?style=flat&logo=gitbook&logoColor=fec8bf&label=License&color=fec8bf">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/repo-size/EstevesX10/_REPO_NAME_?style=flat&logo=googlecloudstorage&logoColor=fec8bf&logoSize=auto&label=Repository%20Size&color=fec8bf">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/stars/EstevesX10/_REPO_NAME_?style=flat&logo=adafruit&logoColor=fec8bf&logoSize=auto&label=Stars&color=fec8bf">
    </a>
    <a href="https://github.com/EstevesX10/_REPO_NAME_/blob/main/DEPENDENCIES.md">
        <img src="https://img.shields.io/badge/Dependencies-DEPENDENCIES.md-white?style=flat&logo=anaconda&logoColor=fec8bf&logoSize=auto&color=fec8bf"> 
    </a>
</div>

## Project Overview

`Sound Classification` is considered one of the most important tasks in the field of **deep learning**. It has great impact on applications of **voice recognition** within virtual assistants (Like Siri or Alexa), **customer service** as well as in **music and media content recommendation systems**. Moreover, it also impacts the Medical field wheteher to detect abnormalities in heartbeats or repiratory sounds. In addition it is also used within **Security and Surveillance systems** to help detect and assess a possible security breach inside a home whether it is infered by distress calls or even gunshots and glass breaking. Therefore, we aim to develop **deep learning algorithms** that can enable us to properly classify some environmental sounds provided by the `UrbanSound8k Dataset`.

## Project Development

### Dependencies & Execution

This project was developed using a `Notebook`. Therefore if you're looking forward to test it out yourself, keep in mind to either use a **[Anaconda Distribution](https://www.anaconda.com/)** or a 3rd party software that helps you inspect and execute it.

Therefore, for more informations regarding the **Virtual Environment** used in Anaconda, consider checking the [DEPENDENCIES.md](https://github.com/EstevesX10/_REPO_NAME_/blob/main/DEPENDENCIES.md) file.

### Planned Work

The project includes several key phases, including:

1. `Exploratory Data Analysis` : We begin by **examining the UrbanSound8k dataset** to gain deeper insights into its **structure and content** to helps us understand the distribution of sound classes.
2. `Data pre-processing` : **Cleaning and Preparing the audio samples** to ensure their consistency and quality over the .
3. `Feature Engineering` : Utilizing the **Librosa** library, we extract **meaningful features** from the audio data such as **Mel-frequency cepstral coefficients (MFCCs)**.
4. `Model architecture definition` : We develop the **architecture of artificial neural networks** tailored for sound classification, which involves **experimenting with different deep learning models**.
5. `Training and Performance Evaluation` : Employing the pre-partitioned dataset, we perform **10-fold cross-validation** on each developed networks to then assess the models' performance using key metrics such as accuracy and confusion matrices.
6. `Statistical Inference` : Perform a **Statistical Evaluation** of the performance between all the developed networks.

## UrbanSound8K Dataset

The `UrbanSound8k` dataset contains **8732 labeled sound excerpts** ($\le$ 4s) of urban sounds from **10 classes**: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the **urban sound taxonomy**.

For a **detailed description** of the dataset please consider checking the dataset web page available [here](https://urbansounddataset.weebly.com/urbansound8k.html). In case you are interested in the **compilation process**, the dataset creators have published a paper outlining the Taxonomy for Urban Sound Research. You can access it [here](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf).

### Additional Datasets

If you're interested in trying this project yourself, you'll need access to the `complete datasets` we've created. Since GitHub has **file size limits**, we've made them all available [here](https://drive.google.com/drive/folders/13LYorB-vWtQVRRSUJi1nwCNquWJY-Ubi?usp=drive_link).

## Project Results

### Model Performance

<!-- Network Architectures - Performances -->

<table width="100%">
    <thead>
        <th>
            <div align="center">
                Network Architecture
            </div>
        </th>
        <th>
            <div align="center">
                Final Global Confusion Matrix
            </div>
        </th>
    </thead>
    <tbody>
        <tr>
            <td width="25%">
                <p align="center" width="100%">
                    MLP
                </p>
            </td>
            <td width="75%">
                <p align="center">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/ModelPerformancePlots/MLP_GlobalConfusionMatrix.png" width="50%" />
                </p>
            </td>
        </tr>
        <tr>
            <td width="25%">
                <p align="center" width="100%">
                    CNN
                </p>
            </td>
            <td width="75%">
                <p align="center">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/ModelPerformancePlots/CNN_GlobalConfusionMatrix.png" width="50%" />
                </p>
            </td>
        </tr>
        <tr>
            <td width="25%">
                <p align="center" width="100%">
                    CNN Pre-Trained with YAMNET
                </p>
            </td>
            <td width="75%">
                <p align="center">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/ModelPerformancePlots/CNN_YAMNET_GlobalConfusionMatrix.png" width="50%" />
                </p>
            </td>
        </tr>
        <tr>
            <td width="25%">
                <p align="center" width="100%">
                    ResNet
                </p>
            </td>
            <td width="75%">
                <p align="center">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/ModelPerformancePlots/ResNet_GlobalConfusionMatrix.png" width="50%" />
                </p>
            </td>
        </tr>
    </tbody>
</table>

- **MLP**: Achieved **45%** accuracy, struggling with the complexity of the sound data.
- **CNN**: Performed better at **55%**, benefiting from 2D time-frequency representations (MFCCs).
- **YAMNet**: Leveraging transfer learning, YAMNet outperformed other models with **70%** accuracy.
- **ResNet**: Achieved **55%**, similar to CNN, but not as effective as YAMNet.

### Dimensionality Reduction Visualization

<!-- Data Distribution Scatter Plots -->

<table width="100%">
    <thead align="center">
        <th colspan="3">
            <div align="center">Data Distribution Scatter Plots</div>
        </th>
    </thead>
    <tbody>
        <tr>
            <td></td>
            <td>
                <div align="center">
                    1-Dimensional Processed MFCC's
                </div>
            </td>
            <td>
                <div align="center">
                    2-Dimensional Raw MFCC's
                </div>
            </td>
        </tr>
        <tr>
            <td width="10%">
                <p align="center" width="100%">
                    PCA
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/DataDistributionPlots/PCA-Plot-1D-Data.png" width="100%"/>
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/DataDistributionPlots/PCA-Plot-2D-Data.png" width="100%"/>
                </p>
            </td>
        </tr>
        <tr>
            <td width="10%">
                <p align="center" width="100%">
                    t-SNE
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/DataDistributionPlots/t-SNE-Plot-1D-Data.png" width="100%"/>
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/DataDistributionPlots/t-SNE-Plot-2D-Data.png" width="100%"/>
                </p>
            </td>
        </tr>
    </tbody>
</table>

### Critical Differences Diagram

<!-- Critical Differences Diagram -->

<table width="40%" height="40%">
    <thead>
        <th>
            <div align="center">
                Critical Differences Diagram
            </div>
        </th>
    </thead>
    <tbody>
        <tr>
            <td width="40%">
                <p align="center">
                    <img src="./Urban-Sound-Classification/ExperimentalResults/ModelPerformancePlots/CriticalDifferencesDiagram.png" />
                </p>
            </td>
        </tr>
    </tbody>
</table>

This **critical difference** diagram shows the ranks of every model:

1. **YAMNET (1)** is the best model, significantly **outperforming others**.
2. **CNN (2.6)** and **ResNet (2.6)** have similar performance, with **no statistical difference** between them.
3. **MLP (3.8)** is the worst, significantly **worse than YAMNET** and likely CNN / ResNet.

## Conclusion

Our experiments showed that **YAMNet** with transfer learning produced the best results. Regularization techniques such as **Dropout** and **L2 regularization** helped **reduce overfitting**. While the models performed well overall, difficulties in distinguishing **similar sound classes** suggest that further improvements in feature extraction and model design could enhance performance.

## Authorship

- **Authors** &#8594; [Gonçalo Esteves](https://github.com/EstevesX10), [Nuno Gomes](https://github.com/NightF0x26) and [Pedro Afonseca](https://github.com/PsuperX)
- **Course** &#8594; Machine Learning II [[CC3043](https://sigarra.up.pt/fcup/en/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=546532)]
- **University** &#8594; Faculty of Sciences, University of Porto

<div align="right">
<sub>

<!-- <sup></sup> -->

`README.md by Gonçalo Esteves`
</sub>

</div>
