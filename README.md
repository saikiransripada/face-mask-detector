
# face-mask-detector

# Installation
1. Make sure you are an administrator to avoid running into issues.
2. Install [Anaconda (preferred)](https://docs.anaconda.com/anaconda/install/) or any other Python environment.
3. Follow these steps in the below link to install Anaconda on Windows.
	- [https://docs.anaconda.com/anaconda/install/windows/](https://docs.anaconda.com/anaconda/install/windows/) 
4. If you are on mac, run these commands.
	- `brew cask install anaconda`
	-   `export PATH="/usr/local/anaconda3/bin:$PATH"`
5.	Create a new environment in Ananconda
	-  `conda create --name myenv python`
	- Replace `myenv` with your convenient name.
6. Clone this repository.
7. Once you have new environment created, activate the environment using one of the below commands.
	- `source activate myenv`
	- `conda activate myenv`
8. In your environment, navigate to this project repository.
9. Install the dependent packages using the below command.
	- `pip3 install requirements.txt`
	- Most of the packages should be available in the Anaconda environment. It just installs the missing dependencies.
10. Your setup should be complete if you run these steps without errors.

# Usage

## Extract embeddings
Run the below command to extract embeddings from a dataset.

    python3 face_recognition/extract_embeddings.py --dataset face_recognition/dataset --embeddings face_recognition/output/embeddings.pickle --detector face_detector --embedding-model face_recognition/openface_nn4.small2.v1.t7

## Train the model
Run the below command to train the model.

    python3 face_recognition/train_model.py --embeddings face_recognition/output/embeddings.pickle --recognizer face_recognition/output/recognizer.pickle --le face_recognition/output/le.pickle

> Embedding and training should be done only for a new installation or when something is added to or removed from a dataset.

## Run the web server
Run the below command to run the web server. No configuration needed.

    python3 run_web_server.py

## Run the project
Run the below command to run the project.

    python3 detect_mask_video.py --embedding-model face_recognition/openface_nn4.small2.v1.t7 --recognizer face_recognition/output/recognizer.pickle --le face_recognition/output/le.pickle

> Make sure to take your pictures and add them to the below directory in order to recognize you.
> `face_recognition/dataset`
