[![DOI](https://zenodo.org/badge/288950002.svg)](https://zenodo.org/badge/latestdoi/288950002)

## "SEDENOSS: SEparating and DENOising Seismic Signals with dual-path recurrent neural network architecture" 
_Novoselov Artemii, Peter Balazs, G&ouml;tz Bokelmann_

<img src="https://img.univie.ac.at/fileadmin/user_upload/i_img/documents_imgw/graphic/logo_imgw_color_with_text_2100x660.png"
     alt="IMGW Logo"
     style="float: left; margin-right: 10px;" width=50%/>

This is the code to reproduce ["SEDENOSS: SEparating and DENOising Seismic Signals with dual-path recurrent neural network architecture"](https://www.essoar.org/doi/10.1002/essoar.10504944.2) paper 2021

## How to install?
```bash
%cd /content
!rm -rf '/content/source-separation/'
!git clone https://github.com/IMGW-univie/source-separation.git
%cd /content/source-separation/sedenoss
!pip install -r requirements.txt
!pip install -e .
```

## How to use?
Please refer to
```bash
%cd /content/source-separation/sedenoss/sedenoss
!python train.py --help
```

## Easy start
Start `SEDENOSS.ipynb` in the Google Colaboratory and follow instructions

For earthquake related applications please download the data from [the STEAD Dataset](https://github.com/smousavi05/STEAD) and adopt the notebooks accoringly to your data locations (as there is no automated way to download the data as of 18.11.2020)
