[![DOI](https://zenodo.org/badge/288950002.svg)](https://zenodo.org/badge/latestdoi/288950002)

## "SEDENOSS: SEparating and DENOising Seismic Signals with dual-path recurrent neural network architecture" 
_Novoselov Artemii, Peter Balazs, G&ouml;tz Bokelmann_

<img src="https://img.univie.ac.at/fileadmin/user_upload/i_img/documents_imgw/graphic/logo_imgw_color_with_text_2100x660.png"
     alt="IMGW Logo"
     style="float: left; margin-right: 10px;" width=50%/>

This is the code to reproduce ["SEDENOSS: SEparating and DENOising Seismic Signals with dual-path recurrent neural network architecture"](https://www.essoar.org/doi/10.1002/essoar.10504944.2) paper 2021

## Abstract
> Seismologists have to deal with overlapping and noisy signals. Techniques such as source separation can be used to solve this problem. Over the past few decades, signal processing techniques used for source separation have advanced significantly for multi-station settings. But not so many options are available when it comes to single-station data. Using Machine Learning, we demonstrate the possibility of separating sources for single-station, one-component seismic recordings. The technique that we use for seismic signal separation is based on a dual-path recurrent neural network which is applied directly to the time domain data. Such source separation may find applications in most tasks of seismology, including earthquake analysis, aftershocks, nuclear verification, seismo-acoustics, and ambient-noise tomography. We train the network on seismic data from STanford EArthquake Dataset (STEAD) and demonstrate that our approach is a) capable of denoising seismic data and b) capable of separating two earthquake signals from one another. In this work, we show that Machine Learning is useful for earthquake-induced source separation. 

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

## Work in progress
This repo will be updated as soon as I'm done with all the other burning papers and repos (~end of october). Promise!

## Know issues
If you are having troubles with `torch-audiomentation` try installing it with `pip install git+https://github.com/asteroid-team/torch-audiomentations.git`
