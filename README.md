# PHASEN

Unofficial implementation of the PHASEN network by [Yin et al., (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6489). The network is designed as a phase-and-harmonics-aware speech enhancement network.


# Requirements

* Python 3+
* Pytorch
* Scipy
* Numpy
* `ffmpeg`
* `youtube-dl`
* `gshuf` from `coreutils` if using Mac OS
* `mir_eval` library (optional for computing the SDR metric). Install from [here](https://craffel.github.io/mir_eval/)
* `python-pesq` (optional for computing the PESQ metric). Install from [here](https://github.com/ludlows/python-pesq)
* `pystoi` (optional for computing the STOI metric). Install from [here](https://github.com/mpariente/pystoi)

# Datasets

From the original work of [Yin et al., (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6489), the [AVSpeech](https://looking-to-listen.github.io/avspeech/index.html) dataset is used along with the [AudioSet](https://research.google.com/audioset/) dataset to create a noisy speech dataset. The bash script under the `datasets/AVSpeech` directory is used to download and process the AVSpeech dataset from the information available on the web. Please note that the CSV files that contain the video ID's from the dataset must be downloaded in advance. Place such files along with the script in the `datasets/AVSpeech` folder. The AudioSet dataset can be downloaded directly by running the script inside `datasets/AudioSet`.

# References

The following is a minimal `bibtex` citation of the work by [Yin et al., (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6489) and other relevant sources.

	# Main work by Yin et al.
	@inproceedings{yin2020phasen,
	title={Phasen: A phase-and-harmonics-aware speech enhancement network},
	author={Yin, Dacheng and Luo, Chong and Xiong, Zhiwei and Zeng, Wenjun},
	booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	volume={34},
	number={05},
	pages={9458--9465},
	year={2020},
	url = {https://ojs.aaai.org/index.php/AAAI/article/view/6489/6345}
	}

	#AVSpeech dataset
	@article{ephrat2018looking,
	title={Looking to listen at the cocktail party: A speaker-independent audio-visual model for speech separation},
	author={Ephrat, A. and Mosseri, I. and Lang, O. and Dekel, T. and Wilson, K and Hassidim, A. and Freeman, W. T. and Rubinstein, M.},
	journal={arXiv preprint arXiv:1804.03619},
	year={2018}
	}

	# Paper including cIRM computations
	@article{williamson2015complex,
	title={Complex ratio masking for monaural speech separation},
	author={Williamson, Donald S and Wang, Yuxuan and Wang, DeLiang},
	journal={IEEE/ACM transactions on audio, speech, and language processing},
	volume={24},
	number={3},
	pages={483--492},
	year={2015}
	}
