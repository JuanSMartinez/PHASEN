# PHASEN

Unofficial implementation of the PHASEN network by [Yin et al., (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6489). The network is designed as a phase-and-harmonics-aware speech enhancement network.


# Requirements

* Python 
* Pytorch
* ffmpeg
* youtube-dl

# Databases

From the original work of [Yin et al., (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6489), the [AVSpeech](https://looking-to-listen.github.io/avspeech/index.html) dataset is used along with the [AudioSet](https://research.google.com/audioset/) to create a noisy speech dataset. The contents under the `datasets/AVSpeech` directory correspond to code and files to download and process the AVSpeech dataset from the available information on the web on the Spring of 2021.

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
