# Deep learning based domain adaptation for mitochondria segmentation on EM volumes
                                                                                                                       
This repository contains the code to reproduce the methods described in the publication entitled "Deep learning based domain adaptation for mitochondria segmentation on EM volumes". 

The code and detailed instructions of the four different implemented strategies are available in the following links:

- [Style-transfer based domain adaptation](CUT).
- [Self-supervised learning (SSL) based domain adpation](SSL).
- [DAMT-Net](DAMT-Net).
- [Attention_Y-Net](Attention_Y-Net).

Qualitative results of all methods are summarized in the next figure:
<p align="center">
  <img src="./img/DAoverview.png" width="800"></a>
</p>

## Datasets

The EM datasets used are available here:
- [Lucchi++](https://sites.google.com/view/connectomics/ "Lucchi++")
- [Kasthuri++](https://sites.google.com/view/connectomics/ "Kasthuri++")
- [VNC](https://github.com/unidesigner/groundtruth-drosophila-vnc "VNC")
- [Histogram matched datasets](https://ehubox.ehu.eus/s/X3qRpYsPftxgjPw "Histogram matched datasets")

## Citation                                                                                                             
                                                                                                                        
This repository is the base of the following work:                                                                      
    
```                                                                                                                     
@misc{francobarranco2022deep,
      title={Deep learning based domain adaptation for mitochondria segmentation on EM volumes}, 
      author={Franco-Barranco, Daniel and Pastor-Tronch, Julio and Gonzalez-Marfil, Aitor and Mu{\~{n}}oz-Barrutia, Arrate and Arganda-Carreras, Ignacio},
      year={2022},
      eprint={2202.10773},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```                                                                                                                     
