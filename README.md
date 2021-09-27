# Sensor-Guided Optical Flow

Demo code for "Sensor-Guided Optical Flow", ICCV 2021

This code is provided to replicate results with flow hints obtained from LiDAR data.

**At the moment, we do not plan to release training code.**

[[Project page]](https://mattpoggi.github.io/projects/iccv2021poggi/) - [[Paper]](https://mattpoggi.github.io/assets/papers/poggi2021iccv.pdf) - [[Supplementary]](https://mattpoggi.github.io/assets/papers/poggi2021iccv_supp.pdf) 

![Alt text](https://mattpoggi.github.io/assets/img/sensorguidedflow/teaser.png?raw=true "Sensor-Guided Optical Flow estimation")

## Reference

If you find this code useful, please cite our work:
```shell
@inproceedings{Poggi_ICCV_2021,
  title     = {Sensor-Guided Optical Flow},
  author    = {Poggi, Matteo and
               Aleotti, Filippo and
               Mattoccia, Stefano},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2021}
}
```

## Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data](#data)
4. [Weights](#weights)
5. [Usage](#usage)
6. [Contacts](#contacts)
7. [Acknowledgments](#acknowledgments)

## Introduction

This paper proposes a framework to guide an optical flow network with external cues to achieve superior accuracy either on known or unseen domains.
Given the availability of sparse yet accurate optical flow hints from an external source, these are injected to modulate the correlation scores computed by a state-of-the-art optical flow network and guide it towards more accurate predictions.
Although no real sensor can provide sparse flow hints, we show how these can be obtained by combining depth measurements from active sensors with geometry and hand-crafted optical flow algorithms, leading to accurate enough hints for our purpose.
Experimental results with a state-of-the-art flow network on standard benchmarks support the effectiveness of our framework, both in simulated and real conditions.

## Installation

Install the project requirements in a new python 3 environment:

```
virtualenv -p python3 guided_flow_env
source guided_flow_env/bin/activate
pip install -r requirements.txt
```

Compile the `guided_flow` module, written in C (required for guided flow modulation):

```
cd external/guided_flow
bash compile.sh
cd ../..
```

## Data

Download [KITTI 2015 optical flow training set](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) and [precomputed flow hints](https://drive.google.com/drive/folders/1NU5SH3DdsFFI7wj_U6I6qhonZ2K69rFq?usp=sharing). Place them under the `data` folder as follows:

```
data
├──training
    ├──image_2
        ├── 000000_10.png
        ├── 000000_11.png
        ├── 000001_10.png
        ├── 000001_11.png
        ...
    ├──flow_occ
        ├── 000000_10.png
        ├── 000000_11.png
        ├── 000001_10.png
        ├── 000001_11.png
        ...
    ├──hints
        ├── 000002_10.png
        ├── 000002_11.png
        ├── 000003_10.png
        ├── 000003_11.png
        ...
```

## Weights

We provide QRAFT models tested in Tab. 4. Download the [weights](https://drive.google.com/drive/folders/104PQ0J2CW3h905LbAlXHX3RLJtTdJ1zJ?usp=sharing) and unzip them under `weights` as follows:

```
weights
├──raw
    ├── C.pth
    ├── CT.pth
    ...
├──guided
    ├── C.pth
    ├── CT.pth
    ...    
```

## Usage

You are now ready to run the `demo_kitti142.py` script:

```
python demo_kitti142.py --model CTK --guided
```

Use `--model` to specify the weights you want to load among C, CT, CTS and CTK. By default, raw models are loaded, specify `--guided` to load guided weights and enable sensor-guided optical flow.

## Contacts
m [dot] poggi [at] unibo [dot] it

## Acknowledgments

Thanks to Zachary Teed for sharing [RAFT](https://github.com/princeton-vl/RAFT) code, used as codebase in our project.