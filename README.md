# Implementation-of-GelSight
An implementation of [GelSight Wedge](https://arxiv.org/abs/2106.08851).

Test passed on Ubuntu 18.04 and Windows 10. 🚩
## Requirements
python>=3.7

opencv-python

opencv4.x (C++ Version)

pybind11

pyyaml

argparse

glob

numpy

matplotlib

scipy

skimage

open3d

## Hardware
🔨Coming soon!🔨
## Software

### Step -1: set `config.yaml`
`camid` for `cv2.VideoCapture(camid)`.

`sample: from` for `{data_path}/sample_{sample_from}.jpg`.

`sample: to` for `{data_path}/sample_{sample_to}.jpg`.

`data_path` for `{data_path}`
### Step 0: run `pref_ref_and_sample.py`
    python pref_ref_and_sample.py -r -s


`-r` or `--ref` for capturing `{data_path}/ref.jpg`.

Click `left button` to take `ref.jpg`, you can click more than once until you are satiesfied. Then press `q` to exit or continue. 

`-s` or `--ref` for capturing `{data_path}/sample_xx.jpg`.

Click `left button` to take `sample_xx.jpg`. If `sample_to - sample_from` pictures are captured, it will terminate automatically. Or you can manually press `q` to exit in advance or continue. 

### Step 1: run `calibration_abberration.py`

Capture & save `ref.jpg` .
Resize to (320, 427) ?
![avatar](/asset/ref.jpg)

### Step 2: Get prepared for running `calibration.py`


#### 2.1 Measure `self.BallRad` eg: 3 
![avatar](/asset/BallRad.jpg)
#### 2.2 Measure `self.Pixmm` = $\frac{Length\ (mm)}{Pixel}$ 
##### 2.2.1 Capture & save `Pixmm.jpg`

Eg: 
![avatar](/asset/Pixmm.jpg)

##### 2.2.2 Get measurement of `mm`
Eg: mm = 3.40(mm)
![avatar](/asset/Pixmm_mm.jpg)

##### 2.2.3 Use `mesaure_Pixmm.py` to select 2 keypoints and calculate their distance (in `pixel`).

Click once on the first keypoint, then click once on the other keypoint, you will see an arrow linking 2 keypoints with their distance.
Eg: distance = 103.07764
![avatar](/asset/Pixmm_result.png)

##### 2.2.4 Calculate the Pixmm and fill it into `calibration.py`
In `line 13` of `calibration.py`:

`self.Pixmm` = $\frac{Length\ (mm)}{Pixel} = \frac{3.40}{103.0776}=0.03298$ 
#### 2.3 Capture & save 30 x `sample_xx.jpg`, from `sample_1.jpg` to `sample_30,jpg`. eg:
![avatar](/asset/sample_1.jpg)

#### 2.4 run `calibration.py`


### Step 3: run `test_poisson.py` or `Test0Cable.py`

Get reconstruction result.


## Improvements
🔨Coming soon!🔨
Thin plate spline for inpaint of markers?
## References
We used https://github.com/siyuandong16/gelsight_heightmap_reconstruction for calibration and heightmap reconstruction and https://github.com/GelSight/tracking for tracking.
## Acknowledgements
https://arxiv.org/abs/2106.08851

https://github.com/siyuandong16/gelsight_heightmap_reconstruction

https://github.com/GelSight/tracking

https://tutorial.cytron.io/2020/12/29/raspberry-pi-zero-usb-webcam/