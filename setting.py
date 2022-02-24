from pytest import mark
import yaml
f = open("config.yaml",'r+',encoding='utf-8')
cfg = yaml.load(f, Loader=yaml.FullLoader)

camid = cfg['camid']
data_path = cfg["data_path"]
calibration_cfg = cfg['calibration']
marker = cfg['marker']
tracking = cfg['tracking']
def init():
    global RESCALE, N_, M_, x0_, y0_, dx_, dy_, fps_
    RESCALE = tracking['RESCALE']

    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker (in original size)
    dx_, dy_: the horizontal and vertical interval between adjacent markers (in original size)
    fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
    """
    N_ = marker['N_']
    M_ = marker['M_']
    fps_ = tracking['fps_']
    x0_ = marker['x0_'] / RESCALE
    y0_ = marker['y0_'] / RESCALE
    dx_ = marker['dx_'] / RESCALE
    dy_ = marker['dy_'] / RESCALE
