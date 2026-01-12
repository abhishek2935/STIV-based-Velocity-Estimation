import pyorc
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from dask.diagnostics import ProgressBar
from matplotlib import patches
import copy
import numpy as np


def process(VideoPath , JSONpath , bbox_coords , NetCDF_path ):

    video_file = VideoPath # Parameter 1 - Vid Path
    cam_config = pyorc.load_camera_config(JSONpath) # Parameter 2 - JSON path

    # Parameter 3 - bbox coords as list 
    stabilize = bbox_coords 

    video = pyorc.Video(
        video_file,
        camera_config=cam_config,
        start_frame=0,
        end_frame=125,
        stabilize=stabilize,
        h_a=0.,
    )

    da = video.get_frames()
    da_norm = da.frames.normalize()
    da_norm_proj = da_norm.frames.project(method="numpy") # remove method = numpy to use default OpenCV method 

    piv = da_norm_proj.frames.get_piv(engine="numba") # Velocimetry Computation (PIV / FFPIV / OpenPIV)

    if (NetCDF_path):
        piv.to_netcdf(NetCDF_path)    

    return piv 


##--------

def mask():
    ds_mask = copy.deepcopy
    mask_corr = ds_mask.velocimetry.mask.corr(inplace=True)
    mask_minmax = ds_mask.velocimetry.mask.minmax(inplace=True)
    mask_rolling = ds_mask.velocimetry.mask.rolling(inplace=True)
    mask_outliers = ds_mask.velocimetry.mask.outliers(inplace=True)
    mask_var = ds_mask.velocimetry.mask.variance(inplace=True)
    mask_angle = ds_mask.velocimetry.mask.angle(inplace=True)
    mask_count = ds_mask.velocimetry.mask.count(inplace=True)
    

def mask2(VideoPath , NetCDF_path , Masked_NetCDF_Path):

    video_file = VideoPath     # parameter 1 
    ds = xr.open_dataset(NetCDF_path)  # parameter 2

    video = pyorc.Video(video_file, start_frame=0, end_frame=125)

    # borrow the camera config from the velocimetry results
    video.camera_config = ds.velocimetry.camera_config

    #da_rgb = video.get_frames(method="rgb")

    ## Mask
    ds_mask2 = copy.deepcopy(ds)
    ds_mask2.velocimetry.mask.corr(inplace=True)
    ds_mask2.velocimetry.mask.minmax(inplace=True)
    ds_mask2.velocimetry.mask.rolling(inplace=True)
    ds_mask2.velocimetry.mask.outliers(inplace=True)
    ds_mask2.velocimetry.mask.variance(inplace=True)
    ds_mask2.velocimetry.mask.angle(angle_tolerance=0.5*np.pi)
    ds_mask2.velocimetry.mask.count(inplace=True)
    ds_mask2.velocimetry.mask.window_mean(wdw=2, inplace=True, tolerance=0.5, reduce_time=True)

    #ds_mean_mask2 = ds_mask2.mean(dim="time", keep_attrs=True)


    ds_mask2.velocimetry.set_encoding()
    ds_mask2.to_netcdf(Masked_NetCDF_Path)

    mean_plt(VideoPath , NetCDF_path)

    return ds_mask2 


def mean_plt(VideoPath , NetCDF_path):
    ds = xr.open_dataset(NetCDF_path)  # parameter 2

    video_file = VideoPath
    video = pyorc.Video(video_file, start_frame=0, end_frame=125)
    
    video.camera_config = ds.velocimetry.camera_config

    da_rgb = video.get_frames(method="rgb")
    da_rgb_proj = da_rgb.frames.project()
    p = da_rgb_proj[0].frames.plot()

    ds_mean = ds.mean(dim="time", keep_attrs=True)

    # first a pcolormesh
    ds_mean.velocimetry.plot.pcolormesh(
        ax=p.axes,
        alpha=0.3,
        cmap="rainbow",
        add_colorbar=True,
        vmax=0.6
    )

    ds_mean.velocimetry.plot(
        ax=p.axes,
        color="w",
        alpha=0.5,
        width=0.0015,
    )

    p.axes.figure.savefig("Modularize/layered_plot.png", dpi=300, bbox_inches="tight")