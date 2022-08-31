from PIL import Image
import rasterio
import multiprocessing as mp
import numpy as np
import glob
import os
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.pyplot as plt
from skimage import io, img_as_bool, measure, morphology
import cv2

from shapely.geometry import box, Polygon, Point
import pyproj
from shapely.ops import transform, unary_union, orient
import shapely
import pickle
from pyproj import Geod
from skimage.segmentation import find_boundaries
import multiprocessing as mp

import argparse

parser = argparse.ArgumentParser(description="DeeplabV3+ And Evaluation")

# model parameters
parser.add_argument('--oak-fp', type=str)
parser.add_argument('--year', type=int)

args = parser.parse_args()

def make_global(oak_fp, year):
    global oak_fp_global
    global year_global

    oak_fp_global = oak_fp
    year_global = year

def driver(im):
    fn = im.split('.')[0]
    with rasterio.open(os.path.join(oak_fp_global, 'san_jose_naip_512', year_global, 'raw_tif', 
                                    im)) as inds:
        city_lim = gpd.read_file('/oak/stanford/groups/deho/building_compliance/san_jose_suppl/san_jose_City_Limits.geojson')
        city_lim = city_lim['geometry'].values[0]
        
        # get TIF bounds
        bounds = inds.bounds
        geom = box(*bounds)

        # prepare to convert TIF bounds to standard 4326
        wgs84 = pyproj.CRS('EPSG:26910')
        utm = pyproj.CRS('EPSG:4326')

        project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

        # convert
        utm_geom = shapely.ops.transform(project, geom)
        
        iou = utm_geom.intersection(city_lim).area/utm_geom.area
        if iou > 0:
            df = gpd.read_file(os.path.join(oak_fp_global, 'san_jose_suppl', 'san_jose_Zoning_Districts.geojson'),
                           mask=utm_geom)
            df = df[(df['ZONING'].str.contains('R-1')) | (df['ZONING'].str.contains('R-2')) |\
                 ((df['ZONING'].str.contains('R-M')) & (df['ZONING'] != 'R-MH'))]
            df = gpd.clip(df, utm_geom)
#             df = df[(df['ZONINGABBREV'].str.contains('R-')) | \
#                    ((df['ZONINGABBREV'] == 'A(PD)') & (df['PDUSE'] == 'Res'))]
            
            if len(df) > 0:
                # filter masks based on the residential zones
                res_zone = unary_union(df['geometry'])
                
                df_masks = gpd.read_file(os.path.join(oak_fp_global, 'san_jose_suppl', 'san_jose_OSM_footprints.geojson'), 
                                     mask=res_zone)
                
                if len(df_masks) > 0:
                    df_masks = gpd.clip(df_masks, df['geometry'])

                    # there are some buildings that only partially intersect the residneital zones -- filter out
                    df_masks['iou'] = df_masks['geometry'].apply(lambda x: x.intersection(res_zone).area/x.area)

                    # filter masks based on size
                    geod = Geod(ellps="WGS84")
                    # apply orient() before passing to Geod so that the area is not negative
                    df_masks['area'] = df_masks['geometry'].apply(lambda x: geod.geometry_area_perimeter(orient(x))[0])
                    df_masks = df_masks[(df_masks['area'] >= 15) & (df_masks['iou'] >= 0.8)] # CAN CHANGE THIS ARBITRARILY

                    if len(df_masks) > 0:
                        df_masks['binary_wts'] = df_masks['area'].apply(lambda x: 1 if x >= 15 and x <= 115 else 0)
                        df_masks['continuous_wts'] = df_masks['area'].apply(lambda x: 1 if x >= 15 and x <= 115 else 0)

                        df = df.to_crs('epsg:26910')
                        df_masks = df_masks.to_crs('epsg:26910')

                        mask_im, trans = mask(inds, list(df['geometry']))
                        mask_im = np.rollaxis(mask_im, 0, 3)

                        mask_im_footprints, _ = mask(inds, list(df_masks['geometry']))
                        mask_im_footprints = np.rollaxis(mask_im_footprints, 0, 3)
                        mask_footprints = np.zeros((mask_im_footprints.shape[0], mask_im_footprints.shape[1]))
                        mask_footprints[np.sum(mask_im_footprints == 0, axis=2) < 4] = 1


                        ### ---- MAKE LOSS_WEIGHTS ----
                        # make a pixel lat/lon map
                        band1 = inds.read(1)
                        height = band1.shape[0]
                        width = band1.shape[1]
                        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                        xs, ys = rasterio.transform.xy(inds.transform, rows, cols)
                        lons = np.array(xs)[0]
                        lats = [i[0] for i in np.array(ys)]

                        # segmentation from scikit-image.measure.label
                        labels = measure.label(mask_footprints)

                        # labels image in clusters #1-num_clusters (cluster 0 is the background, so we exclude in the loop)
                        clusters = []
                        for i in range(1, len(np.unique(labels))):
                            clusters.append(np.column_stack(np.where(labels == i)))

                        loss_weights_mask = np.zeros_like(mask_footprints, dtype=float)
                        for cluster in clusters:
                            # for each cluster, just take the first point and find the associated area of building
                            first = cluster[0]
                            point = Point(lons[first[1]], lats[first[0]])
                            building = df_masks[df_masks.contains(point)]

                            assert not building.empty

                            area = building['area'].values[0]

                            for c in cluster:
                                loss_weights_mask[c[0], c[1]] = area.round(2)
                        
                        # last safeguard against non-square tiles
                        if mask_im.shape[0] != 512 or mask_im.shape[1] != 512:
                            if mask_im.shape[0] >= 250 or mask_im.shape[1] >= 250:
                                dim = (512, 512)
                                mask_im = cv2.resize(mask_im, dim, interpolation = cv2.INTER_AREA)
                                loss_weights_mask = cv2.resize(loss_weights_mask, dim, interpolation = cv2.INTER_AREA)
                            else:
                                # unusable for training, but save anyway because we can still infer these buildings
                                np.save(os.path.join(oak_fp_global, 'san_jose_naip_512', year_global, 'keep_infer_images',
                                     f'{fn}.npy'), mask_im)
                                return
                        np.save(os.path.join(oak_fp_global, 'san_jose_naip_512', year_global, 'images',
                                     f'{fn}.npy'), mask_im)
                        np.save(os.path.join(oak_fp_global, 'san_jose_naip_512', year_global, 'masks',
                                     f'{fn}_mask.npy'), loss_weights_mask)
                    else:
                        with open(os.path.join(oak_fp_global, 'san_jose_naip_512', 
                                     year_global, 'no_res_buildings.txt'), 'a') as w:
                            w.write(f'{fn}\n')
                else:
                    with open(os.path.join(oak_fp_global, 'san_jose_naip_512', 
                                     year_global, 'no_res_buildings.txt'), 'a') as w:
                        w.write(f'{fn}\n')
            else:
                with open(os.path.join(oak_fp_global, 'san_jose_naip_512', 
                                     year_global, 'no_res.txt'), 'a') as w:
                    w.write(f'{fn}\n')
        else:
            with open(os.path.join(oak_fp_global, 'san_jose_naip_512', 
                                     year_global, 'not_sj.txt'), 'a') as w:
                w.write(f'{fn}\n')

def read_txt(fp):
    with open(fp, 'r') as f:
        return [i[:-1] for i in f.readlines()]




# year = '2018'
# oak_fp = '/oak/stanford/groups/deho/building_compliance/'

oak_fp = args.oak_fp
year = str(args.year)

fp_out = os.path.join(oak_fp, 'san_jose_naip_512', year)

if not os.path.exists(fp_out):
    os.mkdir(fp_out)

for i in ['images', 'masks', 'keep_infer_images']:
    if not os.path.exists(os.path.join(fp_out, i)):
        os.mkdir(os.path.join(fp_out, i))


done = [i.replace('.npy', '') for i in os.listdir(os.path.join(oak_fp, 'san_jose_naip_512', 
                                     year, 'images'))]

for i in ['not_sj.txt', 'no_res_buildings.txt', 'no_res.txt']:
    if os.path.exists(os.path.join(oak_fp, 'san_jose_naip_512', year, i)):
        done = done + read_txt(os.path.join(oak_fp, 'san_jose_naip_512', year, i))
        
done = done + [i.replace('.npy', '') for i in os.listdir(os.path.join(oak_fp, 'san_jose_naip_512', 
                                     year, 'keep_infer_images'))]

tifs = os.listdir(os.path.join(oak_fp, 'san_jose_naip_512', year, 'raw_tif'))
tifs = [i.replace('.tif', '') for i in tifs]

leftover = list(set(tifs) - set(done))
leftover = [f'{i}.tif' for i in leftover]

nprocs = mp.cpu_count()

p = mp.Pool(processes=nprocs, initializer=make_global, initargs=(oak_fp, year, ))
p.map(driver, leftover)