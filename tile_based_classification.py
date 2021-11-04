from osgeo import gdal_array, osr, gdal
import os
import glob
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.colors

def stack_sort(stack_in, lst_code, sorted_list):
    """Sort the stack."""  
    b,r,c = stack_in.shape
    stack_sorted = np.zeros((r,c,b), dtype=np.uint16)
    
    len_list_bands = len(lst_code)
    
    c = np.zeros((len_list_bands),dtype=np.uint8)
    count = 0
    count_sort = 0
    while count_sort != len_list_bands:
        if lst_code[count] == sorted_list[count_sort]:
            c[count_sort] = count
            count_sort = count_sort + 1
            count = 0
        else:
            count = count + 1   
    print('[AI4EO_MOOC]_log: sorted list:', sorted_list)
    print('[AI4EO_MOOC]_log: bands:', c)
    for i in range(0, len_list_bands):
        stack_sorted[:,:,i]=stack_in[c[i],:,:]
        
    return stack_sorted


def sentinel2_format(total_stack):
    """
    Function that transforms the multistack into sentinel2 format array with bands in the right position for the AI model.
    
    Input:
        total_stack: array that is the concatenation of stack10, stack_20mTo10m and stack_60mTo10m

    Output: 
        sentinel2: sentinel2 format array  
    """
    row_tot, col_tot, bands_tot = total_stack.shape
    sentinel2 = np.zeros((row_tot, col_tot,bands_tot),dtype=np.uint16)   
    
    print("[AI4EO_MOOC]_log: Creating total stack with following bands list:")
    print("[AI4EO_MOOC]_log: Band 1 – Coastal aerosol")
    print("[AI4EO_MOOC]_log: Band 2 – Blue")
    print("[AI4EO_MOOC]_log: Band 3 – Green")
    print("[AI4EO_MOOC]_log: Band 4 – Red")
    print("[AI4EO_MOOC]_log: Band 5 – Vegetation red edge")
    print("[AI4EO_MOOC]_log: Band 6 – Vegetation red edge")
    print("[AI4EO_MOOC]_log: Band 7 – Vegetation red edge")
    print("[AI4EO_MOOC]_log: Band 8 – NIR")
    print("[AI4EO_MOOC]_log: Band 8A – Narrow NIR")
    print("[AI4EO_MOOC]_log: Band 9 – Water vapour")
    print("[AI4EO_MOOC]_log: Band 10 – SWIR – Cirrus")
    print("[AI4EO_MOOC]_log: Band 11 – SWIR")
    print("[AI4EO_MOOC]_log: Band 12 – SWIR")

    sentinel2[:, :, 0] = total_stack[:, :, 10]
    sentinel2[:, :, 1] = total_stack[:, :, 0]
    sentinel2[:, :, 2] = total_stack[:, :, 1]
    sentinel2[:, :, 3] = total_stack[:, :, 2]
    sentinel2[:, :, 4] = total_stack[:, :, 4]
    sentinel2[:, :, 5] = total_stack[:, :, 5]
    sentinel2[:, :, 6] = total_stack[:, :, 6]
    sentinel2[:, :, 7] = total_stack[:, :, 3]
    sentinel2[:, :, 8] = total_stack[:, :, 9]
    sentinel2[:, :, 9] = total_stack[:, :,11]
    sentinel2[:, :,10] = total_stack[:, :,12]
    sentinel2[:, :,11] = total_stack[:, :, 7]
    sentinel2[:, :,12] = total_stack[:, :, 8]
    
    del (total_stack)
    return sentinel2

def from_folder_to_stack(safe_path, data_bands_20m=True, data_bands_60m=True):
    """
    This function transform the .SAFE file into three different arrays (10m, 20m and 60m).

    Input: 
        safe_path: the path of the .SAFE file.
    data_bands_20m: if True, the function computes stack using Sentinel2 band with 20m of pixel resolution. Defaults to True.
    data_bands_60m: if True, the function computes stack using Sentinel2 band with 60m of pixel resolution. Defaults to True. 
    
    Output:
        stack_10m: stack with the following S2L1C bands (B02,B03,B04,B08)
        stack_20m: stack with the following S2L1C bands (B05,B06,B07,B11,B12,B8A)
        stack_60m: stack with the following S2L1C bands (B01,B09,B10) 

    """        
    
    level_folder_name_list = glob.glob(safe_path + 'GRANULE/*')    
    level_folder_name = level_folder_name_list[0]
    
    if level_folder_name.find("L2A") < 0:
        safe_path = [level_folder_name + '/IMG_DATA/']
    else:
        safe_path_10m = level_folder_name + '/IMG_DATA/R10m/'
        safe_path = [safe_path_10m]
    
    text_files = []

    for i in range(0,len(safe_path)):
        print("[AI4EO_MOOC]_log: Loading .jp2 images in %s" % (safe_path[i]))
        text_files_tmp = [f for f in os.listdir(safe_path[i]) if f.endswith('.jp2')]
        text_files.append(text_files_tmp)
        
    lst_stack_60m=[]
    lst_code_60m =[]
    lst_stack_20m=[]
    lst_code_20m =[]
    lst_stack_10m=[]
    lst_code_10m =[]
    for i in range(0,len(safe_path)):        
        
        print("[AI4EO_MOOC]_log: Reading .jp2 files in %s" % (safe_path[i]))
        for name in range(0, len(text_files[i])):            
            text_files_tmp = text_files[i]               
            if data_bands_60m == True:
                cond_60m = ( (text_files_tmp[name].find("B01") > 0) or (text_files_tmp[name].find("B09") > 0) 
                            or (text_files_tmp[name].find("B10") > 0))
                if cond_60m:
                    print("[AI4EO_MOOC]_log: Using .jp2 image: %s" % text_files_tmp[name])
                    lst_stack_60m.append(gdal_array.LoadFile(safe_path[i] + text_files_tmp[name]))
                    lst_code_60m.append(text_files_tmp[name][24:26])
                
            if data_bands_20m == True:                    
                cond_20m = (text_files_tmp[name].find("B05") > 0) or (text_files_tmp[name].find("B06") > 0) or (
                            text_files_tmp[name].find("B07") > 0) or (text_files_tmp[name].find("B11") > 0) or (
                                       text_files_tmp[name].find("B12") > 0) or (text_files_tmp[name].find("B8A") > 0)
                cond_60m_L2 = (text_files_tmp[name].find("B05_60m") < 0) and (text_files_tmp[name].find("B06_60m") < 0) and (
                            text_files_tmp[name].find("B07_60m") < 0) and (text_files_tmp[name].find("B11_60m") < 0) and (
                                       text_files_tmp[name].find("B12_60m") < 0) and (text_files_tmp[name].find("B8A_60m") < 0)
                cond_20m_tot = cond_20m and cond_60m_L2
                if cond_20m_tot:
                    print("[AI4EO_MOOC]_log: Using .jp2 image: %s" % text_files_tmp[name])
                    lst_stack_20m.append(gdal_array.LoadFile(safe_path[i] + text_files_tmp[name]))
                    lst_code_20m.append(text_files_tmp[name][24:26])
            else:
                stack_20m = 0
                    
            cond_10m = (text_files_tmp[name].find("B02") > 0) or (text_files_tmp[name].find("B03") > 0) or (
                        text_files_tmp[name].find("B04") > 0) or (text_files_tmp[name].find("B08") > 0)
            cond_20m_L2 = (text_files_tmp[name].find("B02_20m") < 0) and (text_files_tmp[name].find("B03_20m") < 0) and (
                        text_files_tmp[name].find("B04_20m") < 0) and (text_files_tmp[name].find("B08_20m") < 0)
            cond_60m_L2 = (text_files_tmp[name].find("B02_60m") < 0) and(text_files_tmp[name].find("B03_60m") < 0) and(
                        text_files_tmp[name].find("B04_60m") < 0) and (text_files_tmp[name].find("B08_60m") < 0)
            cond_10m_tot = cond_10m and cond_20m_L2 and cond_60m_L2
            
            if cond_10m_tot:
                print("[AI4EO_MOOC]_log: Using .jp2 image: %s" % text_files_tmp[name])
                lst_stack_10m.append(gdal_array.LoadFile(safe_path[i] + text_files_tmp[name]))
                lst_code_10m.append(text_files_tmp[name][24:26])
                 
    
    stack_10m=np.asarray(lst_stack_10m)
    sorted_list_10m = ['02','03','04','08']    
    print('[AI4EO_MOOC]_log: Sorting stack 10m...')
    stack_10m_final_sorted = stack_sort(stack_10m, lst_code_10m, sorted_list_10m)
    
    stack_20m=np.asarray(lst_stack_20m)
    sorted_list_20m = ['05','06','07','11','12','8A']
    print('[AI4EO_MOOC]_log: Sorting stack 20m...')
    stack_20m_final_sorted = stack_sort(stack_20m, lst_code_20m, sorted_list_20m)
    
    stack_60m=np.asarray(lst_stack_60m)
    sorted_list_60m = ['01','09','10']    
    print('[AI4EO_MOOC]_log: Sorting stack 60m...')
    stack_60m_final_sorted = stack_sort(stack_60m, lst_code_60m, sorted_list_60m)
                
    return stack_10m_final_sorted, stack_20m_final_sorted, stack_60m_final_sorted

def sliding(shape, window_size, step_size=None, fixed=True):
    """
    Sliding function.

    Input:
        shape: The target shape.
        window_size: The shape of the window.
        step_size: Defaults to None.
        fixed: Defaults to True.
    Output:
        windows
    """
    h, w = shape
    if step_size:
        h_step = step_size
        w_step = step_size
    else:
        h_step = window_size
        w_step = window_size
        
    h_wind = window_size
    w_wind = window_size
    windows = []
    for y in range(0, h, h_step):
        for x in range(0, w, w_step):
            h_min = min(h_wind, h - y)
            w_min = min(w_wind, w - x)
            if fixed:
                if h_min < h_wind or w_min < w_wind:
                    continue
            window = (x, y, w_min, h_min)
            windows.append(window)

    return windows

def resample_3d(stack, row10m, col10m, rate):
    """
    Wrapper of ndimage zoom. Bilinear interpolation for resampling array.

    Input:
        stack: Array to be resampled.
        row10m: The expected row.
        col10m: The expected col.
        rate: The rate of the transformation.
    Output:
        stack_10m: Resampled array.
    """
    row, col, bands = stack.shape
    print("[AI4EO_MOOC]_log: Array shape (%d,%d,%d)" % (row, col, bands))
    stack_10m = np.zeros((row10m, col10m, bands),dtype=np.uint16)
    print("[AI4EO_MOOC]_log: Resize array bands from (%d,%d,%d) to (%d,%d,%d)" % (
        row, col, bands, row10m, col10m, bands))
    for i in range(0, bands):
        stack_10m[:, :, i] = ndimage.zoom(stack[:, :,i], rate)
        
    del (stack)
    
    return stack_10m

def array2raster(newRasterfn, dataset, array, dtype):
    """
    Function that transforms array into tif file with the same metadata of dataset input.

    Input:
        newRasterfn: File path and name of the new tif.
        dataset: Is the dataset that contains metadata information.
        array: Array to be transformed.
        dtype: Tif type.

    Output:
        tiff file saved into newRasterfn path.
    
    """
   
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform()

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte": 
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    elif dtype == "UInt16":
        GDT_dtype = gdal.GDT_UInt16
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


if __name__ == "__main__":
    data_input = {}
    data_input['main_path'] = 'C:\Users\ochsp\OneDrive\Personal projects\Data Science\Python dev\earthobslearn\earthobslearn'

    data_input['sentinel2_safe_name'] ='S2A_MSIL1C_20210331T100021_N0300_R122_T33TTG_20210331T113321.SAFE/'
    data_input['sentinel2_safe_path'] = data_input['main_path']+ 'raw_data\tile_detection_training' + data_input['sentinel2_safe_name']

    safe_path = data_input['sentinel2_safe_path']
    stack_10m, stack_20m, stack_60m = from_folder_to_stack(safe_path)

    r,c,b10=stack_10m.shape

    print('[AI4EO_MOOC]_log: Resampling stack with 20 m pixel size...')
    stack_20mTo10m = resample_3d(stack_20m,r,c,2)

    print('[AI4EO_MOOC]_log: Resampling stack with 60 m pixel size...')
    stack_60mTo10m = resample_3d(stack_60m,r,c,6)

    print('[AI4EO_MOOC]_log: Creating multistack with 10-20-60 m pixel size')
    total_stack=np.concatenate((stack_10m,stack_20mTo10m,stack_60mTo10m), axis=2)

    from matplotlib import pyplot as plt
    plt.plot(total_stack[200,200,:])
    plt.show()

    print('[AI4EO_MOOC]_log: Sentile2 L1C formatting...')
    s2_arr = sentinel2_format(total_stack)

    plt.plot(s2_arr[10000,10000,:])
    plt.show()

    # Load a pretrained model
    data_input['pre_trained_model_name'] = 'keras_sentinel2_classification_trained_model_e50_9190.h5'
    data_input['pre_trained_model_path'] = data_input['main_path']+'trained_models/'

    print('[AI4EO_MOOC]_log: Load pretrained model')
    model = tf.keras.models.load_model(data_input['pre_trained_model_path'] + data_input['pre_trained_model_name'])
    model.summary()

    print('[AI4EO_MOOC]_log: Divide all image into windows for inference step')
    batch_size = 10
    target_shape = (s2_arr.shape[0], s2_arr.shape[1])
    windows = sliding(target_shape, 64, fixed=True)

    windows_ = iter(windows)
    windows_class = iter(windows)

    total_chips = len(windows)
    num_steps = int(total_chips/batch_size)

    print('[AI4EO_MOOC]_log: Inference step...')

    img_classes = np.zeros((total_stack.shape[0], total_stack.shape[1]), dtype=np.uint8)


    predictions = []
    progbar = tf.keras.utils.Progbar(num_steps)
    for b in range(num_steps):
        chips = np.empty([batch_size, 64, 64, 13])
        for k in range(batch_size):
            ymin, xmin, xmax, ymax = next(windows_)
            chips[k] = s2_arr[xmin:xmin+xmax, ymin:ymin+ymax, :]        
        
        preds = model.predict(chips)
        predictions.append(np.argmax(preds, axis=-1))
        for i in range(0, batch_size):
            ymin_cl, xmin_cl, xmax_cl, ymax_cl = next(windows_class)
            img_classes[xmin_cl:xmin_cl+xmax_cl, ymin_cl:ymin_cl+ymax_cl] = predictions[b][i]
        progbar.update(b + 1)

    print('[AI4EO_MOOC]_log: Show the final classified image...')

    label=['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial',
        'Pasture','PermanentCrop','Residential','River','SeaLake']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", 
        ["linen","darkgreen","lime","grey","k",
        "olive","darkgoldenrod","lightgrey","azure","lightblue"])

    plt.figure(figsize=(10,10))
    plt.imshow(img_classes, cmap=cmap)
    cbar=plt.colorbar()
    cbar.ax.set_yticklabels(label)
    plt.show()

    print('[AI4EO_MOOC]_log: Saving predicted image in a tiff format...')

    data_input['output_file_name'] = 'predicted.tif'
    data_input['output_file_path'] = data_input['main_path']+'output_data/'

    OutNameTiff  = data_input['output_file_path'] + data_input['output_file_name']

    level_folder_name_list = glob.glob(data_input['sentinel2_safe_path'] + 'GRANULE/*')
    level_folder_name = level_folder_name_list[0] + '/IMG_DATA/'
    text_files_tmp = [f for f in os.listdir(level_folder_name) if f.endswith('.jp2')]
    InNameRaster = level_folder_name + text_files_tmp[1]    
    InDatabase = gdal.Open(InNameRaster)

    array2raster(OutNameTiff, InDatabase, img_classes, 'Byte')

    print('[AI4EO_MOOC]_log: Saving rgb sentinel2 image in a tiff format...')

    data_input['output_file_name_rgb_tif'] = 'rgb.tif'

    OutNameTiff_rgb  = data_input['output_file_path'] + data_input['output_file_name_rgb_tif']
    s2_arr_rgb=np.zeros((r,c,3), dtype=np.uint16)
    s2_arr_rgb[:,:,0]=s2_arr[:,:,3]
    s2_arr_rgb[:,:,1]=s2_arr[:,:,2]
    s2_arr_rgb[:,:,2]=s2_arr[:,:,1]
    array2raster(OutNameTiff_rgb, InDatabase, s2_arr_rgb, 'UInt16')
