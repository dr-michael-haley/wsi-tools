import wsi_background
import os

os.environ['JAVA_TOOL_OPTIONS'] = '-Xmx128G'
os.environ["PATH"] += os.pathsep + r"C:/bioformats/bftools"
os.environ["PATH"] += os.pathsep + r"C:/bioformats/bioformats2raw-0.10.0-rc2/bin"
os.environ["PATH"] += os.pathsep + r"C:/bioformats/raw2ometiff-0.8.0-rc1/bin"

def main():
    client = wsi_background.start_dask_cluster(n_workers=8, threads_per_worker=2, memory_limit='128GB')
    
    wsi_background.vsi_background_subtract(vsi_path = "S:/Mariapia_Grassia/SlideScanner/10_01_20250325_120429.vsi", 
                                        raw_folder = 'Images_zarr',
                                        output_ometiff = 'pyramid.ome.tiff',
                                        overwrite=False,
                                        max_workers=16,
                                        load_into_memory=True,
                                        patch_size=1024)
if __name__ == "__main__":
    main()

