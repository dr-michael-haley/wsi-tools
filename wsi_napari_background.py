import wsi_background
import os

os.environ['JAVA_TOOL_OPTIONS'] = '-Xmx128G'
os.environ["PATH"] += os.pathsep + r"C:/bioformats/bftools"
os.environ["PATH"] += os.pathsep + r"C:/bioformats/bioformats2raw-0.10.0-rc2/bin"
os.environ["PATH"] += os.pathsep + r"C:/bioformats/raw2ometiff-0.8.0-rc1/bin"

def main():
    #client = wsi_background.start_dask_cluster(n_workers=8, threads_per_worker=2, memory_limit='128GB')
    
    wsi_background.napari_tile_inspector(zarr_path = r"E:/PROJECTS/Michael_Haley/WSI/05_03_20250328_154237_zarr")

if __name__ == "__main__":
    main()

