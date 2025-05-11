import os
import shutil
import glob

def copy_files_with_prefix(src_pattern, dst_dir, prefix):
    """Copies files matching a pattern to a destination directory with a prefix."""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    copied_count = 0
    src_files = glob.glob(src_pattern)
    if not src_files:
        print(f"Warning: No files found matching pattern: {src_pattern}")
        return copied_count

    for src_file_path in src_files:
        if os.path.isfile(src_file_path):
            base_filename = os.path.basename(src_file_path)
            dst_filename = f"{prefix}_{base_filename}"
            dst_file_path = os.path.join(dst_dir, dst_filename)
            try:
                shutil.copy2(src_file_path, dst_file_path) # copy2 preserves metadata
                copied_count += 1
            except Exception as e:
                print(f"Error copying {src_file_path} to {dst_file_path}: {e}")
    print(f"Copied {copied_count} files from {os.path.dirname(src_pattern)} to {dst_dir} with prefix '{prefix}'")
    return copied_count

def merge_datasets(base_db_path, dataset_names, target_db_name="all_dataset"):
    """Merges specified datasets into a single target dataset directory."""
    target_base_path = os.path.join(base_db_path, target_db_name)
    
    print(f"Target directory: {target_base_path}")

    # Define source and destination paths mapping
    data_splits = {
        "train": ("yolo/train", "images", "labels"),
        "val": ("yolo/val", "images", "labels"),
        "test": ("test", "images", "labels") # Assuming test labels exist
    }

    total_copied = 0

    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_prefix = dataset_name # Use dataset name as prefix

        for split, (src_subpath, img_folder, lbl_folder) in data_splits.items():
            src_base = os.path.join(base_db_path, dataset_name, src_subpath)
            
            # Define target directories
            target_img_dir = os.path.join(target_base_path, split, "images") # New structure
            target_lbl_dir = os.path.join(target_base_path, split, "labels") # New structure

            # Define source patterns
            src_img_pattern = os.path.join(src_base, img_folder, "*.*") # Match common image extensions
            src_lbl_pattern = os.path.join(src_base, lbl_folder, "*.txt")

            # Copy images
            print(f"Copying {split} images...")
            if os.path.exists(os.path.join(src_base, img_folder)):
                total_copied += copy_files_with_prefix(src_img_pattern, target_img_dir, dataset_prefix)
            else:
                print(f"Warning: Image source directory not found: {os.path.join(src_base, img_folder)}")

            # Copy labels
            print(f"Copying {split} labels...")
            if os.path.exists(os.path.join(src_base, lbl_folder)):
                 total_copied += copy_files_with_prefix(src_lbl_pattern, target_lbl_dir, dataset_prefix)
            else:
                print(f"Warning: Label source directory not found: {os.path.join(src_base, lbl_folder)}")

    print(f"\nFinished merging datasets. Total files copied: {total_copied}")


if __name__ == "__main__":
    database_path = "yolov5/database"
    datasets_to_merge = [
        "China_MotorBike",
        "China_Drone",
        "United_States",
        "Norway",
        "Japan"
    ]
    
    merge_datasets(database_path, datasets_to_merge)
