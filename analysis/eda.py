from utils import get_path, load_dataset_csv
import json
from config import PATH

def get_region_name(path):
    """
    Get region name from path.

    Args:
        - path (str): CSV file path
    Returns:
        - str: Region name
    """
    region = (path.split("/")[-1]).split("_")[0]
    return region

def is_csv(path):
    """
    Checks for CSV file.

    Args:
        - path (str): CSV file path
    Returns:
        - bool: True for CSV
    """
    path = path.split(".")
    return path[-1] == "csv"

def combine_dataset(paths, output_path):
    """
    Combines dataset and save to JSON.

    Args:
        - paths (list of str): List of paths
        - ouput_path (str): JSON path of output dataset
    """
    dic = {}
    for path in paths:
        if not is_csv(path):
            continue
        df = load_dataset_csv(path)
        region = get_region_name(path)
        if region not in dic.keys():
            dic[region] = []
        for index, row in df.iterrows():
            dic[region].append((row['text'], row['ipa']))

    all_data = []
    idx = 1
    for k in dic:
        for item in dic[k]:
            small_dic = {}
            small_dic['id'] = idx
            small_dic['region'] = k
            small_dic['text'] = item[0]
            small_dic['ipa'] = item[1]
            all_data.append(small_dic)
            idx += 1
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

def save_merged_file():
    """
    Saved merged train and test set.
    """
    # folder = PATH['project_folder_path'] + "/" + "data/input/train"
    folder = PATH['project_folder_path'] + "/" + "data/input/test"

    merged_train_data_path = PATH['project_folder_path'] + "/" + "data/input/train" + "/" + "train.json"
    merged_test_data_path = PATH['project_folder_path'] + "/" + "data/input/test" + "/" + "test.json"
    paths = get_path(folder)

    correct_paths = []
    for path in paths:
        if is_csv(path):
            correct_paths.append(path)

    paths = correct_paths
    print(f"Total files={len(paths)}")
    sum = 0
    
    # combine_dataset(paths, merged_train_data_path)
    combine_dataset(paths, merged_test_data_path)

    for path in paths:
        get_region_name(path)

    for path in paths:
        df = load_dataset_csv(path)
        print(f"path={path} len={len(df)}")
        sum += len(df)
    print(f"sum={sum}")

if __name__ == "__main__":
   save_merged_file()