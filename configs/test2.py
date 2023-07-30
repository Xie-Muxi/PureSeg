import os
import sys
import argparse

def print_directory_tree(folder_path, depth, max_files_per_folder, indent=0):
    if not os.path.exists(folder_path):
        print("目录不存在！")
        return

    if os.path.isfile(folder_path):
        print("提供的路径是文件路径而不是目录路径！")
        return

    if depth < 0:
        return

    files = os.listdir(folder_path)
    for idx, file in enumerate(sorted(files)):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            print(" " * indent + "|--", file)
        else:
            print(" " * indent + "|--", file)
            if max_files_per_folder == 0:
                continue
            print_directory_tree(file_path, depth - 1, max_files_per_folder - 1, indent + 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="打印目录树")
    parser.add_argument("folder_path", help="要查看的文件夹路径")
    parser.add_argument("-d", "--depth", type=int, default=-1, help="指定递归深度，-1表示无限制")
    parser.add_argument("-f", "--max_files_per_folder", type=int, default=-1, help="限制每个子目录显示的文件个数，-1表示无限制")
    args = parser.parse_args()

    folder_path = args.folder_path
    depth = args.depth
    max_files_per_folder = args.max_files_per_folder
    print_directory_tree(folder_path, depth, max_files_per_folder)
