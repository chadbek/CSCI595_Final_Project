import json


if __name__ == "__main__":
    path = "/home/gpuhead-2/genai_project/datasets/gqa/spatialFeatures/spatial/gqa_spatial_info.json"

    info = json.load(open(path, 'r'))    

    print(len(info))