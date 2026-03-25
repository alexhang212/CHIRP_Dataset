"""Visualize COCO/CSV annotations"""

import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import sys
from ast import literal_eval

sys.path.append("./")
from utils import COCOReader

ColourDictionary = {'hd_bill_tip': (31, 119, 180), 'hd_bill_tiplower': (255, 127, 14), 'hd_bill_base': (44, 160, 44), 'hd_eye_right': (214, 39, 40), 'bd_shoulder_right': (148, 103, 189), 'bd_tail_base': (140, 86, 75), 'bd_tail_tip': (227, 119, 194), 'bd_ankle_right': (127, 127, 127), 'bd_feet_right': (188, 189, 34), 'hd_eye_left': (23, 190, 207), 'bd_shoulder_left': (31, 119, 180), 'bd_ankle_left': (255, 127, 14), 'bd_feet_left': (44, 160, 44)}

def _wait_for_nav(window):
    """Wait for navigation key. Returns 'next' or 'quit'."""
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return "quit"
        elif key == ord('n'):
            return "next"


def _overlay_controls(img):
    cv2.putText(img, "n=next  q=quit", (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def VisualizeCOCO(JSONPath, ImagePath):

    Reader = COCOReader.COCOParser(JSONPath)
    ImgIDs = Reader.get_imgIds()

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    print(f"Loaded {len(ImgIDs)} images. Controls: n=next, q=quit")

    for imgID in ImgIDs:
        annIDs = Reader.get_annIds(imgID)
        anns = Reader.load_anns(annIDs)

        ImgPath = Reader.im_dict[imgID]["file_name"]
        print(ImgPath)

        img = cv2.imread(os.path.join(ImagePath, ImgPath))

        for ann in anns:
            ClassString = Reader.cat_dict[ann["category_id"]]["name"]

            if "bbox" in ann and len(ann["bbox"]) > 0:
                bbox = ann["bbox"]
                bboxXYXY = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
                bboxXYXY = [int(x) for x in bboxXYXY]
                img = cv2.rectangle(img, (bboxXYXY[0],bboxXYXY[1]), (bboxXYXY[2],bboxXYXY[3]), (0,255,0), 2)
                img = cv2.putText(img, ClassString, (bboxXYXY[0],bboxXYXY[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if "segmentation" in ann and len(ann["segmentation"]) > 0:
                mask = ann["segmentation"][0]
                mask = [int(x) for x in mask]
                pts = np.array([[mask[x],mask[x+1]] for x in range(0,len(mask),2)])
                img = cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=5)

            if "keypoints" in ann and len(ann["keypoints"]) > 0:
                KPs = Reader.cat_dict[ann["category_id"]]["keypoints"]
                for kp in range(len(KPs)):
                    if ann["keypoints"][kp*3+2] > 0:
                        x = ann["keypoints"][kp*3]
                        y = ann["keypoints"][kp*3+1]
                        img = cv2.circle(img, (int(x),int(y)), 1, ColourDictionary[KPs[kp]], 10)

        _overlay_controls(img)
        cv2.imshow("img", img)
        if _wait_for_nav("img") == "quit":
            break

    cv2.destroyAllWindows()

def VisualizeCSV(CSVPath, ImagePath):
    """Visualize CSV annotations"""
    df = pd.read_csv(CSVPath)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    if "Annotation" in df.columns:
        df["Annotation"] = df["Annotation"].apply(literal_eval)
    elif "BBox" and "Keypoints" in df.columns:
        df["BBox"] = df["BBox"].apply(literal_eval)
        df["Keypoints"] = df["Keypoints"].apply(literal_eval)
        ClassJSON = os.path.join(os.path.dirname(CSVPath),"keypoint_classes.json")
        with open(ClassJSON, 'r') as f:
            ClassDict = json.load(f)
    elif "BBox" and "segmentation" in df.columns:
        df["BBox"] = df["BBox"].apply(literal_eval)
        df["segmentation"] = df["segmentation"].apply(literal_eval)
        
    UnqImages = df["ImagePath"].unique()
    print(f"Loaded {len(UnqImages)} images. Controls: n=next, q=quit")

    for imgPath in UnqImages:
        print(imgPath)
        ImgSubDF = df[df["ImagePath"] == imgPath]

        img = cv2.imread(os.path.join(ImagePath, imgPath))

        for i in range(len(ImgSubDF.index)):
            ClassString = ImgSubDF["Class"].iloc[i]
            TypeString = ImgSubDF["Type"].iloc[i]
            
            if TypeString == "BBox":
                bbox = ImgSubDF["Annotation"].iloc[i]
                bboxXYXY = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
                bboxXYXY = [int(x) for x in bboxXYXY]
                
                img = cv2.rectangle(img, (bboxXYXY[0],bboxXYXY[1]), (bboxXYXY[2],bboxXYXY[3]), (0,255,0), 2)
                img = cv2.putText(img, ClassString, (bboxXYXY[0],bboxXYXY[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            if "Mask" in TypeString:
                mask = ImgSubDF["Annotation"].iloc[i][0]
                mask = [int(x) for x in mask]
                pts = np.array([[mask[x],mask[x+1]] for x in range(0,len(mask),2)])
                img = cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=5)
            if TypeString == "Keypoints2D":
                KP2D = ImgSubDF["Keypoints"].iloc[i]
                bbox = ImgSubDF["BBox"].iloc[i]
                
                bboxXYXY = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
                bboxXYXY = [int(x) for x in bboxXYXY]
                
                img = cv2.rectangle(img, (bboxXYXY[0],bboxXYXY[1]), (bboxXYXY[2],bboxXYXY[3]), (0,255,0), 2)
                img = cv2.putText(img, ClassString, (bboxXYXY[0],bboxXYXY[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                for kp in range(len(ClassDict.keys())):
                    kpName = list(ClassDict.keys())[kp]
                    if KP2D[kp*3+2] > 0:
                        x = KP2D[kp*3]
                        y = KP2D[kp*3+1]
                        img = cv2.circle(img, (int(x),int(y)), 1, ColourDictionary[kpName], 10)
                
            if TypeString == "Ring":
                mask = ImgSubDF["segmentation"].iloc[i]
                bbox = ImgSubDF["BBox"].iloc[i]
                
                bboxXYXY = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
                bboxXYXY = [int(x) for x in bboxXYXY]
                
                img = cv2.rectangle(img, (bboxXYXY[0],bboxXYXY[1]), (bboxXYXY[2],bboxXYXY[3]), (0,255,0), 2)
                img = cv2.putText(img, ClassString, (bboxXYXY[0],bboxXYXY[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                mask = [int(x) for x in mask]
                imsize = [img.shape[0],img.shape[1]]
                empty = np.zeros(imsize, dtype=np.uint8)
                
                pts = np.array([[mask[x],mask[x+1]] for x in range(0,len(mask),2)])
                
                MaskedImage = cv2.fillPoly(empty, [pts], color=[255,0,0])
                MaskedImage_colored = cv2.merge([np.zeros_like(MaskedImage), MaskedImage, MaskedImage])
                blended = cv2.addWeighted(img,0.5, MaskedImage_colored, 0.5, 0)
                img = np.where(MaskedImage[:, :, None] > 0, blended, img)
                
        _overlay_controls(img)
        cv2.imshow("img", img)
        if _wait_for_nav("img") == "quit":
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize annotations from JSON or CSV files')
    parser.add_argument('--annot', required=True, help='Path to annotation file (JSON or CSV)')
    parser.add_argument('--image', required=True, help='Path to image directory')
    
    args = parser.parse_args()
    
    # Determine file type by extension
    if args.annot.lower().endswith('.json'):
        VisualizeCOCO(args.annot, args.image)
    elif args.annot.lower().endswith('.csv'):
        VisualizeCSV(args.annot, args.image)
    else:
        print("Error: Annotation file must be either .json or .csv")
        sys.exit(1)