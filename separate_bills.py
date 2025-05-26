from dataclasses import dataclass

import cv2


@dataclass
class BBox:
    """Where x0,y0 is top-left and x1,y1 is bottom-right
    """
    x0: float
    y0: float
    x1: float
    y1: float


def __bbox_overlap(box1: tuple[int,int,int,int], box2: tuple[int,int,int,int]) -> int:
    """Compute overlap of 2 bboxes

    Args:
        box1 (tuple[int,int,int,int]): x,y,w,h format
        box2 (tuple[int,int,int,int]): x,y,w,h format

    Returns:
        int: area
    """
    
    # convert to BBox format for my sanity
    Box1 = BBox(box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3])
    Box2 = BBox(box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3])

    if Box1.x0 > Box2.x1 or Box2.x0 > Box1.x1 or Box1.y0 > Box2.y1 or Box2.y0 > Box1.y1: return 0

    width = min(Box1.x1, Box2.x1) - max(Box1.x0, Box2.x0) 
    height = min(Box1.y1, Box2.y1) - max(Box1.y0, Box2.y0) 

    # print(width * height)
    return width * height


def is_bbox_child(potential_child: tuple[int,int,int,int], potential_adult: tuple[int,int,int,int], overlap_threshold: float = 0.9) -> bool:
    """_summary_

    Args:
        potential_child (tuple[int,int,int,int]): _description_
        potential_adult (tuple[int,int,int,int]): _description_
        overlap_threshold (float, optional): What ratio of the child is overlapped. Defaults to 0.9.

    Returns:
        bool: _description_
    """
    
    # convert to BBox format for my sanity
    child_area = potential_child[2] * potential_child[3]
    adult_area = potential_adult[2] * potential_adult[3]

    if child_area > adult_area: return False # child cannot be bigger than adult

    overlap_area = __bbox_overlap(potential_child, potential_adult)
    if overlap_area == 0: return False

    assert adult_area >= overlap_area

    if overlap_area/child_area > overlap_threshold: return True
    return False


# Read the image
image_data = cv2.imread('Image 1.png')

img_height, img_width = image_data.shape[:2]
image_area = img_width * img_height

print("image_data.shape")
print(image_data.shape)

# Preprocessing: grayscale, blur, and edge detection
gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"len of contours: {len(contours)}")

# Collect bounding boxes based on size criteria
bounding_boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    if bbox_area / image_area > 0.05: bounding_boxes.append((x, y, w, h))
    # bounding_boxes.append((x, y, w, h))
print(f"len of bounding_boxes: {len(bounding_boxes)}")

# bounding_boxes = [
#     [0,0,10,20],
#     [1,1,2,2],
#     [11,21,1,1]
# ]
    

bounding_boxes.sort(key=lambda x: x[2]*x[3]) # smallest first for average efficiency in the next step

final_crops = []

# we want to retain the bounding_boxes that are the child of none
for i in range(0, len(bounding_boxes)):
    selected_box = bounding_boxes[i]
    selected_box_is_child = False
    for j in range(0, len(bounding_boxes)):
        if i == j: continue
        compare_box = bounding_boxes[j]
        if is_bbox_child(selected_box, compare_box): 
            selected_box_is_child = True
            break
    if not selected_box_is_child: final_crops.append(selected_box)

print(f"Len of final_crops is: {len(final_crops)}")

image_copy = image_data.copy()
for box in final_crops:
    start = (box[0], box[1])
    end = (box[0] + box[2], box[1] + box[3])
    cv2.rectangle(image_copy, start, end, (255,0,0), 3)

cv2.namedWindow('Image with Rectangle', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image with Rectangle', 800, 600)
cv2.imshow('Image with Rectangle', image_copy)
cv2.waitKey(0)