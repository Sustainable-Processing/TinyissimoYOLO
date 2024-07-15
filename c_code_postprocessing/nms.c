#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float x, y, w, h; // Top-left coordinates, width and height of the bounding box
    float score; // Confidence score of the bounding box
    int class_id; // Class ID of the bounding box
} Box;

// Function to calculate the area of a bounding box
float area(Box box) {
    return box.w * box.h;
}

// Function to calculate the intersection over union (IoU) of two bounding boxes
float iou(Box a, Box b) {
    float inter_x1 = fmax(a.x, b.x);
    float inter_y1 = fmax(a.y, b.y);
    float inter_x2 = fmin(a.x + a.w, b.x + b.w);
    float inter_y2 = fmin(a.y + a.h, b.y + b.h);

    float inter_area = fmax(0, inter_x2 - inter_x1) * fmax(0, inter_y2 - inter_y1);
    float union_area = area(a) + area(b) - inter_area;

    return inter_area / union_area;
}

// Comparison function for sorting boxes by score in descending order
int compare_boxes(const void *a, const void *b) {
    Box *boxA = (Box *)a;
    Box *boxB = (Box *)b;
    float diff = boxB->score - boxA->score;
    if (diff > 0) return 1;
    else if (diff < 0) return -1;
    else return 0;
}

// Function to perform non-maximum suppression
void non_maximum_suppression(Box *boxes, int *indices, int num_boxes, float iou_threshold) {
    // Sort the boxes by their scores in descending order
    qsort(boxes, num_boxes, sizeof(Box), compare_boxes);
    
    int *suppressed = (int *)calloc(num_boxes, sizeof(int));

    for (int i = 0; i < num_boxes; ++i) {
        if (suppressed[i]) continue;

        indices[i] = 1; // Mark this box as kept

        for (int j = i + 1; j < num_boxes; ++j) {
            if (suppressed[j]) continue;

            // Only compare boxes of the same class
            if (boxes[i].class_id == boxes[j].class_id && iou(boxes[i], boxes[j]) > iou_threshold) {
                suppressed[j] = 1; // Suppress the box
            }
        }
    }

    free(suppressed);
}

// Function to print the boxes
void print_boxes(Box *boxes, int *indices, int num_boxes) {
    for (int i = 0; i < num_boxes; ++i) {
        if (indices[i]) {
            printf("Box %d: [x: %.2f, y: %.2f, w: %.2f, h: %.2f], score: %.2f, class_id: %d\n", i, boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h, boxes[i].score, boxes[i].class_id);
        }
    }
}



// int main() {
//     Box boxes[] = {
//         {1, 1, 9, 9, 0.9, 1},
//         {2, 2, 9, 9, 0.85, 1},
//         {8, 8, 7, 7, 0.7, 2},
//         {12, 12, 8, 8, 0.6, 2},
//         {3, 3, 6, 6, 0.95, 1}
//     };
//     int num_boxes = sizeof(boxes) / sizeof(boxes[0]);
//     float threshold = 0.3;
//     int *indices = (int *)calloc(num_boxes, sizeof(int));

//     non_maximum_suppression(boxes, indices, num_boxes, threshold);
//     print_boxes(boxes, indices, num_boxes);

//     free(indices);
//     return 0;
// }
