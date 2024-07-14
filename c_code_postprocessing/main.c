#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "helper.h"

#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

unsigned char voc_colors[][3] = {
    {255, 0, 0},      // Red for "aeroplane"
    {0, 255, 0},      // Green for "bicycle"
    {0, 0, 255},      // Blue for "bird"
    {255, 255, 0},    // Yellow for "boat"
    {0, 255, 255},    // Cyan for "bottle"
    {255, 0, 255},    // Magenta for "bus"
    {192, 192, 192},  // Silver for "car"
    {128, 0, 0},      // Maroon for "cat"
    {128, 128, 0},    // Olive for "chair"
    {0, 128, 0},      // Dark Green for "cow"
    {128, 0, 128},    // Purple for "diningtable"
    {0, 128, 128},    // Teal for "dog"
    {0, 0, 128},      // Navy for "horse"
    {255, 165, 0},    // Orange for "motorbike"
    {255, 192, 203},  // Pink for "person"
    {128, 128, 128},  // Gray for "pottedplant"
    {210, 105, 30},   // Chocolate for "sheep"
    {255, 20, 147},   // Deep Pink for "sofa"
    {32, 178, 170},   // Light Sea Green for "train"
    {173, 216, 230}   // Light Blue for "tvmonitor"
};


// Define struct which represents a bounding box
typedef struct {
    float x, y, w, h; // Top-left coordinates, width and height of the bounding box
    float score; // Confidence score of the bounding box
    int class_id; // Class ID of the bounding box
} Box;

// Function prototype declaration
void drawBox(unsigned char* imageData, int imageWidth, int imageHeight, Box box, unsigned char color[3]);
void draw_bounding_boxes_for_indices(Box *boxes, int *indices, int num_boxes, unsigned char *imageData, int imageWidth, int imageHeight);
void saveImageAsPNG(unsigned char* imageData, int width, int height, const char* filename);
void transpose(float original[24][30], float transposed[30][24]);
void getAMax(float *array, int size, float *maxValue, int *maxIndex);
void get_boxes_passing_threshold(float transposed[30][24], unsigned char* imageData, int width, int height, Box boxes[30], int *boxesCount, float confidence_threshold);
void putTextOnImage(unsigned char *imageData, int width, int height, int channels, const char *text, int x, int y);


// Declaring functiion prototypes for the functions in nms.c
void non_maximum_suppression(Box *boxes, int *indices, int num_boxes, float iou_threshold);


int model_image_width = 160;
int model_image_height = 192;

int MODEL_OUTPUT_ARRAY_SIZE =24;
int MODEL_OUTPUT_SUBARRAY_SIZE = 30;

int main() {

    // Setp 1: Load the image data from the PNG file.

    // Replace the manual reading and filling with red color with:
    int width, height, channels;

    // Use stbi_load to read the image data from the png file
    unsigned char *image_data = stbi_load("../gap9_tests/1.png", &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image.\n");
        return 1;
    }


    //Step 2: Transpose the output array of the model.

    // Declare the 2D arrays we shall use for the transpose operation.
    float original[MODEL_OUTPUT_ARRAY_SIZE][MODEL_OUTPUT_SUBARRAY_SIZE];
    float transposed[MODEL_OUTPUT_SUBARRAY_SIZE][MODEL_OUTPUT_ARRAY_SIZE];

    // Copy the values from the MODEL_OUTPUT_ARRAY to the 'original' array
    for (int i = 0; i < MODEL_OUTPUT_ARRAY_SIZE; i++) {
        for (int j = 0; j < MODEL_OUTPUT_SUBARRAY_SIZE; j++) {
            original[i][j] = MODEL_OUTPUT_ARRAY[i][j];
        }
    }

    // Call the transpose function
    transpose(original, transposed);


    // Step 3: Process the transposed array to get potential bounding boxes and draw them on the image.
    
    // Define an array of Box structs to store the bounding boxes
    Box boxes[30];
    int boxesCount = 0;
    float confidence_threshold = 0.18;
    get_boxes_passing_threshold(transposed, image_data, width, height, boxes, &boxesCount, confidence_threshold);

    // for (int i = 0; i < boxesCount; i++) {
    //     printf("Box %d: [x: %.2f, y: %.2f, w: %.2f, h: %.2f], score: %.2f, class_id: %d\n", i, boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h, boxes[i].score, boxes[i].class_id);
    // }


    // Step 4: Perform non-maximum suppression to remove overlapping boxes

    // Define an array of integers to store the indices of the boxes to keep after non-maximum suppression
    int *indices = (int *)calloc(30, sizeof(int));
    float iou_threshold = 0.5;
    non_maximum_suppression(boxes, indices, 30, iou_threshold);

    draw_bounding_boxes_for_indices(boxes, indices, boxesCount, image_data, width, height);
    free(indices);

    //Step 4: Save the image with the bounding boxes drawn on it.
    saveImageAsPNG(image_data, width, height, "test_output.png");
    return 0;
}


void drawBox(unsigned char* imageData, int imageWidth, int imageHeight, Box box, unsigned char color[3]){
    int boxThickness = 1;

    for (int y = box.y; y < box.y + box.h; y++) {
        for (int x = box.x; x < box.x + box.w; x++) {
            // Check if the current coordinates are within the image boundaries
            if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight) {
                // Check if the current pixel is within the thickness range of the border
                if ((x >= box.x && x < box.x + boxThickness) || // Left border
                    (x < box.x + box.w && x >=  box.x + box.w - boxThickness) || // Right border
                    (y >= box.y && y < box.y + boxThickness) || // Top border
                    (y < box.y + box.h && y >= box.y + box.h - boxThickness)) { // Bottom border
                    // Calculate the position in the imageData array
                    int pos = (y * imageWidth + x) * 3; // Multiply by 3 for RGB
                    // Set the pixel color for the border
                    imageData[pos] = color[0];     // Red
                    imageData[pos + 1] = color[1]; // Green
                    imageData[pos + 2] = color[2]; // Blue
                }
            }
        }
    }

    putTextOnImage(imageData, imageWidth, imageHeight, 3, voc_names[box.class_id], box.x, box.y);

    }


void draw_bounding_boxes_for_indices(Box *boxes, int *indices, int num_boxes, unsigned char *imageData, int imageWidth, int imageHeight) {
    for (int i = 0; i < num_boxes; ++i) {
        if (indices[i]) {
            // Draw the bounding box if index is marked as kept
            drawBox(imageData, imageWidth, imageHeight, boxes[i], voc_colors[boxes[i].class_id]);
        }
    }
}


void saveImageAsPNG(unsigned char* imageData, int width, int height, const char* filename) {
    // The stride_length parameter is the image width * number of channels.
    // Assuming imageData is 3 channels (RGB) per pixel.
    int stride_length = width * 3;
    
    // Write the image to a PNG file.
    // The function automatically handles the creation and compression of the PNG file.
    if (!stbi_write_png(filename, width, height, 3, imageData, stride_length)) {
        perror("Failed to save PNG image");
    }
}

// Function to transpose a 2D array of floats 24x30 to 30x24. Used to transpose output of the model.
void transpose(float original[24][30], float transposed[30][24]) {
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 30; j++) {
            transposed[j][i] = original[i][j];
        }
    }
}

// Function to find the maximum value and its index in an array
void getAMax(float *array, int size, float *maxValue, int *maxIndex) {
    *maxValue = array[0];
    *maxIndex = 0;

    for (int i = 1; i < size; i++) {
        if (array[i] > *maxValue) {
            *maxValue = array[i];
            *maxIndex = i;
        }
    }
}

void get_boxes_passing_threshold(float transposed[30][24], unsigned char* imageData, int width, int height, Box boxes[30], int *boxesCount, float confidence_threshold) {
    float threshold = confidence_threshold;

    for(int rowNumb = 0; rowNumb < 30; rowNumb++) {
        float row[20];
        for (int i = 0; i < 20; i++) {
            row[i] = transposed[rowNumb][i+4];
        }

        // Find the maximum value and its index in the row
        float max_value;
        int max_index;
        getAMax(row, 20, &max_value, &max_index);

    
        if (max_value > threshold) {
    
            float scale_x = width / model_image_width;
            float scale_y = height / model_image_height;

            float x = transposed[rowNumb][0] * scale_x;
            float y = transposed[rowNumb][1] * scale_y;
            float w = transposed[rowNumb][2] * scale_x;
            float h = transposed[rowNumb][3] * scale_y;

            float left = (x - w / 2);
            float top = (y - h / 2);

            if (left < 0) {
                left = 0;
            }
            if (top < 0) {
                top = 0;
            }
            
            int boxX = left;
            int boxY = top;
            float boxWidth = w;
            float boxHeight = h;

            unsigned char color[3] = {255, 0, 0};
            
            
            Box box;
            box.x = boxX;
            box.y = boxY;
            box.w = boxWidth;
            box.h = boxHeight;
            box.score = max_value;
            box.class_id = max_index;

            boxes[*boxesCount] = box;

            (*boxesCount)++;

            // drawBox(imageData, width, height, box, color); // Draw the bounding box on the image, use for debugging.
        }
    }

}

void putTextOnImage(unsigned char *imageData, int width, int height, int channels, const char *text, int x, int y) {
    char buffer[99999]; // Buffer to store the generated vertices
    int num_quads;

    // Generate the vertex buffer for the text
    num_quads = stb_easy_font_print((float)x, (float)y, (char *)text, NULL, buffer, sizeof(buffer));

    // Corrected loop to iterate through each quad and set the corresponding pixels in the image data
    for (int i = 0; i < num_quads * 4; i++) {
        float *quad = (float *)(buffer + i * 16); // Each quad has 4 vertices, each vertex has 4 floats (x, y, s, t), hence 16 bytes per quad

        // Iterate through the four corners of the quad
        for (int j = 0; j < 4; j++) {
            int ix = (int)quad[j * 4]; // x coordinate of the j-th vertex
            int iy = (int)quad[j * 4 + 1]; // y coordinate of the j-th vertex

            // Check if the pixel is within the image bounds
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                int index = (iy * width + ix) * channels; // Calculate the index in the image data array

                // Set the pixel color to white (for simplicity, assuming the image is RGB or RGBA)
                imageData[index] = 255;     // R
                imageData[index + 1] = 255; // G
                imageData[index + 2] = 255; // B
                if (channels == 4) {
                    imageData[index + 3] = 255; // A (if the image has an alpha channel)
                }
            }
        }
    }
}

