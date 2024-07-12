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


// Function prototype declaration
void drawBox(unsigned char* imageData, int imageWidth, int imageHeight, int boxX, int boxY, int boxWidth, int boxHeight, unsigned char color[3]);
void saveImageAsPNG(unsigned char* imageData, int width, int height, const char* filename);
void transpose(float original[24][30], float transposed[30][24]);
void getAMax(float *array, int size, float *maxValue, int *maxIndex);
void helperFunction(float transposed[30][24], unsigned char* imageData, int width, int height);
void putTextOnImage(unsigned char *imageData, int width, int height, int channels, const char *text, int x, int y);

int model_image_width = 160;
int model_image_height = 192;

int main() {
    // Your code here

    // Replace the manual reading and filling with red color with:
    int width, height, channels;
    // Use stbi_load to read the image data
    unsigned char *image_data = stbi_load("gap9_tests/1.png", &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image.\n");
        return 1;
    }

    printf("Image width: %d, height: %d, channels: %d\n", width, height, channels);


    int MODEL_OUTPUT_ARRAY_SIZE =24;
    int MODEL_OUTPUT_SUBARRAY_SIZE = 30;

    //loop through the MODEL_OUTPUT_ARRAY and print
    for (int i = 0; i < MODEL_OUTPUT_ARRAY_SIZE; i++) {
        for (int j = 0; j < MODEL_OUTPUT_SUBARRAY_SIZE; j++) {
            printf("%f ", MODEL_OUTPUT_ARRAY[i][j]);
        }
        printf("\n"); // New line after printing each sub-array for better readability
    }

    float original[MODEL_OUTPUT_ARRAY_SIZE][MODEL_OUTPUT_SUBARRAY_SIZE];
    float transposed[MODEL_OUTPUT_SUBARRAY_SIZE][MODEL_OUTPUT_ARRAY_SIZE];

    // Copy the values from the MODEL_OUTPUT_ARRAY to the original array
    for (int i = 0; i < MODEL_OUTPUT_ARRAY_SIZE; i++) {
        for (int j = 0; j < MODEL_OUTPUT_SUBARRAY_SIZE; j++) {
            original[i][j] = MODEL_OUTPUT_ARRAY[i][j];
        }
    }

    // Call the transpose function
    transpose(original, transposed);

    printf("\nTransposed array:\n");
    // Print the transposed array
    for (int i = 0; i < MODEL_OUTPUT_SUBARRAY_SIZE; i++) {
        for (int j = 0; j < MODEL_OUTPUT_ARRAY_SIZE; j++) {
            printf("%f ", transposed[i][j]);
        }
        printf("\n"); // New line after printing each sub-array for better readability
    }

    // Call the helper function
    helperFunction(transposed, image_data, width, height);

    // unsigned char color[3] = {255, 255, 0};

    // int boxX = 50;
    // int boxY = 50;
    // int boxWidth = 100;
    // int boxHeight = 100;

    // drawBox(image_data, width, height, boxX, boxY, boxWidth, boxHeight, color);
    saveImageAsPNG(image_data, width, height, "test_output.png");
    return 0;
}


void drawBox(unsigned char* imageData, int imageWidth, int imageHeight, int boxX, int boxY, int boxWidth, int boxHeight, unsigned char color[3]) {
    int boxThickness = 1;

    for (int y = boxY; y < boxY + boxHeight; y++) {
        for (int x = boxX; x < boxX + boxWidth; x++) {
            // Check if the current coordinates are within the image boundaries
            if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight) {
                // Check if the current pixel is within the thickness range of the border
                if ((x >= boxX && x < boxX + boxThickness) || // Left border
                    (x < boxX + boxWidth && x >= boxX + boxWidth - boxThickness) || // Right border
                    (y >= boxY && y < boxY + boxThickness) || // Top border
                    (y < boxY + boxHeight && y >= boxY + boxHeight - boxThickness)) { // Bottom border
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

    putTextOnImage(imageData, imageWidth, imageHeight, 3, "class", 0, 0);

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


void transpose(float original[24][30], float transposed[30][24]) {
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 30; j++) {
            transposed[j][i] = original[i][j];
        }
    }
}


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

void helperFunction(float transposed[30][24], unsigned char* imageData, int width, int height) {
    float threshold = 0.18;

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
            printf("\nRow number: %i\n", rowNumb);
            printf("Max value: %f\n", max_value);
            printf("Max index: %i\n", max_index);
            printf("Class: %s\n", voc_names[max_index]);
            printf("X coordinate: %f\n", transposed[rowNumb][0]);
            printf("Y coordinate: %f\n", transposed[rowNumb][1]);
            printf("Width: %f\n", transposed[rowNumb][2]);
            printf("Height: %f\n", transposed[rowNumb][3]);


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
            drawBox(imageData, width, height, boxX, boxY, boxWidth, boxHeight, color);
            break;
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



