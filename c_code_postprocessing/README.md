## C code for post processing

This code is used to translate raw data from model output and draw boxes onto images/capture from the camera.

The `main.c` code assumes two inputs:
1. Model output: A (24x30) array
2. Original Image: An image of dimension 160 px width and 192 px height.

The code is assuming that:

1. model input is `MODEL_OUTPUT_ARRAY[24][30]` stored in `helper.h`
2. original image is `../gap9_tests.c/1.png`


## Compilation
Use the following command to compile:

```shell
gcc main.c nms.c -o main.exe
```


## File Structure
Relevant files


```shell
└─── gap9_tests/
|   |    1.png #image used as input to model
|
└─── c_code_postprocessing/
    |    main.c #contains bulk of the code for drawing onto images
    |    nms.c #contains code for Non-Maximum supression algorithmn
    |    helper.h #contains constants
    |    stb_image_wrtie.h #library used to save png file
    |    stb_image.h #library used to read png file
    |    stb_easy_font.h #library used to draw text onto image.
    |    README.md #documentation
```

## Developer Guide

Read the `main` method in `main.c` to understand the process, comments are detailed. Summary Below.


- Reads model output, `MODEL_OUTPUT_ARRAY`, which is a constant for now, from `helper.h`
- Reads captured image from, which is a file for now, from `../gap9_tests/.png`
- Transposes `MODEL_OUTPUT_ARRAY` and saves it into the variable `transposed`
- Calls `get_boxes_passing_threshold`: which reads `transposed` and assigns boxes passing the `confidence_threshold` to the `boxes` array passed into the method
- Calls `non_maximum_suppression`: which assigns `1`s to the `indices` array where the boxes are not suppressed.
- Calls `draw_bounding_boxes_for_indices`: which draws boxes onto the image where `indices[index]` is non-zero (i,e, boxes not suppressed by nms)
- Calls `saveImageAsPNG` which saves the image

## Notes:

-  Model Output is assumed to be `24x30` but in reality is `1x24x30`, so flattening must be performed.
- Facing issues drawing detection class over image.
- `Color Palette` in use is:
    ```c
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
    ```