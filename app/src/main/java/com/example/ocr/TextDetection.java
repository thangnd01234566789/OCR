package com.example.ocr;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import androidx.annotation.NonNull;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.sql.Array;
import java.util.ArrayList;
import java.util.List;

public class TextDetection {
    private String model_path;
    private Interpreter tflite_Interpreter;
    private Activity activity;
    private int inputSize;
    private final static float IMAGE_MEAN = 128;
    private final static float IMAGE_STD = 128f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 1.0f;

    private MappedByteBuffer tfliteModel;
    private int imageSizeX;
    private int imageSizeY;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer_score;
    private TensorBuffer outputProbabilityBuffer_geometry;
    private TensorProcessor probabilityProcessor;


    public TextDetection(String model_path, Activity activity, int inputSize){
        this.model_path = model_path;
        this.activity = activity;
        this.inputSize = inputSize;
    }

    private MappedByteBuffer loadModel(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(this.model_path);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void init() throws IOException{
        loadModel();
    }

    private void loadModel() throws IOException {
        tfliteModel = FileUtil.loadMappedFile(activity, model_path);
        // Create a TFLITE interpreter instance
        tflite_Interpreter = new Interpreter(tfliteModel);

        // Read type and shape of input and output tensors, respectively
        int imageTensorIndex = 0;
        int[] imageShape = tflite_Interpreter.getInputTensor(imageTensorIndex).shape();
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];

        DataType imageDataType = tflite_Interpreter.getInputTensor(imageTensorIndex).dataType();

        // Hava 2 Out put Score And Geometry
        int probabilityTensorIndex_1 = 0;
        int probabilityTensorIndex_2 = 1;

        int[] probabilityShape_1 = tflite_Interpreter.getOutputTensor(probabilityTensorIndex_1).shape();//{1, num_class}
        int[] probabilityShape_2 = tflite_Interpreter.getOutputTensor(probabilityTensorIndex_2).shape();

        DataType probabilityDataType_1 = tflite_Interpreter.getOutputTensor(probabilityTensorIndex_1).dataType();
        DataType probabilityDataType_2 = tflite_Interpreter.getOutputTensor(probabilityTensorIndex_2).dataType();
        // Creates the input tensor
        inputImageBuffer = new TensorImage(imageDataType);

        // Creates The Output Tensor And Its Processor
//        outputProbabilityBuffer_score = TensorBuffer.createFixedSize(probabilityShape_1, probabilityDataType_1);
//        outputProbabilityBuffer_geometry = TensorBuffer.createFixedSize(probabilityShape_2, probabilityDataType_2);

        outputProbabilityBuffer_score = TensorBuffer.createFixedSize(new int[]{1, 1, 80 , 80 }, probabilityDataType_1);
        outputProbabilityBuffer_geometry = TensorBuffer.createFixedSize(new int[]{1, 5, 80, 80}, probabilityDataType_2);

        probabilityProcessor = new TensorProcessor.Builder().add(getPOstprocessNormalizeOp()).build();

    }

    private TensorOperator getPOstprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    public String detections(Bitmap bitmap, int sensorOrientation){
        inputImageBuffer = loadImage(bitmap, sensorOrientation);

        tflite_Interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer_score.getBuffer().rewind());
        tflite_Interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer_geometry.getBuffer().rewind());

        probabilityProcessor.process(outputProbabilityBuffer_score);
        probabilityProcessor.process(outputProbabilityBuffer_geometry);

        int[] geometry = outputProbabilityBuffer_geometry.getIntArray();
        int[] score = outputProbabilityBuffer_score.getIntArray();
//        int geometry = outputProbabilityBuffer_geometry.getIntArray();
        return "";
    }

    private void box_extractor(@NonNull int[] score, @NonNull int[] score_shape,@NonNull int[] geometry, @NonNull int[] geometry_shape,@NonNull float min_confidence){
        int num_rows = score_shape[2];
        int num_cols = score_shape[3];
        ArrayList<List> rectangles = new ArrayList<List>();
        ArrayList<List> confidences = new ArrayList<List>();

        for (int b = 0;  b < num_cols; b = b + 1){
            
        }
//        Math.sin()
    }

    private TensorImage loadImage(Bitmap bitmap, int senorOrientation){
        inputImageBuffer.load(bitmap);
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRoration = senorOrientation / 90;
        // Define an ImageProcessor From TFlite to do preprocessing
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize,cropSize))
                .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(new Rot90Op(numRoration))
                .add(getPOstprocessNormalizeOp())
                .build();

        return imageProcessor.process(inputImageBuffer);
    }
}

(1, 2, 3, 2)
        [[[[    10      2]
        [     3      4]
        [     5      6]]

        [[111110 222222]
        [333333 444444]
        [555555 666666]]]]
        [    10      2      3      4      5      6 111110 222222 333333 444444
        555555 666666]
