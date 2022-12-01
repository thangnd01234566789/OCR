package com.example.ocrdeploy;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import androidx.annotation.NonNull;

import com.example.ocrdeploy.ml.LiteModelKerasOcrDr2;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
//import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class TextRecognition {
    private String model_path;
    private Interpreter tflite_Interpreter;
    private Activity activity;
    private final static float IMAGE_MEAN = 128;
    private final static float IMAGE_STD = 128f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 1.0f;

    private MappedByteBuffer tfliteModel;
    private int imageSizeX;
    private int imageSizeY;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;
    private LiteModelKerasOcrDr2 model;
    private static final String alphabets = "0123456789abcdefghijklmnopqrstuvwxyz";
    ByteBuffer outputByteBuffer;


    public TextRecognition(String model_path, Activity activity){
        this.model_path = model_path;
        this.activity = activity;
    }

    public void init() throws IOException {
        imageSizeX = 31;
        imageSizeY = 200;
        loadModel();
    }

    private void loadModel() throws IOException {
        tfliteModel = FileUtil.loadMappedFile(activity, model_path);
        // Create a TFLITE interpreter instance
        boolean useGPU = true;

        tflite_Interpreter = new Interpreter(tfliteModel);

        // Read type and shape of input and output tensors, respectively
        int imageTensorIndex = 0;
        int[] imageShape = tflite_Interpreter.getInputTensor(imageTensorIndex).shape();
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];

        DataType imageDataType = tflite_Interpreter.getInputTensor(imageTensorIndex).dataType();

        // Hava 2 Out put Score And Geometry
        int probabilityTensorIndex_1 = 0;

        int[] probabilityShape_1 = tflite_Interpreter.getOutputTensor(probabilityTensorIndex_1).shape();

        DataType probabilityDataType_1 = tflite_Interpreter.getOutputTensor(probabilityTensorIndex_1).dataType();
        // Creates the input tensor
        inputImageBuffer = new TensorImage(imageDataType);


//        outputProbabilityBuffer =
        // Creates The Output Tensor And Its Processor
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape_1, DataType.UINT8);
        outputByteBuffer = ByteBuffer.allocateDirect(8 * probabilityShape_1[1]);

        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

//        try {
//            model = LiteModelKerasOcrDr2.newInstance(activity);
//            inputImageBuffer = new TensorImage(DataType.FLOAT32);
//
//            // Creates inputs for reference.
//        } catch (IOException e) {
//            // TODO Handle the exception
//        }
//        # char_list:   'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    }

    private MappedByteBuffer getModel(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(model_path);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    public String recognitions(Bitmap bitmap, int sensorOrientation){

        Mat matImg = new Mat(bitmap.getWidth(), bitmap.getHeight(), CvType.CV_8UC1);
        Utils.bitmapToMat(bitmap, matImg);
        Imgproc.cvtColor(matImg, matImg, Imgproc.COLOR_RGB2GRAY);
        Utils.matToBitmap(matImg, bitmap);

        inputImageBuffer = loadImage(bitmap, sensorOrientation);
        tflite_Interpreter.run(inputImageBuffer.getBuffer(),outputByteBuffer);

        String prediction = "";
        for (int x = 0; x < 48; x = x + 1){
            int index = outputByteBuffer.get(x * 8);
            if (0 <= index && index <= (alphabets.length() - 1)){
                prediction += alphabets.toCharArray()[index];
            }
        }

        return prediction;
    }

    private ArrayList<List> resize_box(@NonNull float[] box, @NonNull int [] box_shape){
        ArrayList<List> box_reshape = new ArrayList<List>();
        for (int x = 1; x <= box_shape[1]; x = x + 1){
            int count = 0;
            ArrayList<List> box_reshape_sub1 = new ArrayList<List>();
            for (int y = 1; y <= box_shape[2]; y = y + 1){
                ArrayList<Float> box_sub2 = new ArrayList<Float>();
                for (int z = 1 ; z <= box_shape[3]; z = z + 1){
                    box_sub2.add(box[x - 1 + box_shape[1] * count]);
                    count++;
                }
                box_reshape_sub1.add(box_sub2);
            }
            box_reshape.add(box_reshape_sub1);
        }
        return box_reshape;
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
                .add(getPostprocessNormalizeOp())
                .add(new TransformToGrayscaleOp())
                .build();

        return imageProcessor.process(inputImageBuffer);
    }
}
