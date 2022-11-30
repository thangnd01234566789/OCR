package com.example.ocrdeploy;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private final String CV_LOG = "OPENCV_LOADER";
    private TextDetection textDetection;
    private static final int CAMERA_REQUEST = 100;
    private Uri imgUri;
    private Bitmap imgBitmap;
    private Button button_1 ;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = (ImageView) findViewById(R.id.imageView);
        button_1 =(Button) findViewById(R.id.button1);
        button_1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openImage();
            }
        });
        try {
            initDetectionsModel();
            Log.i("LOAD MODEL", "Load Model Successfuly !");
        } catch (IOException e) {
            Log.d("LOAD MODEL", "Can't Load Model !");
            e.printStackTrace();
        }
    }

    private BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            super.onManagerConnected(status);
            switch (status){
                case LoaderCallbackInterface.SUCCESS:{
                    Log.i(CV_LOG, "Opencv Loader !");
                }break;
                default:{
                    super.onManagerConnected(status);
                }break;
            }
        }
    };

    //    Init Model
    private void initDetectionsModel() throws IOException {
        textDetection = new TextDetection("lite-model_east-text-detector_fp16_1.tflite","lite-model_keras-ocr_float16_2.tflite", this, 320);
        textDetection.init();
    }
    // Open Image
    private void openImage(){
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, CAMERA_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK && data != null){
            imgUri = data.getData();
            try {
                imgBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imgUri);
            }catch (IOException ex){
                ex.printStackTrace();
            }
        }

        Bitmap predict = textDetection.detections(imgBitmap, 0);
        imageView.setImageBitmap(predict);
//        imageView.setImageBitmap(imgBitmap);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            Log.i(CV_LOG, "Initial Succesfully !");
            mLoaderCallBack.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
        else {
            Log.d(CV_LOG, "Opencv Not Found !" );
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this, mLoaderCallBack);
        }
    }
}