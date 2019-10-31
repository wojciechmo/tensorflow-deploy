package com.example.app;

import android.app.Activity;
import android.hardware.Camera;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.android.Utils;
import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.FileOutputStream;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.text.DateFormat;
import java.io.File;

import android.os.Environment;
import android.widget.PopupWindow;
import android.view.LayoutInflater;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.view.MotionEvent;

public class MainActivity extends Activity implements SurfaceHolder.Callback {

    private Camera mCamera;
    private SurfaceView surfaceView;
    private SurfaceHolder surfaceHolder;
    private Button capture_image;
    private TextView text_view;
    private Interpreter tflite;
    private String[] imagenet_labels;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (OpenCVLoader.initDebug()) {
            Log.e("Libs", "OpenCV loaded successfully");
            //Toast.makeText(getApplicationContext(),"OpenCV loaded", Toast.LENGTH_SHORT).show();
        }
        else {
            Log.e("Libs", "Error while loading OpenCV");
        }

        try {
            tflite = new Interpreter(loadModelFile());
        }
        catch(Exception ex) {
            ex.printStackTrace();
        }

        imagenet_labels = process_classes_names();
        
        capture_image = (Button) findViewById(R.id.capture_image);
        capture_image.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                capture();
            }
        });

        surfaceView = (SurfaceView) findViewById(R.id.surfaceview);
        surfaceHolder = surfaceView.getHolder();
        surfaceHolder.addCallback(MainActivity.this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        try {
            mCamera = Camera.open();
            mCamera.setDisplayOrientation(90);
            mCamera.setPreviewDisplay(surfaceHolder);
            mCamera.startPreview();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void capture(){

        mCamera.takePicture(null, null, null, new Camera.PictureCallback() {

            @Override
            public void onPictureTaken(byte[] data, Camera camera) {

                Camera.Parameters params = camera.getParameters();
                int fw = params.getPictureSize().width;
                int fh = params.getPictureSize().height;
                Mat mat = new Mat(new Size(fw, fh), CvType.CV_8U);
                mat.put(0, 0, data);
                Mat img = Imgcodecs.imdecode(mat, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);

                float[][][][] X = mat2array(img);
                int[][] Y = new int[1][5];
                tflite.run(X,Y);

                String out_text = "";
                for (int i=0; i<5; i++)
                {
                    String classname = imagenet_labels[Y[0][i]];
                    System.out.println(classname);
                    out_text += classname;
                    if (i<4) out_text += '\n';
                }

                onButtonShowPopupWindowClick(out_text);
                mCamera.startPreview();
                //save_opencv_image(img);
            }
        });
    }

    private float[][][][] mat2array(Mat img){

        int w = img.cols();
        int h = img.rows();

        Rect roi;
        if (w<h) roi = new Rect(0, (int)((h-w)/2), w, w);
        else roi = new Rect((int)((w-h)/2), 0, h, h);
        img = new Mat(img, roi);

        Imgproc.resize(img, img, new Size(224,224));
        img.convertTo(img, CvType.CV_32F, 1/255.0);
        Core.rotate(img, img, Core.ROTATE_90_CLOCKWISE);//assume portrait orientation of a phone

        float[][][][] X = new float[1][224][224][3];

        for(int i=0; i<img.height(); i++)
            for(int j=0; j<img.width(); j++)
            {
                double[] data = img.get(i, j);

                float blue = (float)data[0];
                float green = (float)data[1];
                float red = (float)data[2];

                X[0][i][j][0] = red;
                X[0][i][j][1] = green;
                X[0][i][j][2] = blue;
            }

        return X;
    }

    private String[] process_classes_names(){

        String[] classes = new String[1001];
        int idx = 0;
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(this.getAssets().open("ImageNetLabels.txt")));
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                classes[idx]=mLine;
                idx = idx+1;
            }
        } catch (IOException e) {
        } finally {
            if (reader != null) {
                try {reader.close();} catch (IOException e) {}
            }
        }

        return classes;
    }

    public void onButtonShowPopupWindowClick(String text) {

        LayoutInflater inflater = (LayoutInflater) getSystemService(LAYOUT_INFLATER_SERVICE);
        View popupView = inflater.inflate(R.layout.popup, null);

        int width = LinearLayout.LayoutParams.WRAP_CONTENT;
        int height = LinearLayout.LayoutParams.WRAP_CONTENT;
        boolean focusable = true;
        final PopupWindow popupWindow = new PopupWindow(popupView, width, height, focusable);

        text_view = (TextView) popupView.findViewById(R.id.popup_text_view);
        text_view.setText(text);
        View view = findViewById(R.id.rl);
        popupWindow.showAtLocation(view, Gravity.CENTER, 0, 350);

        popupView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                popupWindow.dismiss();
                return true;
            }
        });
    }

    private void save_opencv_image(Mat img){

        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);

        Bitmap bmp = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bmp);

        FileOutputStream out = null;

        Date date = new Date();
        DateFormat df = new SimpleDateFormat("hh-mm-ss");
        String filename = df.format(date) + ".jpg";

        File sd = new File(Environment.getExternalStorageDirectory() + "/DCIM/tmp");
        boolean success = true;
        if (!sd.exists()) {
            success = sd.mkdir();
        }
        if (success) {
            File dest = new File(sd, filename);
            try {
                out = new FileOutputStream(dest);
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out);
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try {
                    if (out != null) {
                        out.close();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {

        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.e("Surface Changed", "format   ==   " + format + ",   width  ===  " + width + ", height   ===    " + height);
        try {
            mCamera.setPreviewDisplay(holder);
            mCamera.startPreview();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        Log.e("Surface Created", "");
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.e("Surface Destroyed", "");
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mCamera != null) {
            mCamera.stopPreview();
            mCamera.release();
        }
    }
}
