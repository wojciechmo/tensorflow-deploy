import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.SavedModelBundle;

import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Mat;

import java.util.*;
import java.io.*;

public class Main
{
    static public Tensor mat2tensor(Mat img)
    {
        Imgproc.resize(img, img, new Size(224,224));
        // Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);
        img.convertTo(img, CvType.CV_32F, 1/255.0);

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

        Tensor t = Tensor.create(X);

        return t;
    }

    public static String[] process_classes_names(String filepath)
    {
        String[] classes = new String[1001];
        classes[0]="none";
        String line = null;

        try
        {
            FileReader fileReader = new FileReader(filepath);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            while((line = bufferedReader.readLine()) != null)
            {
                String[] parts = line.split(":");
                int idx = Integer.parseInt(parts[0].substring(1, parts[0].length()));
                String name = parts[1].substring(2, parts[1].length()-2);

                // in imagenet_names.txt there are 1000 valid classes
                // while model predicts also 'zero' class
                classes[idx+1] = name.toLowerCase();
            }

            bufferedReader.close();
        }
        catch(FileNotFoundException ex)
            {System.out.println("Unable to open file '" + filepath + "'");}

        catch(IOException ex)
            {System.out.println("Error reading file '" + filepath + "'");}

        return classes;
    }

    public static void main(String[] args)
    {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);

        Mat img = Imgcodecs.imread("/tmp/python/tiger.jpg");
        Tensor t = mat2tensor(img);

        // load model saved with tf.saved_model.simple_save function
        // for model saved with tf.train.write_graph function use ReadBinaryProto from JavaCpp package
        SavedModelBundle smb = SavedModelBundle.load("/tmp/python/saved_model", "serve");
        Session session = smb.session();

        List<Tensor<?>> outputs = session.runner().feed("x", t).fetch("predictions:1").fetch("probabilities:0").run();

        Tensor top_k_tensor = outputs.get(0);
        Tensor probs_tensor = outputs.get(1);

        int[][] top_k_data = new int[1][5];
        top_k_tensor.copyTo(top_k_data);

        float[][] probs_data = new float[1][1001];
        probs_tensor.copyTo(probs_data);

        String[] classes = process_classes_names("/tmp/python/imagenet_names.txt");

        System.out.println("Top 5 predicitions:");
        for(int i=0; i<5; i++)
        {
            System.out.print("--name: " + classes[top_k_data[0][i]] + " --prob: " + Float.toString((probs_data[0][top_k_data[0][i]])));
            System.out.println();
        }

    }
}
