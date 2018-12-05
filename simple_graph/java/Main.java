import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.SavedModelBundle;

public class Main
{
    public static void main(String[] args)
    {
        // load model saved with tf.saved_model.simple_save function
        // for model saved with tf.train.write_graph function use ReadBinaryProto from JavaCpp package
        SavedModelBundle smb = SavedModelBundle.load("/tmp/python/model_java", "serve");
        Session session = smb.session();

        float[] X = new float[]{2.0f, 2.0f, 2.0f, 2.0f};
        System.out.println("Input vector:");
        for(int i=0; i<4; i++)
            System.out.print(X[i] + " ");
        System.out.println();

        Tensor tx = Tensor.create(X);
        Tensor output = session.runner().feed("x", tx).fetch("y").run().get(0);

        float[] dst = new float[4];
        output.copyTo(dst);
        System.out.println("Output vector:");
        for(int i=0; i<4; i++)
            System.out.print(dst[i] + " ");
        System.out.println();
    }
}
