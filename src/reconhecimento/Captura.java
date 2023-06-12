package reconhecimento;

import java.awt.event.KeyEvent;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;


public class Captura {
    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws FrameGrabber.Exception {
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();

        CascadeClassifier detectorFace = new CascadeClassifier("src\\recursos\\haarcascade_frontalface_alt.xml");

        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapturado = null;
        Mat imagemColorida = new Mat();
        

        while((frameCapturado = camera.grab()) != null){
            imagemColorida = converteMat.convert(frameCapturado);
            Mat imagemCinza = new Mat();
            org.bytedeco.javacpp.opencv_imgproc.cvtColor(imagemColorida, imagemCinza, 10);
            RectVector facesDetectadas = new RectVector();
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150,150), new Size(500,500));
            for (int i = 0; i < facesDetectadas.size(); i++){
                Rect dadosFace = facesDetectadas.get(0);
                org.bytedeco.javacpp.opencv_imgproc.rectangle(imagemColorida, dadosFace, new Scalar(0,0,255,0));
            }

            if(cFrame.isVisible()){
                cFrame.showImage(frameCapturado);
            }
        }

        cFrame.dispose();
        camera.stop();
    }
}
