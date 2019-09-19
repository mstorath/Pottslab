package pottslab;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class RunMe {
    public static void main(String[] args) throws IOException {
        BufferedImage image = ImageIO.read(new File("desert.jpg"));
        PLImage img = PLImage.fromBufferedImage(image);
        double gamma = 0.3;
        double[][]weights = new double[img.mRow][img.mCol];
        for (int y = 0; y < weights.length; y++) {
            for (int x = 0; x < weights[y].length; x++) {
                weights[y][x] = 1.0;
            }
        }
        double muInit = gamma * 1e-2;
        double muStep = 2;
        double stopTol = 1e-10;
        boolean verbose = true;
        boolean multiThreaded = true;
        boolean useADMM = true;
        double[] omega = new double[]{Math.sqrt(2.0)-1.0, 1.0-Math.sqrt(2.0)/2.0};
        PLImage img2 = JavaTools.minL2PottsADMM8(img, gamma, weights, muInit, muStep, stopTol,  verbose, multiThreaded, useADMM, omega);
        
        ImageIO.write(img2.toBufferedImage(), "png", new File("output.png"));
    }
}
