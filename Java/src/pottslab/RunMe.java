package pottslab;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class RunMe {
    public static void main(String[] args) throws IOException {
        if(args.length < 3) {
            System.err.println("USAGE: java -jar pottslab.jar <input> <output.png> <gamma>");
            System.exit(1);
        }
        PLImage img = PLImage.fromBufferedImage(ImageIO.read(new File(args[0])));
        
        double gamma = Double.valueOf(args[2]);
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
        
        ImageIO.write(img2.toBufferedImage(), "png", new File(args[1]));
    }
}
