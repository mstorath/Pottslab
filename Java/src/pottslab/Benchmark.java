package pottslab;

import java.util.Random;

/**
 * Stand-alone benchmark for the original Java pottslab algorithms.
 * Prints median wall-clock times (ms) for the functions also covered by
 * the Python/Rust benchmark.py, so the two can be compared directly.
 */
public class Benchmark {

    // ── random data helpers ──────────────────────────────────────────────────

    static double[] randArray(int n, Random rng) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = rng.nextGaussian();
        return a;
    }

    static double[][] randArray2D(int n, int c, Random rng) {
        double[][] a = new double[n][c];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < c; j++) a[i][j] = rng.nextGaussian();
        return a;
    }

    static double[][] randImage(int h, int w, Random rng) {
        double[][] img = new double[h][w];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                img[i][j] = Math.max(0, Math.min(1, rng.nextGaussian() * 0.25 + 0.5));
        return img;
    }

    // ── timing ───────────────────────────────────────────────────────────────

    @FunctionalInterface
    interface Thunk { void run(); }

    static double medianMs(Thunk fn, int repeats) {
        double[] times = new double[repeats];
        for (int i = 0; i < repeats; i++) {
            long t0 = System.nanoTime();
            fn.run();
            times[i] = (System.nanoTime() - t0) / 1_000_000.0;
        }
        java.util.Arrays.sort(times);
        return times[repeats / 2];
    }

    // ── L1 Potts DP using IndexedLinkedHistogram ─────────────────────────────

    static double[] l1Potts(double[] data, double gamma) {
        int n = data.length;
        double[] b = new double[n + 1];
        for (int i = 0; i <= n; i++) b[i] = Double.POSITIVE_INFINITY;
        b[0] = -gamma;
        int[] partition = new int[n + 1];

        IndexedLinkedHistogram hist =
            new IndexedLinkedHistogram(ones(n));

        for (int rb = 0; rb < n; rb++) {
            hist.insertSorted(data[rb]);
            double[] devs = hist.computeDeviations();

            double best = Double.POSITIVE_INFINITY;
            int bestLb = 0;
            for (int lb = 0; lb <= rb; lb++) {
                double dev = (lb < devs.length) ? devs[lb] : 0.0;
                double cand = b[lb] + gamma + dev;
                if (cand < best) { best = cand; bestLb = lb; }
            }
            b[rb + 1] = best;
            partition[rb + 1] = bestLb;
        }

        // reconstruct using median of each segment
        double[] u = new double[n];
        int r = n;
        while (r > 0) {
            int l = partition[r];
            double[] seg = java.util.Arrays.copyOfRange(data, l, r);
            java.util.Arrays.sort(seg);
            double med = (seg.length % 2 == 1)
                ? seg[seg.length / 2]
                : (seg[seg.length / 2 - 1] + seg[seg.length / 2]) / 2.0;
            java.util.Arrays.fill(u, l, r, med);
            r = l;
        }
        return u;
    }

    static double[] ones(int n) {
        double[] w = new double[n]; java.util.Arrays.fill(w, 1.0); return w;
    }

    // ── table printing ───────────────────────────────────────────────────────

    static void header(String title) {
        System.out.println();
        System.out.println(title);
        System.out.println("=".repeat(title.length()));
        System.out.printf("%-12s  %-14s%n", "n", "Java (ms)");
        System.out.println("-".repeat(28));
    }

    static void row(Object n, double ms) {
        System.out.printf("%-12s  %-14.2f%n", n, ms);
    }

    // ── main ─────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        final int REPS = 5;
        Random rng = new Random(42);

        System.out.println("Java pottslab benchmark — original algorithm timings");
        System.out.println("Median of " + REPS + " runs, times in milliseconds.");

        // ── min_l2_potts 1D ──────────────────────────────────────────────────
        int[] sizes1d = {100, 500, 1_000, 5_000, 10_000, 50_000, 100_000};
        header("minL2Potts  (1-D scalar)");
        for (int n : sizes1d) {
            double[] f = randArray(n, rng);
            double gamma = 1.0;
            double ms = medianMs(() -> JavaTools.minL2Potts(f.clone(), gamma, null), REPS);
            row(n, ms);
        }

        // ── min_l2_potts vector-valued ───────────────────────────────────────
        int[] sizesVV = {100, 500, 1_000, 5_000, 10_000, 50_000};
        header("minL2Potts  (vector-valued, 3 channels)");
        for (int n : sizesVV) {
            double[][] f = randArray2D(n, 3, rng);
            double gamma = 1.0;
            double ms = medianMs(() -> JavaTools.minL2Potts(deepCopy(f), gamma, (double[]) null), REPS);
            row(n, ms);
        }

        // ── min_l1_potts 1D ──────────────────────────────────────────────────
        int[] sizesL1 = {50, 100, 200, 500, 1_000, 5_000, 10_000};
        header("minL1Potts  (1-D scalar, DP with IndexedLinkedHistogram)");
        for (int n : sizesL1) {
            double[] f = randArray(n, rng);
            double gamma = 1.0;
            double ms = medianMs(() -> l1Potts(f.clone(), gamma), REPS);
            row(n, ms);
        }

        // ── min_l2_potts_2d  (4-connected ADMM) ──────────────────────────────
        int[][] sizes2d = {{32,32},{64,64},{128,128},{256,256},{512,512}};
        header("minL2PottsADMM4  (4-connected, tol=1e-4)");
        for (int[] hw : sizes2d) {
            int h = hw[0], w = hw[1];
            double[][] img = randImage(h, w, rng);
            double gamma = 0.05;
            double muInit = gamma * 1e-2;
            double muStep = 2.0;
            double tol = 1e-4;
            double[][] weights = onesImage(h, w);
            double ms = medianMs(() -> {
                PLImage pli = new PLImage(deepCopy(img));
                JavaTools.minL2PottsADMM4(pli, gamma, weights, muInit, muStep, tol,
                                           false, true, true);
            }, 3);
            row(h + "×" + w, ms);
        }

        System.out.println();
    }

    static double[][] deepCopy(double[][] src) {
        double[][] c = new double[src.length][];
        for (int i = 0; i < src.length; i++) c[i] = src[i].clone();
        return c;
    }

    static double[][] onesImage(int h, int w) {
        double[][] wt = new double[h][w];
        for (double[] row : wt) java.util.Arrays.fill(row, 1.0);
        return wt;
    }
}
