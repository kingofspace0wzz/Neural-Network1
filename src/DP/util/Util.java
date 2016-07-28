package DP.util;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;



import DP.CNN.Layer.Size;

/**
 * Created by kingofspace on 2016/7/25.
 */
public class Util {
    /**
     *
     * @author kingofspace
     */
    public interface Operator extends Serializable{
        public double process(double value);
    }

    // Return 1-value
    public static final Operator one_value = (value) -> {return 1-value;};

    public static final Operator sigmoid =  (value) -> {return 1/(1+Math.pow(Math.E, -value));};


    /**
     *
     */
    public interface OperatorOnTwo extends Serializable{
        public double process(double valueA, double valueB);
    }

    public static final OperatorOnTwo multiply = (valueA, valueB) -> {return valueA * valueB;};

    public static final OperatorOnTwo divide = (valueA, valueB) -> {return valueA / valueB;};

    public static final OperatorOnTwo plus = (valueA, valueB) -> {return valueA + valueB;};

    public static final OperatorOnTwo minus = (valueA, valueB) -> {return valueA - valueB;};



    // Print the Matrix elements
    public static void printMatrix(double[][] matrix){
        for (int i = 0; i < matrix.length; i++) {
            String line = Arrays.toString(matrix[i]);
            line = line.replaceAll(", ", "\t" );
            System.out.println(line);
        }
        System.out.println();
    }

    public static double[][] transpose(double[][] matrix){
        double[][] resultMatrix = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                resultMatrix[j][i] = matrix[i][j];
            }
        }
        return matrix;
    }

    // Rotate the matrix by 180 degree
    public static double[][] rot180(double[][] matrix){
        int n = matrix.length; // Rows
        int m = matrix[0].length; // Cols
        double[][] resultMatrix = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                resultMatrix[i][j] = matrix[n-i+1][m-j+1];
            }
        }
        return resultMatrix;
    }

    public static double[][] rot90(double[][] matrix){
        int n = matrix.length;
        int m = matrix[0].length;
        double[][] resultMatrix = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                resultMatrix[i][j] = matrix[j][n-i+1];
            }
        }
        return resultMatrix;
    }
    // Construct a random matrix
    public static double[][] randomMatrix(int r, int c, boolean b){
        double[][] matrix = new double[r][c];
        Random rdm = new Random();
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                matrix[i][j] = (rdm.nextDouble()-0.05) / 10;
            }
        }
        return matrix;
    }

    public static double[] randomArray(int len){
        double[] a = new double[len];
        Random rdm = new Random();
        for (int i = 0; i < len; i++) {
            a[i] = (rdm.nextDouble()-0.05) / 10;
        }

        return a;
    }



    public static int getNumRow(double[][] matrix){
        return matrix.length;
    }

    public static double[] getRows(double[][] matrix, int index){
        return matrix[index];
    }

    public static int getNumCol(double[][] matrix){
        return matrix[0].length;
    }

    public static double[] getCols(double[][] matrix, int index){
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = matrix[i][index];
        }
        return result;
    }

    public static double[][] matrixOp(double[][] ma, Operator op){
        final int n = ma.length;
        final int m = ma[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                ma[i][j] = op.process(ma[i][j]);
            }
        }
        return ma;
    }

    public static double[][] matrixOp(double[][] ma, double[][] mb, Operator opa, Operator opb, OperatorOnTwo opTwo){
        final int n1 = ma.length;
        final int m1 = ma[0].length;
        final int n2 = ma.length;
        final int m2 = ma[0].length;
        if (m1 != m2 || n1 != n2){
            throw new RuntimeException("Unequal length, ma.length:" + ma.length + "\n" + "mb.length" + mb.length);
        }
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < m1; j++) {
                double a = ma[i][j];
                if (opa != null) a = opa.process(a);
                double b = mb[i][j];
                if (opb != null) b = opb.process(b);
                mb[i][j] = opTwo.process(a, b);
            }
        }
        return mb;
    }

    public static double[][] matrixMultiply(double[][] ma, double[][] mb){
        final int n1 = ma.length;
        final int m1 = ma[0].length;
        final int n2 = ma.length;
        final int m2 = ma[0].length;
        double[][] resultMatrix = new double[n1][m2];
        if (m1 != n2) throw new RuntimeException("Cols of matrixA != Rows of matrixB");
        if (n1 <= 50 || n2 <= 50) {
            double[][] resultC = new double[n1][m2];
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < m1; j++) {
                    for (int k = 0; k < n2; k++) {
                        resultC[i][j] = ma[i][k] + mb[k][j];
                    }
                }
            }
            resultMatrix = resultC;
        }// Updating for Strassen


        return resultMatrix;
    }

    public static double[][] cloneMatrix(final double[][] matrix) {

        final int m = matrix.length;
        int n = matrix[0].length;
        final double[][] outMatrix = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                outMatrix[i][j] = matrix[i][j];
            }
        }
        return outMatrix;
    }

    /**
     * Gaussian generator ,   Updating for Generic type of mean and variance
     * @param mean
     * @param variance
     * @return
     */
    public static double gaussian(double mean, double variance){
        Random rdm = new Random();
        return (rdm.nextGaussian() * Math.sqrt(variance)) + mean;
    }

    // Normal Gaussian
    public static double gaussian(){
        Random rdm = new Random();
        return rdm.nextGaussian();
    }
    /**
     * Uniform samples generator
     * @param min
     * @param max
     * @return
     */
    public static double uniform(double min, double max){
        Random rdm = new Random();
        return rdm.nextDouble() * (max - min) + min;
    }

    /**
     * binormal samples generator
     * @param n     # of samples
     * @param p     probability
     * @return
     */
    public static int binormal(int n, double p){
        if(p < 0 || p > 0) throw new RuntimeException(p + " is not appropriate");

        Random rdm = new Random();
        int c = 0;
        for (int i = 0; i < n; i++) {
            if (rdm.nextDouble() <= p) {
                c ++;
            }
        }
        return c;
    }

    public static double sigmoid(double x) { return 1./(1.+Math.pow(Math.E, -x)); }

    public static double dsigmoid(double x) { return x * (1. - x); }

    public static double tanh(double x) { return Math.tanh(x); }

    public static double dtanh(double x) { return 1.- x * x; }

    public static double[][] kronecker(final double[][] matrix, final Size scale) {
        final int n = matrix.length;
        int m = matrix[0].length;
        final double[][] outMatrix = new double[n * scale.x][m * scale.y];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
                    for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++) {
                        outMatrix[ki][kj] = matrix[i][j];
                    }
                }
            }
        }
        return outMatrix;
    }

    public static double[][] scaleMatrix(final double [][] matrix, final Size scale){
        final int n = matrix.length;
        int m = matrix[0].length;
        int sn = n / scale.x;
        int sm = m / scale.y;
        final double[][] outMatrix = new double[sn][sm];
        int size = scale.x * scale.y;
        for (int i = 0; i < sn; i++) {
            for (int j = 0; j < sm; j++) {
                int sum = 0;
                for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
                    for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++) {
                        sum += matrix[si][sj];
                    }
                }
                outMatrix[i][j] = sum / size;
            }
        }

        return outMatrix;
    }

    // 2-D convolution  ,
    // Valid: return part of the convolution that do not have zero-edges
    public static double[][] conValid(final double[][] matrix, double[][] kernel){
        int n = matrix.length;
        int m = matrix[0].length;
        int kn = kernel.length;
        int km = kernel[0].length;
        int kns = n - kn +1;
        int kms = m - km +1;
        double[][] result = new double[kns][kms];
        for (int ki = 0; ki < kns; ki++) {
            for (int kj = 0; kj < kms; kj++) {
                int sum = 0;
                for (int i = 0; i < kn; i++) {
                    for (int j = 0; j <km; j++) {
                        sum += matrix[ki+i][kj+j] * kernel[ki][kj];
                    }
                }
                result[ki][kj] = sum;
            }
        }

        return result;
    }


    // 3-D convolution
    // Valid : return  part of the convolution that do not have zero-edges
    public static double[][] conValid(final double[][][][] matrix,
                                        int mapNoX, double[][][][] kernel, int mapNoY) {
        int m = matrix.length;
        int n = matrix[0][mapNoX].length;
        int h = matrix[0][mapNoX][0].length;
        int km = kernel.length;
        int kn = kernel[0][mapNoY].length;
        int kh = kernel[0][mapNoY][0].length;
        int kms = m - km + 1;
        int kns = n - kn + 1;
        int khs = h - kh + 1;
        final double[][][] outMatrix = new double[kms][kns][khs];
        for (int i = 0; i < kms; i++) {
            for (int j = 0; j < kns; j++)
                for (int k = 0; k < khs; k++) {
                    double sum = 0.0;
                    for (int ki = 0; ki < km; ki++) {
                        for (int kj = 0; kj < kn; kj++)
                            for (int kk = 0; kk < kh; kk++) {
                                sum += matrix[i + ki][mapNoX][j + kj][k + kk]
                                        * kernel[ki][mapNoY][kj][kk];
                            }
                    }
                    outMatrix[i][j][k] = sum;
                }
        }
        return outMatrix[0];
     }

    // 2-D convolution
    // Full: return full convolution
    public static double[][] conFull(final double[][] matrix, double[][] kernel){
        int n = matrix.length;
        int m = matrix[0].length;
        int kn = kernel.length;
        int km = kernel[0].length;
        int kns = n + kn - 1;
        int kms = m + km - 1;
        double[][] result = new double[kns][kms];
        fill(result, 0.);
        for (int ki = 0; ki < kns; ki++) {
            for (int kj = 0; kj < kms; kj++) {
                int sum = 0;
                for (int i = 0; i < kn; i++) {
                    for (int j = 0; j <km; j++) {

                        if (ki + i <= n && kj + j <= m) sum += matrix[ki + i][kj + j] * kernel[ki][kj];

                    }
                }
                result[ki][kj] = sum;
            }
        }

        return result;
    }

    // 3-D convolution
    // Full : return  the full convolution
    public static double[][] conFull(final double[][][][] matrix,
                                        int mapNoX, double[][][][] kernel, int mapNoY) {
        int m = matrix.length;
        int n = matrix[0][mapNoX].length;
        int h = matrix[0][mapNoX][0].length;
        int km = kernel.length;
        int kn = kernel[0][mapNoY].length;
        int kh = kernel[0][mapNoY][0].length;
        int kms = m + km - 1;
        int kns = n + kn - 1;
        int khs = h + kh - 1;
        final double[][][] outMatrix = new double[kms][kns][khs];
        for (int i = 0; i < kms; i++) {
            for (int j = 0; j < kns; j++)
                for (int k = 0; k < khs; k++) {
                    double sum = 0.0;
                    for (int ki = 0; ki < km; ki++) {
                        for (int kj = 0; kj < kn; kj++)
                            for (int kk = 0; kk < kh; kk++) {
                                if ( i + ki <= m && j + kj <= n && k + kk <= h ) sum += matrix[i + ki][mapNoX][j + kj][k + kk]
                                        * kernel[ki][mapNoY][kj][kk];
                            }
                    }
                    outMatrix[i][j][k] = sum;
                }
        }
        return outMatrix[0];
    }

    public static double sum(double[][] error) {
        int m = error.length;
        int n = error[0].length;
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum += error[i][j];
            }
        }
        return sum;
    }

    public static double[][] sum(double[][][][] errors, int j) {
        int m = errors[0][j].length;
        int n = errors[0][j][0].length;
        double[][] result = new double[m][n];
        for (int mi = 0; mi < m; mi++) {
            for (int nj = 0; nj < n; nj++) {
                double sum = 0;
                for (int i = 0; i < errors.length; i++)
                    sum += errors[i][j][mi][nj];
                result[mi][nj] = sum;
            }
        }
        return result;
    }

    public static void fill(double[][] matrix ,double value){
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = value;
            }
        }
    }

    public static int binaryArray2int(double[] array) {
        int[] d = new int[array.length];
        for (int i = 0; i < d.length; i++) {
            if (array[i] >= 0.500000001)
                d[i] = 1;
            else
                d[i] = 0;
        }
        String s = Arrays.toString(d);
        String binary = s.substring(1, s.length() - 1).replace(", ", "");
        int data = Integer.parseInt(binary, 2);
        return data;

    }



}
