package DP.CNN;

import DP.util.*;

import java.io.Serializable;

/**
 * Created by kingofspace on 2016/7/25.
 *
 * Layer in CNN
 *
 */
public class Layer {

    LayerType type; //input, output, conv, samp
    int outMapNum; // number of map
    double[][][][] kernel;  // front map index, out map index, x, y
    double[] bias;
    double[][][][] outMaps; // batch index, out map index, x, y
    double[][][][] errors;  // batch index, out map index, x, y

    Size mapSize;
    Size kernelSize;
    Size scaleSize; // Size of samp layer



    private static int recordInBatch = 0;  // batch index
    private static int classNum = -1;  // number of class


    public static void prepareNewBatch(){ recordInBatch = 0; }

    public static void prepareNewRecord(){ recordInBatch ++;}

    public class Size implements Serializable{
        public int x;
        public int y;
        public Size(int x, int y){
            this.x = x;
            this.y = y;
        }

        public Size divide(Size scaleSize) { return new Size(this.x/ scaleSize.x, this.y/ scaleSize.y); }

        public Size subtract(Size size, int append){ return new Size(this.x - size.x + append, this.y - size.y + append); }

    }

    public enum LayerType{ input, output, conv, samp ;}

    public static Layer buildInputLayer(Size mapSize){
        Layer layer = new Layer();
        layer.type = LayerType.input;
        layer.outMapNum =1;
        layer.setMapSize(mapSize);

        return layer;
    }

    public static Layer buildConvLayer(int outMapNum, Size kernelSize){
        Layer layer = new Layer();
        layer.type = LayerType.conv;
        layer.outMapNum = outMapNum;
        layer.kernelSize = kernelSize;

        return layer;
    }

    public static Layer buildSampLayer(Size scaleSize){
        Layer layer = new Layer();
        layer.type = LayerType.samp;
        layer.scaleSize = scaleSize;

        return layer;
    }

    public static Layer buildOutputLayer(int classNum){
        Layer layer = new Layer();
        layer.type = LayerType.output;
        layer.classNum = classNum;

        return layer;
    }

    public Size getMapSize() {
        return mapSize;
    }

    /**
     *
     *
     * @param mapSize
     */
    public void setMapSize(Size mapSize) {
        this.mapSize = mapSize;
    }

    /**
     *
     *
     * @return
     */
    public LayerType getType() {
        return type;
    }

    /**
     *
     *
     * @return
     */

    public int getOutMapNum() {
        return outMapNum;
    }

    /**
     *
     *
     * @param outMapNum
     */
    public void setOutMapNum(int outMapNum) {
        this.outMapNum = outMapNum;
    }

    /**
     *
     *
     * @return
     */
    public Size getKernelSize() {
        return kernelSize;
    }

    /**
     *
     *
     * @return
     */
    public Size getScaleSize() {
        return scaleSize;
    }




    /**
     *
     *
     * @param frontMapNum
     */
    public void initKernel(int frontMapNum) {
//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
        this.kernel = new double[frontMapNum][outMapNum][kernelSize.x][kernelSize.y];
        for (int i = 0; i < frontMapNum; i++)
            for (int j = 0; j < outMapNum; j++)
                kernel[i][j] = Util.randomMatrix(kernelSize.x, kernelSize.y, true);
    }

    /**
     *
     *
     * @param frontMapNum
     * @param size
     */
    public void initOutputKerkel(int frontMapNum, Size size) {
        kernelSize = size;
//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
        this.kernel = new double[frontMapNum][outMapNum][kernelSize.x][kernelSize.y];
        for (int i = 0; i < frontMapNum; i++)
            for (int j = 0; j < outMapNum; j++)
                kernel[i][j] = Util.randomMatrix(kernelSize.x, kernelSize.y,false);
    }

    /**
     *
     *
     * @param frontMapNum
     */
    public void initBias(int frontMapNum) {
        this.bias = Util.randomArray(outMapNum);
    }

    /**
     *
     *
     * @param batchSize
     */
    public void initOutmaps(int batchSize) {
        outMaps = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
    }

    /**
     *
     *
     * @param mapNo
     *
     * @param mapX
     *
     * @param mapY
     *
     * @param value
     */
    public void setMapValue(int mapNo, int mapX, int mapY, double value) {
        outMaps[recordInBatch][mapNo][mapX][mapY] = value;
    }

    static int count = 0;

    /**
     * ֵ
     *
     * @param mapNo
     * @param outMatrix
     */
    public void setMapValue(int mapNo, double[][] outMatrix) {
        // Log.i(type.toString());
        // Util.printMatrix(outMatrix);
        outMaps[recordInBatch][mapNo] = outMatrix;
    }

    /**
     *
     *
     *
     * @param index
     * @return
     */
    public double[][] getMap(int index) {
        return outMaps[recordInBatch][index];
    }

    /**
     *
     *
     * @param i
     *
     * @param j
     *
     * @return
     */
    public double[][] getKernel(int i, int j) {
        return kernel[i][j];
    }

    /**
     *
     *
     * @param mapNo
     * @param mapX
     * @param mapY
     * @param value
     */
    public void setError(int mapNo, int mapX, int mapY, double value) {
        errors[recordInBatch][mapNo][mapX][mapY] = value;
    }

    /**
     *
     *
     * @param mapNo
     * @param matrix
     */
    public void setError(int mapNo, double[][] matrix) {
        // Log.i(type.toString());
        // Util.printMatrix(matrix);
        errors[recordInBatch][mapNo] = matrix;
    }

    /**
     *
     *
     *
     * @param mapNo
     * @return
     */
    public double[][] getError(int mapNo) {
        return errors[recordInBatch][mapNo];
    }

    /**
     *
     *
     * @return
     */
    public double[][][][] getErrors() {
        return errors;
    }

    /**
     *
     *
     * @param batchSize
     */
    public void initErros(int batchSize) {
        errors = new double[batchSize][outMapNum][mapSize.x][mapSize.y];
    }

    /**
     *
     * @param lastMapNo
     * @param mapNo
     * @param kernel
     */
    public void setKernel(int lastMapNo, int mapNo, double[][] kernel) {
        this.kernel[lastMapNo][mapNo] = kernel;
    }

    /**
     *
     *
     * @param mapNo
     * @return
     */
    public double getBias(int mapNo) {
        return bias[mapNo];
    }

    /**
     *
     *
     * @param mapNo
     * @param value
     */
    public void setBias(int mapNo, double value) {
        bias[mapNo] = value;
    }

    /**
     *
     *
     * @return
     */

    public double[][][][] getMaps() {
        return outMaps;
    }

    /**
     *
     *
     * @param recordId
     * @param mapNo
     * @return
     */
    public double[][] getError(int recordId, int mapNo) {
        return errors[recordId][mapNo];
    }

    /**
     *
     *
     * @param recordId
     * @param mapNo
     * @return
     */
    public double[][] getMap(int recordId, int mapNo) {
        return outMaps[recordId][mapNo];
    }

    /**
     *
     *
     * @return
     */
    public int getClassNum() {
        return classNum;
    }

    /**
     *
     *
     * @return
     */
    public double[][][][] getKernel() {
        return kernel;
    }


















}
