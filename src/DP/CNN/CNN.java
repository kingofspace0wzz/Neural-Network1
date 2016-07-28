package DP.CNN;

import java.util.List;
import DP.CNN.Layer;
import DP.CNN.LogisticRegression;
import DP.CNN.ConcurrenceRunner.TaskManager;
import DP.util.Util;
/**
 * Created by kingofspace on 2016/7/26.
 *
 *
 * Convolutional Neural Network with Layers
 *
 *
 */
public class CNN {

    List<Layer> layers;
    public int layerNum;   // number of layers
    int batchSize;
    private double lambda = 0;
    private double alpha = 0.85;

    private Util.Operator multiply_lambda = (Util.Operator) (value) -> { return value * (1 - alpha * lambda);};
    private Util.Operator multiply_alpha = (Util.Operator)(value) -> {return value * alpha;};
    private Util.Operator divide_batchSize = (Util.Operator)(value) -> {return value / this.batchSize;};

    public CNN(LayerBuilder layerBuilder, int batchSize){
        layers = layerBuilder.mLayers;
        layerNum = layers.size();
        this.batchSize = batchSize;
        setup(batchSize);

    }

    private void updateKernels(final Layer layer, final Layer lastLayer, boolean onLine){
        int mapNum = layer.getOutMapNum();
        int lastMapNum = lastLayer.getOutMapNum();
        new TaskManager(mapNum){


            @Override
            public void process(int start, int end){
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < lastMapNum; j++) {
                        double[][] deltaKernel = null;
                        for (int k = 0; k < batchSize; k++) {
                            double[][] error = layer.getError(k, i);
                            if (deltaKernel == null){
                                deltaKernel = Util.conValid(lastLayer.getMap(k, j), error);
                            }else{
                                deltaKernel = Util.matrixOp(Util.conValid(lastLayer.getMap(k, j), error), deltaKernel, null, null, Util.plus);
                            }
                            if (onLine == true){
                                double[][] kernel = layer.getKernel(i, j);
                                kernel = Util.matrixOp(kernel, deltaKernel, multiply_lambda, multiply_alpha, Util.plus);
                                layer.setKernel(j, i, kernel);
                            }
                        }

                        deltaKernel = Util.matrixOp(deltaKernel, divide_batchSize);
                        double[][] kernel = layer.getKernel(j, i);
                        kernel = Util.matrixOp(kernel, deltaKernel, multiply_lambda, multiply_alpha, Util.plus);
                        layer.setKernel(j, i, kernel);
                    }
                }
            }
        }.start();
    }

    private void updateBias(final Layer layer, final Layer lastLayer){
        int mapNum = layer.getOutMapNum();

    }

    private void setHiddenErrors(){
        for (int i = this.layerNum - 2; i > 0; i++) {
            Layer layer = layers.get(i);
            Layer nextLayer = layers.get(i + 1);
            switch (layer.getType()){

                case conv:
                    setConvErrors(layer, nextLayer);
                    break;
                case samp:
                    setSampErrors(layer, nextLayer);
                    break;
                    default:
                        break;
            }
        }
    }

    private void setConvErrors(final Layer layer,final Layer nextLayer){
        int mapNum = layer.getOutMapNum();
        new TaskManager(mapNum) {

            @Override
            public void process(int start, int end) {
                for (int m = start; m < end; m++) {
                    Layer.Size scale = nextLayer.getScaleSize();
                    double[][] nextError = nextLayer.getError(m);
                    double[][] map = layer.getMap(m);

                    double[][] outMatrix = Util.matrixOp(map,
                            Util.cloneMatrix(map), null, Util.one_value,
                            Util.multiply);
                    outMatrix = Util.matrixOp(outMatrix,
                            Util.kronecker(nextError, scale), null, null,
                            Util.multiply);
                    layer.setError(m, outMatrix);
                }

            }

        }.start();

    }

    private void setSampErrors(final Layer layer, final Layer nextLayer){
        int mapNum = layer.getOutMapNum();
        int nextMapNum = nextLayer.getOutMapNum();
        new TaskManager(mapNum) {

            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    double[][] sum = null;
                    for (int j = 0; j < nextMapNum; j++) {
                        double[][] nextError = nextLayer.getError(j);
                        double[][] kernel = nextLayer.getKernel(i, j);

                        if (sum == null)
                            sum = Util
                                    .conFull(nextError, Util.rot180(kernel));
                        else
                            sum = Util.matrixOp(
                                    Util.conFull(nextError,
                                            Util.rot180(kernel)), sum, null,
                                    null, Util.plus);
                    }
                    layer.setError(i, sum);
                }
            }

        }.start();
    }

    private void setup(int batchSize) {
        Layer inputLayer = layers.get(0);
        inputLayer.initOutmaps(batchSize);
        for (int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            Layer frontLayer = layers.get(i-1);
            int frontMapNum = frontLayer.getOutMapNum();
            switch (layer.getType()){
                case input:
                    break;
                case conv:
                    layer.setMapSize(frontLayer.getMapSize().subtract(layer.getMapSize(), 1));

                    layer.initKernel(frontLayer.outMapNum);

                    layer.initBias(frontLayer.outMapNum);

                    layer.initOutmaps(batchSize);
                case samp:
                    layer.setOutMapNum(frontMapNum);

                    layer.setMapSize(frontLayer.getMapSize().divide(layer.scaleSize));

                    layer.initErros(batchSize);

                    layer.initOutmaps(batchSize);
                case output:
                    layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize());

                    layer.initBias(frontMapNum);

                    layer.initErros(batchSize);

                    layer.initOutmaps(batchSize);
                    break;
            }
        }

    }


    static class LayerBuilder{
        List<Layer> mLayers;

        public LayerBuilder(){

        }

        public LayerBuilder(Layer layer){
            mLayers.add(layer);
        }

        public LayerBuilder addLayer(Layer layer){
            this.mLayers.add(layer);
            return this;
        }

    }






}
