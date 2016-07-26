package DP.CNN;

import DP.util.Util;

import java.util.Random;

/**
 * Created by kingofspace on 2016/7/26.
 */
public class LogisticRegression {

    int N;  // number of samples
    int n_in; // number of input
    int n_out; // number of classes

    public double[][] W;   // weights
    public double[] b; // bias


    public LogisticRegression(int N, int n_in, int n_out ){
        this.N = N;
        this.n_in = n_in;
        this.n_out = n_out;
        this.W = new double[n_in][n_out];
        this.b = new double[n_out];

    }

    // Softmax classifier
    public void softmax(double[] y){
        double max = 0.;
        double sum = 0;
        for (int i = 0; i < n_out; i++) {
            if ( max <= y[i] ) max = y[i];
        }

        for (int i = 0; i < n_out; i++) {
            y[i] = Math.exp(y[i] - max);
            sum += y[i];
        }

        for (int i = 0; i < n_out; i++) y[i] /= sum;

    }

    /**
     *
     * @param x
     * @param y
     * @param lr    // Learning Rate
     * @param ol    // On-line trainning
     * @return
     */
    public double[] train(double[] x, double[] y, double lr, boolean ol){
        double[] p_y_given_x = new double[n_out];
        double[] dy = new double[n_out];
        Random rdm = new Random();



        for (int i = 0; i < n_out; i++) {
            p_y_given_x[i] = 0;
            for (int j = 0; j < n_in; j++) {
                p_y_given_x[i] += W[i][j] * x[j];
            }
            p_y_given_x[i] += b[i];
        }

        softmax(p_y_given_x);
        if (ol == true){
            for (int i = 0; i < n_out; i++) {
                dy[i] = y[i] - p_y_given_x[i];

                for (int j = 0; j < n_in; j++) {
                    W[i][j] += ( lr * dy[i] * x[j] + (rdm.nextDouble() * (1.- 0.5) + 0.5) * W[i][j] );  // (rdm.nextDouble() * (1.- 0.5) + 0.5) is momentum between 0.5-1.0
                }

                b[i] += ( lr * dy[i] + (rdm.nextDouble() * (1.- 0.5) + 0.5) * b[i] );

            }
        }else {
            for (int i = 0; i < n_out; i++) {
                dy[i] = y[i] - p_y_given_x[i];

                for (int j = 0; j < n_in; j++) {
                    W[i][j] += lr * dy[i] * x[j] / N;
                }

                b[i] += lr * dy[i] / N;

            }
        }
        return dy;
    }

    // Predict the outcome
    public void predict(double[] x, double[] y){
        for (int i = 0; i < n_out; i++) {
            y[i] = 0.;
            for (int j = 0; j < n_in; j++) {
                y[i] += W[i][j] * x[j];
            }
            y[i] += b[i];

        }

        softmax(y);

    }




}
