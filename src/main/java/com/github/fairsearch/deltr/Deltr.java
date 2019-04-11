package com.github.fairsearch.deltr;

import ciir.umass.edu.learning.RANKER_TYPE;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.learning.RankerFactory;
import ciir.umass.edu.learning.RankerTrainer;
import org.apache.lucene.search.TopDocs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * This class serves as a wrapper around the utilities we have created for FA*IR ranking
 */
public class Deltr {

    private static final Logger LOGGER = Logger.getLogger(Deltr.class.getName());

    private double gamma; //gamma parameter for the cost calculation in the training phase (recommended to be around 1)

    private int numberOfIterations; // number of iteration in gradient descent
    private double learningRate; // learning rate in gradient descent
    private double lambda; // regularization constant
    private double initVar; // range of values for initialization of weights

    private boolean standardize; // boolean indicating whether the data should be standardized or not

    /**
     *  Disparate Exposure in Learning To Rank
     *  --------------------------------------
     *
     *  A supervised learning to rank algorithm that incorporates a measure of performance and a measure
     *  of disparate exposure into its loss function. Trains a linear model based on performance and
     *  fairness for a protected group.
     *  By reducing disparate exposure for the protected group, increases the overall group visibility in
     *  the resulting rankings and thus prevents systematic biases against a protected group in the model,
     *  even though such bias might be present in the training data.
     * @param gamma gamma parameter for the cost calculation in the training phase (recommended to be around 1)
     */
    public Deltr(double gamma){
        this.gamma = gamma;

        this.numberOfIterations = 3000;
        this.learningRate = 0.001f;
        this.lambda = 0.001f;
        this.initVar = 0.01f;
        this.standardize = false;
    }

    public Deltr(double gamma, int numberOfIterations, double learningRate, double lambda,
                 double initVar, boolean standardize){
        this.gamma = gamma;
        this.numberOfIterations = numberOfIterations;
        this.learningRate = learningRate;
        this.lambda = lambda;
        this.initVar = initVar;
        this.standardize = standardize;
    }

    /**
     *  Trains a DELTR model on a given training set
     * @param ranks list of TopDocs (query -> documents) containing `com.github.fairsearch.deltr.DeltrDoc`
     *             i.e. higher scores are better
     */
    public void train(List<TopDocs> ranks) {
        //
    }

    public TopDocs rank(TopDocs docs) {
        TopDocs result = null;

        return result;
    }

    public static void main(String[] args) {

    }


}
