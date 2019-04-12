package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.DeltrDoc;
import com.github.fairsearch.deltr.models.DeltrTopDocs;
import com.github.fairsearch.deltr.models.TrainStep;
import com.google.common.primitives.Doubles;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    private double[] omega = null;
    private List<TrainStep> log = null;

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

    public Deltr(double gamma, int numberOfIterations){
        this.gamma = gamma;

        this.numberOfIterations = numberOfIterations;
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
     * @param ranks list of DeltrTopDocs (query -> documents) containing `com.github.fairsearch.deltr.models.DeltrDoc`
     *              instance implementations
     */
    public void train(List<DeltrTopDocs> ranks) {
        // create the trainer
        Trainer trainer = new Trainer(this.gamma, this.numberOfIterations, this.learningRate, this.lambda, this.initVar);

        //parse the data for training
        TrainerData trainerData = null;
        for(DeltrTopDocs docs : ranks) {
            TrainerData current = prepareData(docs);
            if(trainerData == null) {
                trainerData = current;
            } else {
                trainerData.append(current);
            }
        }

        //TODO: Check if storeLosses is needed
        this.omega = trainer.train(trainerData.queryIds, trainerData.protectedElementFeature,
                trainerData.featureMatrix, trainerData.trainingScores, true);

        this.log = trainer.getLog();
    }

    /**
     * Uses the trained DELTR model to rank the prediction set
     * @param docs          the prediction set to be (re)ranked
     * @return
     */
    public DeltrTopDocs rank(DeltrTopDocs docs) {
        //check if the model is created
        if(this.omega == null) {
            throw new NullPointerException("You need to train a model first!");
        }

        //re-calculate the judgement for each document
        for(DeltrDoc doc : docs.docs()) {
            double dotProduct = 0;
            for(int i=0; i<doc.size(); i++) {
                dotProduct += doc.feature(i) * this.omega[i];
            }
            doc.rejudge(dotProduct);
        }

        //re-order the docs
        docs.docs().sort((o1, o2) -> {
            if(o1.judgement() < o2.judgement())
                return 1;
            else if(o1.judgement() > o2.judgement())
                return -1;
            return 0;
        });

        return docs;
    }

    private TrainerData prepareData(DeltrTopDocs docs) {
        TrainerData result = new TrainerData();

        //create the array of query ids
        result.queryIds = new int[docs.size()];
        Arrays.fill(result.queryIds, docs.id());

        //initialize the protected element feature and feature matrix
        result.protectedElementFeature = new int[docs.size()];
        result.featureMatrix = Nd4j.create(docs.size(), docs.doc(0).size());

        //initialize this only if it's a training set
        result.trainingScores = Nd4j.create(docs.size(), 1);

        for(int i=0; i<docs.size(); i++) {
            DeltrDoc doc = docs.doc(i);
            //assign the protected element feature and feature matrix
            result.protectedElementFeature[i] = doc.isProtected() ? 1 : 0;
            result.featureMatrix.putRow(i, Nd4j.create(Doubles.toArray(doc.features())));

            //add the training judgement for this document
            result.trainingScores.putScalar(i, doc.judgement());
        }
        return result;
    }

    private static class TrainerData {
        private int[] queryIds;
        private int[] protectedElementFeature;
        private INDArray featureMatrix;
        private INDArray trainingScores;

        private TrainerData append(TrainerData data) {
            //copy query data
            int[] tmp = queryIds;
            queryIds = new int[tmp.length + data.queryIds.length];
            System.arraycopy(data, 0, queryIds, tmp.length, data.queryIds.length);

            //copy protected feature data
            tmp = protectedElementFeature;
            protectedElementFeature = new int[tmp.length + data.protectedElementFeature.length];
            System.arraycopy(data, 0, protectedElementFeature, tmp.length, data.protectedElementFeature.length);

            //copy feature matrix and training scores (if any)
            for(int i=0; i<data.featureMatrix.rows(); i++) {
                featureMatrix.addRowVector(data.featureMatrix.getRow(i));
                if(trainingScores != null && data.trainingScores != null)
                    trainingScores.addRowVector(data.trainingScores.getRow(i));
            }
            return this;
        }
    }

    public double[] getOmega() {
        return omega;
    }

    public List<TrainStep> getLog() {
        return this.log;
    }
}
