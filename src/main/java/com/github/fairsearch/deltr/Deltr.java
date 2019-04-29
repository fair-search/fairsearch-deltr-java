package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.DeltrDoc;
import com.github.fairsearch.deltr.models.DeltrTopDocs;
import com.github.fairsearch.deltr.models.TrainStep;
import com.google.common.primitives.Doubles;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * This class serves as a wrapper around the utilities we have created for FA*IR ranking
 */
public class Deltr implements Serializable {

    protected static final Logger LOGGER = Logger.getLogger(Deltr.class.getName());

    private double gamma; //gamma parameter for the cost calculation in the training phase (recommended to be around 1)

    private int numberOfIterations; // number of iteration in gradient descent
    private double learningRate; // learning rate in gradient descent
    private double lambda; // regularization constant
    private double initVar; // range of values for initialization of weights

    protected boolean shouldStandardize; // boolean indicating whether the data should be standardized or not
    protected double mu = 0; // mu for standardization
    protected double sigma = 0; // sigma for standardization

    protected double[] omega = null;
    protected List<TrainStep> log = null;

    /**
     *  Disparate Exposure in Learning To Rank
     *  --------------------------------------
     *  A supervised learning to rank algorithm that incorporates a measure of performance and a measure
     *  of disparate exposure into its loss function. Trains a linear model based on performance and
     *  fairness for a protected group.
     *  By reducing disparate exposure for the protected group, increases the overall group visibility in
     *  the resulting rankings and thus prevents systematic biases against a protected group in the model,
     *  even though such bias might be present in the training data.
     *
     * @param gamma gamma parameter for the cost calculation in the training phase (recommended to be around 1)
     */
    public Deltr(double gamma){
        this(gamma, false);
    }

    public Deltr(double gamma, boolean shouldStandardize){
        this(gamma, 3000, shouldStandardize);
    }

    public Deltr(double gamma, int numberOfIterations, boolean shouldStandardize){
        this(gamma, numberOfIterations, 0.001f, 0.001f, 0.01f, shouldStandardize);
    }

    public Deltr(double gamma, int numberOfIterations, double learningRate, double lambda,
                 double initVar, boolean shouldStandardize){
        this.gamma = gamma;
        this.numberOfIterations = numberOfIterations;
        this.learningRate = learningRate;
        this.lambda = lambda;
        this.initVar = initVar;
        this.shouldStandardize = shouldStandardize;
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

        // standardize data if required
        if(this.shouldStandardize) {
            this.mu = trainerData.featureMatrix.meanNumber().doubleValue();
            this.sigma = trainerData.featureMatrix.stdNumber().doubleValue();
            trainerData.featureMatrix = trainerData.featureMatrix.sub(this.mu).div(this.sigma);
            trainerData.featureMatrix.putColumn(trainerData.protectedElementFeatureIndex,
                                Nd4j.create(IntStream.of(trainerData.protectedElementFeature)
                                        .mapToDouble((x) -> (double) x).toArray()));
        }

        this.omega = trainer.train(trainerData.queryIds, trainerData.protectedElementFeature,
                trainerData.featureMatrix, trainerData.trainingScores);

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

        // standardize data if required
        if(this.shouldStandardize) {
            for(int i=0; i<docs.size(); i++) {
                DeltrDoc doc = docs.doc(i);
                for(String key : doc.keys()) {
                    if(!key.equals(doc.protectedFeatureName()))
                        doc.put(key, (doc.feature(key) - this.mu)/this.sigma);
                }
            }
        }

        //re-calculate the judgement for each document
        for(int j=0; j<docs.size(); j++) {
            DeltrDoc doc = docs.doc(j);
            double dotProduct = 0;
            for(int i=0; i<doc.size(); i++) {
                dotProduct += doc.feature(i) * this.omega[i];
            }
            doc.rejudge(dotProduct);
        }

        //re-order the docs
        docs.reorder();

        return docs;
    }

    private TrainerData prepareData(DeltrTopDocs docs) {
        TrainerData result = new TrainerData();

        //create the array of query ids
        result.queryIds = new int[docs.size()];
        Arrays.fill(result.queryIds, docs.id());

        //initialize the protected element assignFeature and assignFeature matrix
        result.protectedElementFeature = new int[docs.size()];
        result.featureMatrix = Nd4j.create(docs.size(), docs.doc(0).size());

        //initialize this only if it's a training set
        result.trainingScores = Nd4j.create(docs.size(), 1);

        //find the protected element assignFeature in the assignFeature list
        result.protectedElementFeatureIndex = docs.doc(0).protectedFeatureIndex();

        for(int i=0; i<docs.size(); i++) {
            DeltrDoc doc = docs.doc(i);
            //assign the protected element assignFeature and assignFeature matrix
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
        private int protectedElementFeatureIndex;

        private TrainerData append(TrainerData data) {
            //copy query data
            int[] tmp = queryIds;
            queryIds = new int[tmp.length + data.queryIds.length];
            System.arraycopy(tmp, 0, queryIds, 0, tmp.length);
            System.arraycopy(data.queryIds, 0, queryIds, tmp.length, data.queryIds.length);

            //copy protected assignFeature data
            tmp = protectedElementFeature;
            protectedElementFeature = new int[tmp.length + data.protectedElementFeature.length];
            System.arraycopy(tmp, 0, protectedElementFeature, 0, tmp.length);
            System.arraycopy(data.protectedElementFeature, 0, protectedElementFeature,
                    tmp.length, data.protectedElementFeature.length);

            //create new feature matrix
            INDArray newFeatureMatrix = Nd4j.create(featureMatrix.rows() + data.featureMatrix.rows(),
                    featureMatrix.columns());
            INDArray newTrainingScores = Nd4j.create(trainingScores.rows() + data.trainingScores.rows(),
                    trainingScores.columns());


            //copy assignFeature matrix and training scores (if any)
            for(int i=0; i< featureMatrix.rows(); i++) {
                newFeatureMatrix.putRow(i, featureMatrix.getRow(i));
                newTrainingScores.putRow(i, trainingScores.getRow(i));
            }
            for(int i=0; i< data.featureMatrix.rows(); i++) {
                newFeatureMatrix.putRow(i + featureMatrix.rows(), data.featureMatrix.getRow(i));
                newTrainingScores.putRow(i + featureMatrix.rows(), data.trainingScores.getRow(i));
            }

            featureMatrix = newFeatureMatrix;
            trainingScores = newTrainingScores;

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
