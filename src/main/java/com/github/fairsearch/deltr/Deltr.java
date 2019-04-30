package com.github.fairsearch.deltr;

import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.github.fairsearch.deltr.models.DeltrDoc;
import com.github.fairsearch.deltr.models.DeltrTopDocs;
import com.github.fairsearch.deltr.models.TrainStep;
import com.github.fairsearch.deltr.parsers.DeltrDeserializer;
import com.google.common.primitives.Doubles;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 *  Disparate Exposure in Learning To Rank
 *  --------------------------------------
 *  A supervised learning to rank algorithm that incorporates a measure of performance and a measure
 *  of disparate exposure into its loss function. Trains a linear model based on performance and
 *  fairness for a protected group.
 *  By reducing disparate exposure for the protected group, increases the overall group visibility in
 *  the resulting rankings and thus prevents systematic biases against a protected group in the model,
 *  even though such bias might be present in the training data.
 */

@JsonDeserialize(using = DeltrDeserializer.class)
public class Deltr {

    protected static final Logger LOGGER = Logger.getLogger(Deltr.class.getName());
    @JsonProperty
    private double gamma; //gamma parameter for the cost calculation in the training phase (recommended to be around 1)

    @JsonProperty("number_of_iterations")
    private int numberOfIterations; // number of iteration in gradient descent
    @JsonProperty("learning_rate")
    private double learningRate; // learning rate in gradient descent
    @JsonProperty
    private double lambda; // regularization constant
    @JsonProperty("init_var")
    private double initVar; // range of values for initialization of weights

    @JsonProperty("standardize")
    protected boolean shouldStandardize; // boolean indicating whether the data should be standardized or not
    @JsonProperty
    protected double mu = 0; // mu for standardization
    @JsonProperty
    protected double sigma = 0; // sigma for standardization
    @JsonProperty
    protected double[] omega = null;

    @JsonIgnore
    protected List<TrainStep> log = null;

    /**
     * @param gamma gamma parameter for the cost calculation in the training phase (recommended to be around 1)
     */
    public Deltr(double gamma){
        this(gamma, false);
    }

    /**
     * @param gamma gamma parameter for the cost calculation in the training phase (recommended to be around 1)
     * @param shouldStandardize boolean indicating whether the data should be standardized or not
     */
    public Deltr(double gamma, boolean shouldStandardize){
        this(gamma, 3000, shouldStandardize);
    }

    /**
     * @param gamma gamma parameter for the cost calculation in the training phase (recommended to be around 1)
     * @param numberOfIterations number of iteration in gradient descent
     * @param shouldStandardize boolean indicating whether the data should be standardized or not
     */
    public Deltr(double gamma, int numberOfIterations, boolean shouldStandardize){
        this(gamma, numberOfIterations, 0.001f, 0.001f, 0.01f, shouldStandardize);
    }

    /**
     * @param gamma gamma parameter for the cost calculation in the training phase (recommended to be around 1)
     * @param numberOfIterations number of iteration in gradient descent
     * @param learningRate      learning rate in gradient descent
     * @param lambda            regularization constant
     * @param initVar           range of values for initialization of weights
     * @param shouldStandardize boolean indicating whether the data should be standardized or not
     */
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
     * @param gamma gamma parameter for the cost calculation in the training phase (recommended to be around 1)
     * @param numberOfIterations number of iteration in gradient descent
     * @param learningRate      learning rate in gradient descent
     * @param lambda            regularization constant
     * @param initVar           range of values for initialization of weights
     * @param shouldStandardize boolean indicating whether the data should be standardized or not
     * @param mu                set mu for standardization
     * @param sigma             set sigma for standardization
     * @param omega             set precomputed omega
     */
    public Deltr(double gamma, int numberOfIterations, double learningRate, double lambda,
                 double initVar, boolean shouldStandardize, double mu, double sigma, double[] omega){
        this(gamma, numberOfIterations, learningRate, lambda, initVar, shouldStandardize);
        this.mu = mu;
        this.sigma = sigma;
        this.omega = omega;
    }

    /**
     * Trains a DELTR model on a given training set
     * @param ranks     A list of DeltrTopDocs (query-to-documents) containing `DeltrDoc` instance implementations
     * @see             DeltrTopDocs
     * @see             DeltrDoc
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
     * @param docs         The prediction set to be (re)ranked
     * @return             Returns a new set of re-ranked documents
     * @see                DeltrTopDocs
     * @see                DeltrDoc
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

    /**
     * Returns `omega` as a vector of decimals
     * @return      An array of the double values describing omega
     */
    public double[] getOmega() {
        return omega;
    }

    /**
     * Returns the log of all steps in the training
     * @return      A list of `TrainStep` instances
     * @see         TrainStep
     */
    public List<TrainStep> getLog() {
        return this.log;
    }

    /**
     * Serializes the object to a JSON string. The `log` is not serialized.
     * @return          A string representing the object
     */
    public String toJson() {
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            return objectMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            LOGGER.severe(String.format("Exception in parsing: '%s'", e.getMessage()));
        }
        return null;
    }

    /**
     * Deseralizes a Deltr object from a JSON string.
     * @param jsonString        The JSON representation of the object
     * @return                  The created Deltr instance
     */
    public static Deltr createFromJson(String jsonString) {
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            return objectMapper.readValue(jsonString, Deltr.class);
        } catch (IOException e) {
            LOGGER.severe(String.format("IOException in parsing: '%s'", e.getMessage()));
        }
        return null ;
    }

    @Override
    public String toString() {
        return "Deltr{" +
                "gamma=" + gamma +
                ", numberOfIterations=" + numberOfIterations +
                ", learningRate=" + learningRate +
                ", lambda=" + lambda +
                ", initVar=" + initVar +
                ", shouldStandardize=" + shouldStandardize +
                ", mu=" + mu +
                ", sigma=" + sigma +
                ", omega=" + Arrays.toString(omega) +
                '}';
    }
}
