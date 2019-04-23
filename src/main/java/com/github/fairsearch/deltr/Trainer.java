package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.TrainStep;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class Trainer {

    private double gamma; //gamma parameter for the cost calculation in the training phase (recommended to be around 1)
    private boolean noExposure;

    private int numberOfIterations; // number of iteration in gradient descent
    private double learningRate; // learning rate in gradient descent
    private double lambda; // regularization constant
    private double initVar; // initializer for the weights

    private Map<String, INDArray> dataPerQuery;

    private List<TrainStep> log;

    public Trainer(double gamma, int numberOfIterations, double learningRate, double lambda,
                   double initVar) {
        this.gamma = gamma;
        this.numberOfIterations = numberOfIterations;
        this.learningRate = learningRate;
        this.lambda = lambda;
        this.initVar = initVar;

        this.dataPerQuery = new HashMap<String, INDArray>();

        this.noExposure = false;
        if(this.gamma == 0) {
            this.noExposure = true;
        }

        this.log = new ArrayList<>();
    }

    public double[] train(int[] queryIds, int[] protectedElementFeature, INDArray featureMatrix, INDArray trainingScores) {
        int numberOfElements = featureMatrix.shape()[0]; // rows are elements
        int numberOfFeatures = featureMatrix.shape()[1]; // columns are features

        //initialize data per query
        Arrays.stream(queryIds).parallel().forEach((q) -> {
            this.dataPerQuery.put(keyGen(q, trainingScores), findItemsPerGroupPerQuery(trainingScores,
                    queryIds, q, protectedElementFeature).getJudgementsPerQuery());
            this.dataPerQuery.put(keyGen(q, featureMatrix), findItemsPerGroupPerQuery(featureMatrix,
                    queryIds, q, protectedElementFeature).getJudgementsPerQuery());
        });

        //initialize omega
        INDArray omega = Nd4j.rand(numberOfFeatures, 1).mul(this.initVar);

        INDArray costConvergeJ = Nd4j.zeros(this.numberOfIterations, 1);
        INDArray omegaConverge = Nd4j.create(this.numberOfIterations, numberOfFeatures); // create an empty array

        for(int t=0; t<this.numberOfIterations; t++){
            //calculate scores
            INDArray predictedScores = featureMatrix.mmul(omega).reshape(numberOfElements, 1);

            //calculate data per query predicted
            Map<String, INDArray> dataPerQueryPredicted = new HashMap<String, INDArray>();
            Arrays.stream(queryIds).parallel().forEach((q) -> {
                dataPerQueryPredicted.put(keyGen(q, predictedScores), findItemsPerGroupPerQuery(predictedScores,
                        queryIds, q, protectedElementFeature).getJudgementsPerQuery());

            });

            //get the cost/loss for all queries
            TrainStep trainStep = calculateCost(trainingScores, predictedScores, queryIds,
                    protectedElementFeature, dataPerQueryPredicted);

            INDArray J = trainStep.getCost().add(predictedScores.mul(predictedScores).mul(this.lambda));

            INDArray grad = calculateGradient(featureMatrix, trainingScores, predictedScores, queryIds,
                    protectedElementFeature, dataPerQueryPredicted);

            //add additional items in trainStep
            trainStep.setOmega(omega);
            trainStep.setGrad(grad);
            trainStep.setTotalCost(trainStep.getCost().sumNumber().doubleValue());

            //recalculate omega
            omega = omega.sub(grad.sum(0).mul(this.learningRate));
            omega = omega.reshape(numberOfFeatures, 1);
            omegaConverge.putRow(t, omega.transpose());

            costConvergeJ.putScalar(t, J.sum(0).getDouble(0));

            // add trainStep to log
            this.log.add(trainStep);
        }
        return omega.data().asDouble();
    }

    /**
     * calculates local gradients of current feature weights
     */
    private INDArray calculateGradient(INDArray trainingFeatures, INDArray trainingScores, INDArray predictedScores,
                                       int[] queryIds, int[] protectedIdxs,
                                       Map<String, INDArray> dataPerQueryPredicted) {
        INDArray gradient = Nd4j.create(trainingFeatures.shape());
        AtomicInteger atomicInteger = new AtomicInteger(0);
        Arrays.stream(queryIds).parallel().forEachOrdered((q) -> {
            //L2
            double l2 = 1.0 / Transforms.exp(dataPerQueryPredicted.get(keyGen(q, predictedScores)))
                    .sumNumber().doubleValue();
            //L3
            INDArray res = this.dataPerQuery.get(keyGen(q, trainingFeatures)).transpose()
                    .mmul(Transforms.exp(dataPerQueryPredicted.get(keyGen(q, predictedScores))))
                    .mul(l2);
            //L1
            INDArray t1 = this.dataPerQuery.get(keyGen(q, trainingFeatures)).transpose();
            INDArray t2 = topp(this.dataPerQuery.get(keyGen(q, trainingScores)));
            INDArray t3 = t1.mmul(t2);
            res = res.sub(t3);

            //L deriv
            res = res.div(Math.log(predictedScores.length()));

            if(!this.noExposure) {
                res = res.add(normalizedToppProtDerivPerGroupDiff(trainingFeatures, predictedScores, queryIds, q, protectedIdxs)
                        .mul(this.gamma)
                        .mul(2)
                        .mul(exposureDiff(predictedScores, queryIds, q, protectedIdxs)).transpose());
            }

            gradient.putRow(atomicInteger.getAndIncrement(), res);
        });

        return gradient;
    }

    /**
     * calculates the difference of the normalized topp_prot derivative of the protected and non-protected groups
     */
    private INDArray normalizedToppProtDerivPerGroupDiff(INDArray trainingScores, INDArray predictedScores,
                                                       int[] queryIds, int q, int[] protectedIdxs) {
        ItemGroup trainGroup = findItemsPerGroupPerQuery(trainingScores, queryIds, q, protectedIdxs);
        ItemGroup predictionsGroup = findItemsPerGroupPerQuery(predictedScores, queryIds, q, protectedIdxs);

        INDArray u2 = normalizedToppProtDerivPerGroup(trainGroup.getNonprotectedItemsPerQuery(),
                trainGroup.getJudgementsPerQuery(),
                predictionsGroup.getNonprotectedItemsPerQuery(),
                predictionsGroup.getJudgementsPerQuery());
        INDArray u3 = normalizedToppProtDerivPerGroup(trainGroup.getProtectedItemsPerQuery(),
                trainGroup.getJudgementsPerQuery(),
                predictionsGroup.getProtectedItemsPerQuery(),
                predictionsGroup.getJudgementsPerQuery());

        return u2.sub(u3);
    }

    /**
     * normalizes the results of the derivative of topp prot
     */
    private INDArray normalizedToppProtDerivPerGroup(INDArray groupFeatures, INDArray allFeatures,
                                                     INDArray groupPredictions, INDArray allPredictions) {
        INDArray derivative = toppProtFirstDerivative(groupFeatures, allFeatures, groupPredictions, allPredictions);

        return derivative.div(Math.log(2)).sum(0).div(groupPredictions.length());
    }

    /**
     * Derivative for topp prot in pieces
     */
    private INDArray toppProtFirstDerivative(INDArray groupFeatures, INDArray allFeatures,
                                             INDArray groupPredictions, INDArray allPredictions) {
        INDArray numerator1 = Transforms.exp(groupPredictions).transpose().mmul(groupFeatures);
        double numerator2 = Transforms.exp(allPredictions).sumNumber().doubleValue();
        double numerator3 = Transforms.exp(allPredictions)
                .transpose()
                .mmul(allFeatures)
                .sumNumber().doubleValue();
        double denominator = Math.pow(Transforms.exp(allPredictions).sumNumber().doubleValue(), 2);

        INDArray t1 = numerator1.mul(numerator2);
        INDArray t2 = Transforms.exp(groupPredictions).mul(numerator3);

        INDArray results = Nd4j.create(groupFeatures.shape());

        for(int i=0; i<results.rows(); i++) {
            results.putRow(i, t1.sub(t2.getFloat(i,0)));
        }
        results = results.div(denominator);

        return results;
    }

    private TrainStep calculateCost(INDArray trainingScores, INDArray predictedScores, int[] queryIds,
                                    int[] protectedIdxs, Map<String, INDArray> dataPerQueryPredicted) {
        //the cost has to be of the same shape as the predicted/training scores
        INDArray cost = Nd4j.create(predictedScores.shape());

        AtomicInteger atomicInteger = new AtomicInteger(0);

        Arrays.stream(queryIds).parallel().forEachOrdered((q) -> {
            cost.putRow(atomicInteger.getAndIncrement(),
                    calculateLoss(q, trainingScores, predictedScores, queryIds, protectedIdxs, dataPerQueryPredicted));
        });

        double lossStandard = cost.sumNumber().doubleValue();
        double lossExposure = Arrays.stream(queryIds).parallel().mapToDouble(q ->
                exposureDiff(predictedScores, queryIds, q, protectedIdxs)).sum();

        return new TrainStep(System.currentTimeMillis(), cost, lossStandard, lossExposure);
    }

    /**
     * Calculate loss for a given query
     */
    private INDArray calculateLoss(int whichQuery, INDArray trainingScores, INDArray predictedScores,
                                   int[] queryIds, int[] protectedIdxs,
                                   Map<String, INDArray> dataPerQueryPredicted) {
        INDArray result = topp(this.dataPerQuery.get(keyGen(whichQuery, trainingScores))).transpose()
                .mmul(Transforms.log(topp(dataPerQueryPredicted.get(keyGen(whichQuery, predictedScores)))))
                .div(Math.log(predictedScores.length()))
                .mul(-1);

        if(!this.noExposure) {
            result.addi(Math.pow(exposureDiff(predictedScores, queryIds, whichQuery, protectedIdxs), 2) * this.gamma);
        }

        return result;
    }

    /**
     * computes the exposure difference between protected and non-protected groups
     * @param data              predictions
     * @param queryIds          list of query IDs
     * @param whichQuery        given query ID
     * @param protectedIdxs     list states which item is protected or non-protected
     * @return
     */
    private double exposureDiff(INDArray data, int[] queryIds, int whichQuery, int[] protectedIdxs) {
        ItemGroup itemGroup = findItemsPerGroupPerQuery(data, queryIds, whichQuery, protectedIdxs);

        double exposureProt = normalizedExposure(itemGroup.getProtectedItemsPerQuery(),
                itemGroup.getJudgementsPerQuery());
        double exposureNProt = normalizedExposure(itemGroup.getNonprotectedItemsPerQuery(),
                itemGroup.getJudgementsPerQuery());

        return Math.max(0, (exposureNProt - exposureProt));
    }

    /**
     * calculates the exposure of a group in the entire ranking
     * @param groupData         redictions of relevance scores for one group
     * @param allData           all predictions
     * @return
     */
    private double normalizedExposure(INDArray groupData, INDArray allData) {
        return toppProt(groupData, allData).div(Math.log(2)).sumNumber().doubleValue() / groupData.length();
    }

    /**
     * given a dataset of features what is the probability of being at the top position
     * for one group (group_items) out of all items
     * example: what is the probability of each female (or male respectively) item (group_items) to be in position 1
     * @param groupData     vector of predicted scores of one group (protected or non-protected)
     * @param allData       vector of predicted scores of all items
     * @return
     */
    private INDArray toppProt(INDArray groupData, INDArray allData) {
        return Transforms.exp(groupData).div(Transforms.exp(allData).sumNumber());
    }


    private ItemGroup findItemsPerGroupPerQuery(INDArray data, int[] queryIds, int whichQuery,
                                               int[] protectedIdxs) {
        INDArray judgementsPerQuery = findItemsPerQuery(data, queryIds, whichQuery);
        
        double[] vals = new double[protectedIdxs.length];
        List<Integer> protectedElements = new ArrayList<>();
        List<Integer> nonprotectedElements = new ArrayList<>();
        for(int i=0; i<protectedIdxs.length; i++) {
            vals[i] = (double)protectedIdxs[i];
            if(protectedIdxs[i] == 0) {
                nonprotectedElements.add(i);
            } else {
                protectedElements.add(i);
            }
        }
//        INDArray protFeaturePerQuery = findItemsPerQuery(Nd4j.create(vals), queryIds, whichQuery);

        INDArray protectedItemsPerQuery = judgementsPerQuery.getRows(ArrayUtil.toArray(protectedElements));
        INDArray nonprotectedItemsPerQuery = judgementsPerQuery.getRows(ArrayUtil.toArray(nonprotectedElements));

        return new ItemGroup(judgementsPerQuery, protectedItemsPerQuery, nonprotectedItemsPerQuery);
    }

    private INDArray findItemsPerQuery(INDArray data, int[] queryIds, int whichQuery) {
        return data.getRows(IntStream.range(0, queryIds.length).filter((x) -> queryIds[x] == whichQuery).toArray());
    }

    private static String keyGen(int q, INDArray data) {
        return String.format("%d-%d", q, data.hashCode());
    }

    public List<TrainStep> getLog() {
        return log;
    }

    /**
     * computes the probability of a document being
     * in the first position of the ranking
     * @param data  all training judgments or all predictions
     * @return      float value which is a probability
     */
    private static INDArray topp(INDArray data) {
        return Transforms.exp(data).div(Transforms.exp(data).sumNumber());
    }

    private static class ItemGroup {
        private INDArray judgementsPerQuery;
        private INDArray protectedItemsPerQuery;
        private INDArray nonprotectedItemsPerQuery;

        private int code;

        public ItemGroup(INDArray judgementsPerQuery, INDArray protectedItemsPerQuery, INDArray nonprotectedItemsPerQuery) {
            this.judgementsPerQuery = judgementsPerQuery;
            this.protectedItemsPerQuery = protectedItemsPerQuery;
            this.nonprotectedItemsPerQuery = nonprotectedItemsPerQuery;
        }

        public INDArray getJudgementsPerQuery() {
            return judgementsPerQuery;
        }

        public INDArray getProtectedItemsPerQuery() {
            return protectedItemsPerQuery;
        }

        public INDArray getNonprotectedItemsPerQuery() {
            return nonprotectedItemsPerQuery;
        }

        @Override
        public int hashCode() {
            final int prime = 23;
            int result = 1;
            result = prime * result + (judgementsPerQuery == null ? 0 : judgementsPerQuery.hashCode());
            result = prime * result + (protectedItemsPerQuery == null ? 0 : protectedItemsPerQuery.hashCode());
            result = prime * result + (nonprotectedItemsPerQuery == null ? 0 : nonprotectedItemsPerQuery.hashCode());
            return result;
        }
    }
}


