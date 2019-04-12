package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.TrainStep;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    private double[] train(int[] queryIds, int[] protectedElementFeature,  INDArray featureMatrix, INDArray trainingScores) {
        return train(queryIds, protectedElementFeature, featureMatrix, trainingScores, false);
    }

    public double[] train(int[] queryIds, int[] protectedElementFeature, INDArray featureMatrix, INDArray trainingScores,
                       boolean storeLosses) {
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
            final INDArray predictedScores = featureMatrix.mmul(omega).reshape(numberOfElements, 1);

            //calcuate data per query predicted
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

            omega = omega.sub(grad.sum(0).mul(this.learningRate));
            omegaConverge.put(t, omega.transpose());

            costConvergeJ.putScalar(t, J.sum(0).getDouble(0));

            //add additional items in trainStep
            trainStep.setGrad(grad);
            trainStep.setOmega(omega);
            trainStep.setTotalCost(trainStep.getCost().sumNumber().doubleValue());

            this.log.add(trainStep);
        }
        return omega.data().asDouble();
    }

    private INDArray calculateGradient(INDArray featureMatrix, INDArray trainingScores, INDArray predictedScores,
                                       int[] queryIds, int[] protectedIdxs,
                                       Map<String, INDArray> dataPerQueryPredicted) {
        INDArray gradient = Nd4j.create(predictedScores.shape());
        Arrays.stream(queryIds).parallel().forEach((q) -> {
            //L2
            double l2 = 1.0 / Transforms.exp(dataPerQueryPredicted.get(keyGen(q, predictedScores)))
                    .sumNumber().doubleValue();
            //L3
            INDArray res = this.dataPerQuery.get(keyGen(q,trainingScores)).transpose()
                    .mmul(Transforms.exp(dataPerQueryPredicted.get(keyGen(q,predictedScores)))).transpose()
                    .mul(l2);
            //L1
            res.sub(this.dataPerQuery.get(keyGen(q,trainingScores)).transpose()
                    .mmul(topp(this.dataPerQuery.get(keyGen(q,trainingScores)))).transpose());

            //L deriv
            res.div(Math.log(predictedScores.length()));
            gradient.addRowVector(res);

            //TODO: add no exposure scenario
        });

        return gradient;
    }

    private TrainStep calculateCost(INDArray trainingScores, INDArray predictedScores, int[] queryIds,
                                    int[] protectedIdxs, Map<String, INDArray> dataPerQueryPredicted) {
        //the cost has to be of the same shape as the predicted/training scores
        INDArray cost = Nd4j.create(predictedScores.shape());

        Arrays.stream(queryIds).parallel().forEach((q) -> {
            cost.addiRowVector(calculateLoss(q, trainingScores, predictedScores, queryIds, protectedIdxs,
                    dataPerQueryPredicted));
        });

        double lossStandard = cost.sumNumber().doubleValue();
        double lossExposure = Arrays.stream(queryIds).parallel().mapToDouble(q ->
                exposureDiff(predictedScores, queryIds, q, protectedIdxs)).sum();

        return new TrainStep(System.currentTimeMillis(), cost, lossStandard, lossExposure);
    }

    private INDArray calculateLoss(int whichQuery, INDArray trainingScores, INDArray predictedScores,
                                   int[] queryIds, int[] protectedIdxs,
                                   Map<String, INDArray> dataPerQueryPredicted) {
        INDArray result = topp(this.dataPerQuery.get(keyGen(whichQuery, trainingScores))).transpose()
                .mmul(Transforms.log(topp(dataPerQueryPredicted.get(keyGen(whichQuery, predictedScores)))))
                .div(Math.log(predictedScores.length()))
                .mul(-1);

        if(this.noExposure) {
            result.add(exposureDiff(predictedScores, queryIds, whichQuery, protectedIdxs) *
                    exposureDiff(predictedScores, queryIds, whichQuery, protectedIdxs) * this.gamma);
        }

        return result;
    }

    private double exposureDiff(INDArray predictedScores, int[] queryIds, int whichQuery, int[] protectedIdxs) {
        ItemGroup itemGroup = findItemsPerGroupPerQuery(predictedScores, queryIds, whichQuery, protectedIdxs);

        double exposureProt = normlizedExposure(itemGroup.getProtectedItemsPerQuery(),
                itemGroup.getJudgementsPerQuery());
        double exposureNProt = normlizedExposure(itemGroup.getNonprotectedItemsPerQuery(),
                itemGroup.getJudgementsPerQuery());

        return Math.max(0, (exposureNProt - exposureProt));
    }

    private double normlizedExposure(INDArray groupData, INDArray allData) {
        return toppProt(groupData, allData).div(2).sumNumber().doubleValue() / groupData.length();
    }

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
        return data.getRows(Arrays.stream(queryIds).filter((x) -> x == whichQuery).toArray());
    }

    private static String keyGen(int q, INDArray data) {
        return String.format("%d-%d", q, data.hashCode());
    }

    public List<TrainStep> getLog() {
        return log;
    }

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


