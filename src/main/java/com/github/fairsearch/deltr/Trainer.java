package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.TrainStep;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.IntStream;

public class Trainer {

    private static final Logger LOGGER = Logger.getLogger(Trainer.class.getName());

    private double gamma; //gamma parameter for the cost calculation in the training phase (recommended to be around 1)
    private boolean noExposure;

    private int numberOfIterations; // number of iteration in gradient descent
    private double learningRate; // learning rate in gradient descent
    private double lambda; // regularization constant
    private double initVar; // initializer for the weights

    // internal training caches
    private Map<String, INDArray> dataPerQuery;
    private Map<String, INDArray> itemsPerQueryCache;
    private Map<String, ItemGroup> itemsPerGroupPerQueryCache;
    private Map<String, INDArray> normalizedToppProtDerivPerGroupDiffCache;
    private Map<String, Double> exposureDiffCache;
    private static Map<String, INDArray> toppCache;

    private List<TrainStep> log;

    public Trainer(double gamma, int numberOfIterations, double learningRate, double lambda,
                   double initVar) {
        this.gamma = gamma;
        this.numberOfIterations = numberOfIterations;
        this.learningRate = learningRate;
        this.lambda = lambda;
        this.initVar = initVar;

        cleanCache();

        this.noExposure = false;
        if(this.gamma == 0) {
            this.noExposure = true;
        }

        cleanLog();
    }

    public double[] train(int[] queryIds, int[] protectedElementFeature, INDArray featureMatrix, INDArray trainingScores) {
        int numberOfElements = featureMatrix.shape()[0]; // rows are elements
        int numberOfFeatures = featureMatrix.shape()[1]; // columns are features

        //initialize data per query
        Arrays.stream(queryIds).forEach((q) -> {
            this.dataPerQuery.put(keyGen(q, trainingScores), findItemsPerGroupPerQuery(trainingScores,
                    queryIds, q, protectedElementFeature).getJudgementsPerQuery());
            this.dataPerQuery.put(keyGen(q, featureMatrix), findItemsPerGroupPerQuery(featureMatrix,
                    queryIds, q, protectedElementFeature).getJudgementsPerQuery());
        });

        //initialize omega
        INDArray omega = Nd4j.rand(numberOfFeatures, 1).mul(this.initVar);

        INDArray costConvergeJ = Nd4j.zeros(this.numberOfIterations, 1);
        INDArray omegaConverge = Nd4j.create(this.numberOfIterations, numberOfFeatures); // create an empty array

        cleanLog();

        for(int t=0; t<this.numberOfIterations; t++){
            // log start time
            long startTime = System.currentTimeMillis();

            long stepStart = System.currentTimeMillis();

            //calculate scores
            INDArray predictedScores = featureMatrix.mmul(omega).reshape(numberOfElements, 1);

            //calculate data per query predicted
            Map<String, INDArray> dataPerQueryPredicted = new HashMap<String, INDArray>();
            Arrays.stream(queryIds).forEach((q) -> {
                dataPerQueryPredicted.put(keyGen(q, predictedScores), findItemsPerGroupPerQuery(predictedScores,
                        queryIds, q, protectedElementFeature).getJudgementsPerQuery());

            });

            LOGGER.info("pred: " + (System.currentTimeMillis() - stepStart));
            stepStart = System.currentTimeMillis();

            //get the cost/loss for all queries
            TrainStep trainStep = calculateCost(trainingScores, predictedScores, queryIds,
                    protectedElementFeature, dataPerQueryPredicted);

            LOGGER.info(" cost: " + (System.currentTimeMillis() - stepStart));
            stepStart = System.currentTimeMillis();

            INDArray J = trainStep.getCost().add(predictedScores.mul(predictedScores).mul(this.lambda));


            INDArray grad = calculateGradient(featureMatrix, trainingScores, predictedScores, queryIds,
                    protectedElementFeature, dataPerQueryPredicted);

            LOGGER.info(" grad: " + (System.currentTimeMillis() - stepStart));

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

            // log iteration
            LOGGER.info("Iteration-" + t +":" + (System.currentTimeMillis() - startTime));

        }

        cleanCache();
        return omega.data().asDouble();
    }


    private void cleanCache() {
        this.dataPerQuery = new HashMap<String, INDArray>();
        this.itemsPerQueryCache = new HashMap<String, INDArray>();
        this.itemsPerGroupPerQueryCache = new HashMap<String, ItemGroup>();
        this.normalizedToppProtDerivPerGroupDiffCache = new HashMap<String, INDArray>();
        this.exposureDiffCache = new HashMap<String, Double>();
        Trainer.toppCache = new HashMap<String, INDArray>();
    }

    private void cleanLog() {
        this.log = new ArrayList<>();
    }

    /**
     * calculates local gradients of current feature weights
     */
    private INDArray calculateGradient(INDArray trainingFeatures, INDArray trainingScores, INDArray predictedScores,
                                       int[] queryIds, int[] protectedIdxs,
                                       Map<String, INDArray> dataPerQueryPredicted) {
        INDArray gradient = Nd4j.create(trainingFeatures.shape());
        IntStream.range(0, queryIds.length).forEach((i) -> {
            //get query
            int q = queryIds[i];
            //L2
            long stepTime = System.currentTimeMillis();
            if(i % 100 == 0) {
                LOGGER.info(" grad-step-1:" + (System.currentTimeMillis()-stepTime));
                stepTime = System.currentTimeMillis();
            }
            double l2 = 1.0 / Transforms.exp(dataPerQueryPredicted.get(keyGen(q, predictedScores)))
                    .sumNumber().doubleValue();
            if(i % 100 == 0) {
                LOGGER.info(" grad-step-2:" + (System.currentTimeMillis()-stepTime));
                stepTime = System.currentTimeMillis();
            }

            //L3
            INDArray res = this.dataPerQuery.get(keyGen(q, trainingFeatures)).transpose()
                    .mmul(Transforms.exp(dataPerQueryPredicted.get(keyGen(q, predictedScores))))
                    .mul(l2);
            if(i % 100 == 0) {
                LOGGER.info(" grad-step-3:" + (System.currentTimeMillis()-stepTime));
                stepTime = System.currentTimeMillis();
            }
            //L1
            INDArray t1 = this.dataPerQuery.get(keyGen(q, trainingFeatures)).transpose();
            INDArray t2 = topp(this.dataPerQuery.get(keyGen(q, trainingScores)));
            INDArray t3 = t1.mmul(t2);
            res = res.sub(t3);

            //L deriv
            res = res.div(Math.log(predictedScores.length()));
            if(i % 100 == 0) {
                LOGGER.info(" grad-step-4:" + (System.currentTimeMillis()-stepTime));
                stepTime = System.currentTimeMillis();
            }
            if(!this.noExposure) {
                res = res.add(normalizedToppProtDerivPerGroupDiff(trainingFeatures, predictedScores, queryIds, q, protectedIdxs)
                        .mul(this.gamma)
                        .mul(2)
                        .mul(exposureDiff(predictedScores, queryIds, q, protectedIdxs)).transpose());
            }
            if(i % 100 == 0) {
                LOGGER.info(" grad-step-5:" + (System.currentTimeMillis()-stepTime));
                stepTime = System.currentTimeMillis();
            }
            gradient.putRow(i, res);
        });

        return gradient;
    }

    /**
     * calculates the difference of the normalized topp_prot derivative of the protected and non-protected groups
     */
    private INDArray normalizedToppProtDerivPerGroupDiff(INDArray trainingScores, INDArray predictedScores,
                                                       int[] queryIds, int q, int[] protectedIdxs) {
        String key = keyGen(q, queryIds, protectedIdxs, trainingScores, predictedScores);
        if(!this.normalizedToppProtDerivPerGroupDiffCache.containsKey(key)) {
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

//            return u2.sub(u3);
            this.normalizedToppProtDerivPerGroupDiffCache.put(key, u2.sub(u3));
        }
//
        return this.normalizedToppProtDerivPerGroupDiffCache.get(key);
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

//        IntStream.range(0, results.rows()).forEach((i) -> {
//        });
        for(int i=0; i<results.rows(); i++) {
            results.putRow(i, t1.sub(t2.getFloat(i,0)).div(denominator));
        }
//        results = results.div(denominator);

        return results;
    }

    private TrainStep calculateCost(INDArray trainingScores, INDArray predictedScores, int[] queryIds,
                                    int[] protectedIdxs, Map<String, INDArray> dataPerQueryPredicted) {
        //the cost has to be of the same shape as the predicted/training scores
        INDArray cost = Nd4j.create(predictedScores.shape());

        IntStream.range(0, queryIds.length).forEach((i) -> {
            INDArray loss = calculateLoss(queryIds[i], trainingScores, predictedScores,
                    queryIds, protectedIdxs, dataPerQueryPredicted);
            cost.putRow(i, loss);
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
            result = result.add(Math.pow(exposureDiff(predictedScores, queryIds, whichQuery, protectedIdxs), 2) * this.gamma);
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
        String key = keyGen(whichQuery, queryIds, protectedIdxs, data);
        if(!this.exposureDiffCache.containsKey(key)) {
            ItemGroup itemGroup = findItemsPerGroupPerQuery(data, queryIds, whichQuery, protectedIdxs);

            double exposureProt = normalizedExposure(itemGroup.getProtectedItemsPerQuery(),
                    itemGroup.getJudgementsPerQuery());
            double exposureNProt = normalizedExposure(itemGroup.getNonprotectedItemsPerQuery(),
                    itemGroup.getJudgementsPerQuery());

            this.exposureDiffCache.put(key, Math.max(0, (exposureNProt - exposureProt)));
        }
        return this.exposureDiffCache.get(key);
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
        String key = keyGen(whichQuery, queryIds, protectedIdxs, data);
        if(!this.itemsPerGroupPerQueryCache.containsKey(key)) {
            INDArray judgementsPerQuery = findItemsPerQuery(data, queryIds, whichQuery);

            List<Integer> protectedElements = new ArrayList<>();
            List<Integer> nonprotectedElements = new ArrayList<>();

            int j=0;
            for(int i=0; i<protectedIdxs.length; i++) {
                if(queryIds[i] == whichQuery) {
                    if(protectedIdxs[i] == 0) {
                        nonprotectedElements.add(j);
                    } else {
                        protectedElements.add(j);
                    }
                    j++;
                }

            }

            INDArray protectedItemsPerQuery = judgementsPerQuery.getRows(ArrayUtil.toArray(protectedElements));
            INDArray nonprotectedItemsPerQuery = judgementsPerQuery.getRows(ArrayUtil.toArray(nonprotectedElements));

            this.itemsPerGroupPerQueryCache.put(key,
                    new ItemGroup(judgementsPerQuery, protectedItemsPerQuery, nonprotectedItemsPerQuery));
        }

        return this.itemsPerGroupPerQueryCache.get(key);
    }

    private INDArray findItemsPerQuery(INDArray data, int[] queryIds, int whichQuery) {
        String key = keyGen(whichQuery, queryIds, data);
        if(!this.itemsPerQueryCache.containsKey(key))
            this.itemsPerQueryCache.put(key,
                 data.getRows(IntStream.range(0, queryIds.length).filter((x) -> queryIds[x] == whichQuery).toArray()));

        return this.itemsPerQueryCache.get(key);
    }

    private static String keyGen(INDArray data) {
        return String.format("%d", data.hashCode());
    }

    private static String keyGen(int q, INDArray data) {
        return String.format("%d-%d", q, data.hashCode());
    }

    private static String keyGen(int q, int[] qs, INDArray data) {
        return String.format("%d-%d-%d", q, qs.hashCode(), data.hashCode());
    }

    private static String keyGen(int q, int[] qs, int[] ps, INDArray data) {
        return String.format("%d-%d-%d-%d", q, qs.hashCode(), ps.hashCode(), data.hashCode());
    }
    private static String keyGen(int q, int[] qs, int[] ps, INDArray data, INDArray data2) {
        return String.format("%d-%d-%d-%d-%d", q, qs.hashCode(), ps.hashCode(), data.hashCode(), data2.hashCode());
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
        String key = keyGen(data);
        if(!Trainer.toppCache.containsKey(key)) {
            INDArray t1 = Transforms.exp(data);
            t1 = t1.div(t1.sumNumber());
            Trainer.toppCache.put(key, t1);
        }
        return Trainer.toppCache.get(key);
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


