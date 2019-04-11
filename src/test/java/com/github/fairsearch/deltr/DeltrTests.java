package com.github.fairsearch.deltr;

import ciir.umass.edu.features.FeatureManager;
import ciir.umass.edu.learning.RANKER_TYPE;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.learning.RankerTrainer;
import ciir.umass.edu.metric.METRIC;
import ciir.umass.edu.metric.MetricScorerFactory;
import junitparams.JUnitParamsRunner;
import junitparams.Parameters;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@RunWith(JUnitParamsRunner.class)
public class DeltrTests {

    private static final double OFFSET = 0.0001; // result tolerance for DeltrTests

    @Test
    public void testNd4j() {
        int nRows = 3;
        int nColumns = 5;
        INDArray allRands = Nd4j.rand(1, nColumns);
        INDArray allOnes = Nd4j.ones(nColumns, 1);
        INDArray allZeros = Nd4j.zeros(nRows, nColumns);
        System.out.println(allRands.mmul(allOnes));
//        System.out.println(allOnes.mul(2));
//        System.out.println(allRands.mmul(allOnes));
//        System.out.println(allRands.mmul(allOnes).reshape(3,1));
//        System.out.println(allRands.mmul(allOnes).reshape(1,3));
//        System.out.println(Transforms.log(allOnes));
//        System.out.println(allOnes.shape()[0]);
//        System.out.println(allOnes.shape()[1]);
//        System.out.println(allRands.sum(0));
//        System.out.println(allRands.sum(0).getFloat(0));
//        System.out.println(allRands.sumNumber().floatValue());
    }

    @Test
    public void testRanker() {
        Ranker ranker;
        RankerTrainer trainer = new RankerTrainer();
        MetricScorerFactory mcf = new MetricScorerFactory();

        String file = getClass().getResource("/fixtures/test_data_1.csv").getFile();
        List<RankList> rankLists = FeatureManager.readInput(file);
        for(RankList rnk : rankLists) {
            for(int i=0; i < rnk.size(); i++) {
                System.out.println(rnk.get(i));
            }
        }

        System.out.println(rankLists);

        ranker = trainer.train(RANKER_TYPE.LISTNET, rankLists, FeatureManager.getFeatureFromSampleVector(rankLists), mcf.createScorer(METRIC.ERR));

        System.out.println(">>>>" + ranker.getFeatures());
        Arrays.stream(ranker.getFeatures()).forEach(x -> System.out.print(x + " "));
        System.out.println(">>>>" + ranker.model());
    }

    public Object[] parametersTestCreateAdjustedMTable() {
        List<Object> parameters = new ArrayList<Object>();

        Object[] case1 = {10, 0.2, 0.15, new int[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}};
        Object[] case2 = {20, 0.25, 0.1, new int[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}};
        Object[] case3 = {30, 0.3, 0.05, new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 4, 4, 4, 4, 4}};

        parameters.add(case1);
        parameters.add(case2);
        parameters.add(case3);

        return parameters.toArray();
    }

    @Test
    @Parameters(method = "parametersTestCreateAdjustedMTable")
    public void testCreateAdjustedMTable(int k, double p, double alpha, int[] expected){
    }

    @Test
    @Parameters({"10, 0.25, 0.15, 0.15",
                 "20, 0.25, 0.1, 0.07812500000000001",
                 "30, 0.3, 0.15, 0.075"})
    public void testAdjustAlpha(int k, double p, double alpha, double expected) {
    }
}
