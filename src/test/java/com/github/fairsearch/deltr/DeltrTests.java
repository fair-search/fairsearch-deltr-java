package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.DeltrDoc;
import com.github.fairsearch.deltr.models.DeltrDocImpl;
import com.github.fairsearch.deltr.models.DeltrTopDocs;
import com.github.fairsearch.deltr.models.DeltrTopDocsImpl;
import com.github.fairsearch.deltr.models.TrainStep;
import com.google.common.primitives.Doubles;
import com.sun.javafx.scene.shape.PathUtils;
import junitparams.JUnitParamsRunner;
import junitparams.Parameters;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.string.NDArrayStrings;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.IntStream;

@RunWith(JUnitParamsRunner.class)
public class DeltrTests {

    private static final double OFFSET = 0.0001; // result tolerance for DeltrTests

    @Test
    public void testNd4j() {
        int nRows = 3;
        int nColumns = 5;
        INDArray allRands = Nd4j.rand(nRows, nColumns);
        INDArray allZeros = Nd4j.zeros(nRows, nColumns);
        INDArray allOnes = Nd4j.ones(nColumns, 1);
        INDArray omega = Nd4j.rand(nColumns, 1);

        allRands.put(0, 0, 1);
        allRands.put(1, 0, 0);
        allRands.put(2, 0, 1);
        TreeMap<String, Double> features = new TreeMap<String, Double>();
        features.put("0", 1.0);
        features.put("1", 1.0);

//        features.values().parallelStream().forEach((x) -> x = x/2);

//        features.values().parallelStream().forEach((x) ->{
//            System.out.println(x);
//        });

        System.out.println(Nd4j.create(new double[]{152}));
        System.out.println(Transforms.exp(Nd4j.create(new double[]{152})));
//        System.out.println(allOnes.stdNumber());
//        System.out.println(allRands.mmul(allOnes).reshape(1,3));
//        System.out.println(Transforms.log(allOnes));
//        System.out.println(allOnes.shape()[0]);
//        System.out.println(allRands.sum(0).getFloat(0));
//        System.out.println(allRands.sumNumber().floatValue());
    }

    @Test
    @Parameters({"1, test_data_1.csv, true"})
    public void testTrainFromCSV(double gamma, String fileName, boolean shouldStandardize) {
        String filePath = getClass().getResource(String.format("/fixtures/%s", fileName)).getFile();
        List<DeltrTopDocs> ranks = prepareData(filePath);

        Deltr deltr = new Deltr(gamma, 10 , shouldStandardize);

        deltr.train(ranks);

        evaluateTrainer(deltr);
    }

    @Test
    @Parameters({"1, 20, 5, 1, 10, false"})
    public void testTrainSyntheticData(int numberOfQuestions, int numberOfElementsPerQuestion, int numberOfFeatures,
                                       double gamma, int numberOfIterations, boolean shouldStandardize) {
        //TODO: implement this test!!
        SyntheticDatasetCreator syntheticDatasetCreator = new SyntheticDatasetCreator(numberOfQuestions,
                numberOfElementsPerQuestion, 2, numberOfFeatures);

        List<DeltrTopDocs> trainSet = syntheticDatasetCreator.generateDataset();

        trainSet.get(0).docs().stream().forEach((x) -> System.out.println(x.toString()));

        Deltr deltr = new Deltr(gamma, numberOfIterations, shouldStandardize);

        deltr.train(trainSet);

        evaluateTrainer(deltr);
    }

    private void evaluateTrainer(Deltr deltr) {
        assert deltr.getOmega() != null;
        assert deltr.getLog() != null;

        int precision = 8;
        if(deltr.getLog().size() > 1) {
            TrainStep prev = deltr.getLog().get(0);
            for (int i=1; i<deltr.getLog().size(); i++) {
                System.out.println(new NDArrayStrings(precision).format(deltr.getLog().get(i).getOmega()) + " - " + deltr.getLog().get(i).getTotalCost());
                assert deltr.getLog().get(i).getTotalCost() <= prev.getTotalCost();
                prev = deltr.getLog().get(i);
            }
        } else {
            assert deltr.getLog().get(0) != null;
        }
    }

    private List<DeltrTopDocs> prepareData(String filePath) {
        List<DeltrTopDocs> ranks = new ArrayList<>();

        String line;

        int currentQueryId = -1;
        int currentDocId = 0;

        DeltrTopDocs docs = null;
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {

            String[] header = br.readLine().split(",");

            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                int queryId = Integer.parseInt(row[0]);
                int gender = Integer.parseInt(row[1]);
                double feature = Double.parseDouble(row[2]);
                float judgement = Float.parseFloat(row[3]);

                if(queryId != currentQueryId) {
                    if(docs != null) {
                        ((TopDocs)docs).totalHits = currentDocId;
                        ranks.add(docs);
                    }
                    currentQueryId = queryId;
                    currentDocId = 0;

                    docs = new DeltrTopDocsImpl(queryId);
                    // the first element of the list should be the highest ranking
                    ((TopDocs)docs).setMaxScore(judgement);
                }

                DeltrDocImpl doc = new DeltrDocImpl(currentDocId, judgement, gender == 1);
                doc.set("0", (double)gender, true);
                doc.set("1", feature);

                currentDocId += 1;

                ScoreDoc[] tmp = ((DeltrTopDocsImpl)docs).scoreDocs;
                ((DeltrTopDocsImpl)docs).scoreDocs = new ScoreDoc[tmp.length + 1];
                System.arraycopy(tmp, 0, ((DeltrTopDocsImpl)docs).scoreDocs, 0 ,tmp.length);
                ((DeltrTopDocsImpl)docs).scoreDocs[tmp.length] = doc;

            }
            ranks.add(docs);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ranks;
    }

    @Test
//    public void testRanker() {
//        Ranker ranker;
//        RankerTrainer trainer = new RankerTrainer();
//        MetricScorerFactory mcf = new MetricScorerFactory();
//
//        String file = getClass().getResource("/fixtures/test_data_2.csv").getFile();
//        List<RankList> rankLists = FeatureManager.readInput(file);
//        for(RankList rnk : rankLists) {
//            for(int i=0; i < rnk.size(); i++) {
//                System.out.println(rnk.get(i));
//            }
//        }
//
//        System.out.println(rankLists);
//
//        ranker = trainer.train(RANKER_TYPE.LISTNET, rankLists, FeatureManager.getFeatureFromSampleVector(rankLists), mcf.createScorer(METRIC.ERR));
//
//        System.out.println(">>>>" + ranker.getFeatures());
//        Arrays.stream(ranker.getFeatures()).forEach(x -> System.out.print(x + " "));
//        System.out.println(">>>>" + ranker.model());
//    }

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
