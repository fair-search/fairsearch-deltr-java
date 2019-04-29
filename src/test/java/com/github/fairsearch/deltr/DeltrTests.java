package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.DeltrDocImpl;
import com.github.fairsearch.deltr.models.DeltrTopDocs;
import com.github.fairsearch.deltr.models.DeltrTopDocsImpl;
import com.github.fairsearch.deltr.models.TrainStep;
import junitparams.JUnitParamsRunner;
import junitparams.Parameters;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.IntStream;

@RunWith(JUnitParamsRunner.class)
public class DeltrTests {

    private static final double OFFSET = 0.001; // result tolerance for DeltrTests

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

//        features.values().parallelStream().forEach((x) ->{
//            System.out.println(x);
//        });

        System.out.println(Nd4j.create(new double[]{152}));
        System.out.println(Transforms.exp(Nd4j.create(new double[]{152})));
//        System.out.println(allRands.mmul(allOnes).reshape(1,3));
//        System.out.println(Transforms.log(allOnes));
//        System.out.println(allOnes.shape()[0]);
//        System.out.println(allRands.sum(0).getFloat(0));
    }

    @Test
    @Parameters({"1, test_data_1.csv, true",
                 "1, test_data_1.csv, false"})
    public void testTrainFromFixtures(double gamma, String fileName, boolean shouldStandardize) {
        String filePath = getClass().getResource(String.format("/fixtures/%s", fileName)).getFile();
        List<DeltrTopDocs> ranks = prepareData(filePath);

        Deltr deltr = new Deltr(gamma, 10 , shouldStandardize);

        deltr.train(ranks);

        evaluateTrainer(deltr);
    }

    @Test
    @Parameters({
                "1, 20, 5, 1, 100, false",
                "1, 50, 10, 0.8, 500, false",
                "1, 1000, 3, 1, 1000, false",
                "2, 200, 4, 0.9, 300, false",
                "3, 100, 5, 1, 200, false",
                "4, 50, 6, 1, 100, false",
                "1, 20, 5, 1, 100, true",
                "1, 50, 10, 0.8, 500, true",
                "1, 1000, 3, 1, 1000, true",
                "2, 200, 4, 0.9, 300, true",
                "3, 100, 5, 1, 200, true",
                "4, 50, 6, 1, 100, true"
    })
    public void testTrainSyntheticData(int numberOfQuestions, int numberOfElementsPerQuestion, int numberOfFeatures,
                                       double gamma, int numberOfIterations, boolean shouldStandardize) {
        // create a train dataset
        SyntheticDatasetCreator syntheticDatasetCreator = new SyntheticDatasetCreator(numberOfQuestions,
                numberOfElementsPerQuestion, 2, numberOfFeatures);

        List<DeltrTopDocs> trainSet = syntheticDatasetCreator.generateDataset();

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
                if(deltr.getLog().get(i).getTotalCost() > prev.getTotalCost())
                    System.out.println(deltr.getLog().get(i).getTotalCost() + " <= " +  prev.getTotalCost());
                assert prev.getTotalCost() - deltr.getLog().get(i).getTotalCost() + OFFSET >= 0;
                prev = deltr.getLog().get(i);
            }
        } else {
            assert deltr.getLog().get(0) != null;
        }
    }

    @Test
    public void testRankEmptyDeltr() {
        try{
            Deltr d = new Deltr(1);
            d.rank(null);
            assert false;
        } catch (NullPointerException e) {
            assert true;
        } catch (Exception e) {
            assert false;
        }
    }

    private static class DeltrMock extends Deltr {

        public DeltrMock(double gamma) {
            super(gamma);
        }

        public void setOmega(double[] omega) {
            this.omega = omega;
        }

        public void setMu(double mu) {
            this.mu = mu;
        }

        public void setSigma(double sigma) {
            this.sigma = sigma;
        }
    }

    @Test
    @Parameters({
            "20, 5, false",
            "50, 10, false",
            "1000, 3, false",
            "20, 5, true",
            "50, 10, true",
            "1000, 3, true",
    })
    public void testRankDeltr(int numberOfElements, int numberOfFeatures, boolean shouldStandardize) {
        // create a dataset
        SyntheticDatasetCreator syntheticDatasetCreator = new SyntheticDatasetCreator(1,
                numberOfElements, 2, numberOfFeatures);
        List<DeltrTopDocs> predictionSets = syntheticDatasetCreator.generateDataset();

        // the first (and only) subset is used for prediction
        DeltrTopDocs predictionSet = predictionSets.get(0);

        //generate sample weights for omega
        double[] omega = IntStream.range(0, numberOfFeatures).mapToDouble((x) -> 10 * x).toArray();

        // create the ranker and set omega
        DeltrMock deltrMock = new DeltrMock(1); // gammma is not neccessary here
        deltrMock.setOmega(omega);

        //set mu and sigma if standardization is required
        if(shouldStandardize) {
            deltrMock.setMu(1);
            deltrMock.setSigma(1);
        }

        // get the results
        DeltrTopDocs result = deltrMock.rank(predictionSet);

        // calculate the results manually
        IntStream.range(0, numberOfElements).parallel().forEach((j) -> {
            double score = IntStream.range(0, numberOfFeatures)
                    .parallel()
                    .mapToDouble( x -> omega[x] * predictionSet.doc(j).feature(x))
                    .sum();
            predictionSet.doc(j).rejudge(score);
        });
        //sort the results
        predictionSet.reorder();

        //compare the results
        IntStream.range(0, numberOfElements).parallel().forEach((j) -> {
            assert predictionSet.doc(j).id() == result.doc(j).id();
        });
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
                doc.put("0", gender == 1);
                doc.put("1", feature);

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

}
