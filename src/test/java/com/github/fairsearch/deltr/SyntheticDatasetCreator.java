package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.DeltrDocImpl;
import com.github.fairsearch.deltr.models.DeltrTopDocs;
import com.github.fairsearch.deltr.models.DeltrTopDocsImpl;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.lucene.search.ScoreDoc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.stream.IntStream;

public class SyntheticDatasetCreator {

    private int numberOfQuestions;
    private int numberOfElementsPerQuestion;
    private int numberOfCategories;
    private int numberOfFeatures;

    private MersenneTwister random;

    public SyntheticDatasetCreator(int numberOfQuestions, int numberOfElementsPerQuestion,
                                   int numberOfCategories, int numberOfFeatures) {
        this.numberOfQuestions = numberOfQuestions;
        this.numberOfElementsPerQuestion = numberOfElementsPerQuestion;
        this.numberOfCategories = numberOfCategories;
        this.numberOfFeatures = numberOfFeatures;

        random = new MersenneTwister();
    }

    public List<DeltrTopDocs> generateDataset() {
        List<DeltrTopDocs> result = new ArrayList<>();

        //generate sample weights for dot product
        int[] weights = IntStream.range(0, this.numberOfFeatures).map((x) -> 10 * x).toArray();

        // generate a list of DeltrTopDocs
        IntStream.range(0, this.numberOfQuestions).forEach((i) -> {

            // initialize document array
            ScoreDoc[] scoreDocs = new ScoreDoc[this.numberOfElementsPerQuestion];

            // for each array generate documents
            IntStream.range(0, this.numberOfElementsPerQuestion).forEach((j) -> {
                // generate protected feature and put that as first
                int isProtected = random.nextDouble() < 0.2 ? 1 : 0;

                // initialize doc
                DeltrDocImpl doc = new DeltrDocImpl(j, 0, isProtected == 1);
                doc.put("0", isProtected == 1);

                // for each DeltrTopDoc generate features
                double mu = random.nextDouble();
                double sigma = random.nextDouble();

                // initialize element score
                double score = weights[0] * isProtected;

                // generate rest of features
                for(int k=1; k<this.numberOfFeatures; k++) {
                    // generate feature
                    double feature = mu + random.nextGaussian()*sigma;

                    // add feature
                    doc.put(String.valueOf(k), feature);

                    // add to dot product
                    score += weights[k] * feature;
                }

                doc.rejudge(score);

                scoreDocs[j] = doc;
            });

            // find min and max
            OptionalDouble max = Arrays.stream(scoreDocs).mapToDouble((x) -> (double)x.score).max();
            OptionalDouble min = Arrays.stream(scoreDocs).mapToDouble((x) -> (double)x.score).min();

            // apply min-max normalization
            IntStream.range(0, this.numberOfElementsPerQuestion).forEach((j) -> {
                scoreDocs[j].score = (float)((scoreDocs[j].score - min.getAsDouble())
                        /(max.getAsDouble() - min.getAsDouble()));
            });

            // sort the docs according to the score
            Arrays.sort(scoreDocs, (o1, o2) -> {
                if(o1.score < o2.score)
                    return 1;
                else if(o1.score > o2.score)
                    return -1;
                return 0;
            });

            // create DeltrTopDocs
            DeltrTopDocs docs = new DeltrTopDocsImpl(i, this.numberOfElementsPerQuestion, scoreDocs, Float.NaN);

            // add to list
            result.add(docs);
        });

        return result;
    }
}
