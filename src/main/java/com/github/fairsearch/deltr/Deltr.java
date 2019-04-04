package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.lib.FairTopK;
import com.github.fairsearch.deltr.lib.MTableFailProbPair;
import com.github.fairsearch.deltr.lib.MTableGenerator;
import com.github.fairsearch.deltr.lib.RecursiveNumericFailProbabilityCalculator;
import com.github.fairsearch.deltr.lib.FailProbabilityCalculator;
import com.github.fairsearch.deltr.utils.FairScoreDoc;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * This class serves as a wrapper around the utilities we have created for FA*IR ranking
 */
public class Deltr {

    private static final Logger LOGGER = Logger.getLogger(Deltr.class.getName());

    private int k; //the total number of elements
    private double p; //the proportion of protected candidates in the top-k ranking
    private double alpha; //the significance level

    private FairTopK fairTopK;

    public Deltr(int k, double p, double alpha){
        //check the parameters before using them (this will throw an exception if things don't look good
        validateBasicParameters(k, p, alpha);

        //asign the attributes
        this.k = k;
        this.p = p;
        this.alpha = alpha;

        //initialize the FairTopK re-ranker
        fairTopK = new FairTopK();
    }

    /**
     * Creates an mtable using alpha unadjusted
     * @return            The generated mtable (int[])
     */
    public int[] createUnadjustedMTable() {
        return createMTable(this.alpha, false);
    }

    /**
     * Creates an mtable using alpha adjusted
     * @return            The generated mtable (int[])
     */
    public int[] createAdjustedMTable() {
        return createMTable(this.alpha, true);
    }

    /**
     * Creates an mtable by passing your own alpha
     * @param alpha       The significance level
     * @param adjustAlpha Boolean indicating whether the alpha be adjusted or not
     * @return            The generated mtable (int[])
     */
    private int[] createMTable(double alpha, boolean adjustAlpha) {
        //check if passed alpha is ok
        validateAlpha(alpha);

        //create the mtable
        MTableGenerator generator = new MTableGenerator(this.k, this.p, alpha, adjustAlpha);
        return Arrays.copyOfRange(generator.getMTable(), 1, generator.getMTable().length);
    }

    /**
     * Computes the alpha adjusted for the given set of parameters
     * @return            The adjusted alpha
     */
    public double adjustAlpha() {
        RecursiveNumericFailProbabilityCalculator adjuster = new RecursiveNumericFailProbabilityCalculator(this.k, this.p, this.alpha);
        MTableFailProbPair failProbPair = adjuster.adjustAlpha();
        return failProbPair.getAlpha();
    }

    /**
     * Computes analytically the probability that a ranking created with the simulator will fail to pass the mtable
     * @param mtable      The mtable against to compute the fail probability
     * @return            The fail probability
     */
    public double computeFailureProbability(int[] mtable) {
        if(mtable.length != this.k) {
            LOGGER.severe("Number of elements k and (int[]) mtable length must be equal!");
            System.exit(-1);
        }

        FailProbabilityCalculator calculator = new RecursiveNumericFailProbabilityCalculator(this.k, this.p, this.alpha);

        // the internal mechanics of the MTableGenerator works with k+1 table length
        // so, we must create a longer interim mtable with a 0th position
        int[] interimMTable = new int[mtable.length + 1];
        System.arraycopy(mtable, 0, interimMTable, 1, mtable.length);

        return calculator.calculateFailProbability(interimMTable);
    }

    /**
     * Checks if the ranking is fair in respect to the mtable
     * @param docs        The ranking to be checked
     * @param mtable      The mtable against to check
     * @return            Returns whether the rankings statisfies the mtable
     */
    public static boolean checkRankingMTable(TopDocs docs, int[] mtable) {
        int countProtected = 0;

        //if the mtable has a different number elements than there are in the top docs return false
        if(docs.scoreDocs.length != mtable.length)
            throw new IllegalArgumentException("Number of documents in (TopDocs) docs and (int[]) mtable length are not the same!");

        //check number of protected element at each rank
        for(int i=0; i < docs.scoreDocs.length; i++) {
            countProtected += ((FairScoreDoc)docs.scoreDocs[i]).isProtected ? 1 : 0;
            if(countProtected < mtable[i])
                return false;
        }
        return true;
    }

    /**
     * Checks if the ranking is fair for the given parameters
     * @param docs        The ranking to be checked
     * @return            Returns a boolean which specifies whether the ranking is fair
     */
    public boolean isFair(TopDocs docs) {
        return checkRankingMTable(docs, this.createAdjustedMTable());
    }

    /**
     * Applies FA*IR re-ranking to the input ranking with an adjusted mtable
     * @param docs      The ranking to be re-ranked
     * @return          A new (fair) ranking
     */
    public TopDocs reRank(TopDocs docs) {
        List<ScoreDoc> protectedElements = new ArrayList<ScoreDoc>();
        List<ScoreDoc> nonProtectedElements =  new ArrayList<ScoreDoc>();

        for(int i=0; i<docs.scoreDocs.length; i++) {
            FairScoreDoc current = (FairScoreDoc)docs.scoreDocs[i];
            if(current.isProtected) {
                protectedElements.add(current);
            } else {
                nonProtectedElements.add(current);
            }
        }

        return this.fairTopK.fairTopK(nonProtectedElements, protectedElements, this.k, this.p, this.alpha);
    }

    /**
     * Validates if k, p and alpha are in the required ranges
     * @param k           Total number of elements (above or equal to 10)
     * @param p           The proportion of protected candidates in the top-k ranking (between 0.02 and 0.98)
     * @param alpha       The significance level (between 0.01 and 0.15)
     */
    private static void validateBasicParameters(int k, double p, double alpha) {
        if(k < 10 || k > 400) {
            if(k < 2) {
                throw new IllegalArgumentException("Total number of elements `k` should be between 10 and 400");
            } else {
                LOGGER.warning("Library has not been tested with values outside this range");
            }
        }
        if(p < 0.02 || p > 0.98) {
            if(p < 0 || p > 1) {
                throw new IllegalArgumentException("The proportion of protected candidates `p` in the top-k ranking should " +
                        "be between 0.02 and 0.98");
            } else {
                LOGGER.warning("Library has not been tested with values outside this range");
            }
        }
        validateAlpha(alpha);
    }

    private static void validateAlpha(double alpha) {
        if(alpha < 0.01 || alpha > 0.15) {
            if(alpha < 0.001 || alpha > 0.5) {
                throw new IllegalArgumentException("The significance level `alpha` must be between 0.01 and 0.15");
            } else {
                LOGGER.warning("Library has not been tested with values outside this range");
            }
        }
    }
}
