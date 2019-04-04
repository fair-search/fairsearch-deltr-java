package com.github.fairsearch.deltr;

import junitparams.JUnitParamsRunner;
import junitparams.Parameters;
import org.apache.lucene.search.TopDocs;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@RunWith(JUnitParamsRunner.class)
public class DeltrTests {

    private static final double OFFSET = 0.0001; // result tolerance for DeltrTests

    public Object[] parametersTestIsFair() {
        List<Object> parameters = new ArrayList<Object>();

//        FairScoreDoc[] docs1 = {new FairScoreDoc(20,20,false),new FairScoreDoc(19,19,true),
//                new FairScoreDoc(18,18,false),new FairScoreDoc(17,17,false),
//                new FairScoreDoc(16,16,false),new FairScoreDoc(15,15,false),
//                new FairScoreDoc(14,14,false),new FairScoreDoc(13,13,true),
//                new FairScoreDoc(12,12,false),new FairScoreDoc(11,11,true),
//                new FairScoreDoc(10,10,false),new FairScoreDoc(9,9,false),
//                new FairScoreDoc(8,8,true),new FairScoreDoc(7,7,false),
//                new FairScoreDoc(6,6,false),new FairScoreDoc(5,5,true),
//                new FairScoreDoc(4,4,true),new FairScoreDoc(3,3,false),
//                new FairScoreDoc(2,2,false),new FairScoreDoc(1,1,false)};
//        Object[] case1 = {20, 0.25, 0.1, new TopDocs(docs1.length, docs1, Float.NaN)};
//
//        FairScoreDoc[] docs2 = {new FairScoreDoc(20,20,false),new FairScoreDoc(19,19,true),
//                new FairScoreDoc(18,18,false),new FairScoreDoc(17,17,true),
//                new FairScoreDoc(16,16,true),new FairScoreDoc(15,15,false),
//                new FairScoreDoc(14,14,false),new FairScoreDoc(13,13,true),
//                new FairScoreDoc(12,12,false),new FairScoreDoc(11,11,true),
//                new FairScoreDoc(10,10,false),new FairScoreDoc(9,9,false),
//                new FairScoreDoc(8,8,true),new FairScoreDoc(7,7,false),
//                new FairScoreDoc(6,6,false),new FairScoreDoc(5,5,true),
//                new FairScoreDoc(4,4,true),new FairScoreDoc(3,3,false),
//                new FairScoreDoc(2,2,false),new FairScoreDoc(1,1,false)};
//        Object[] case2 = {20, 0.3, 0.1, new TopDocs(docs2.length, docs2, Float.NaN)};

        parameters.add(null);
        parameters.add(null);

        return parameters.toArray();
    }

    @Test
    @Parameters(method = "parametersTestIsFair")
    public void testIsFair(int k, double p, double alpha, TopDocs ranking) {
        Deltr deltr = new Deltr(k, p, alpha);

        assertEquals(k, ranking.scoreDocs.length);

        assertEquals(true, deltr.isFair(ranking));
    }

    public Object[] parametersTestCreateUnadjustedMTable() {
        List<Object> parameters = new ArrayList<Object>();

        Object[] case1 = {10, 0.2, 0.15, new int[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}};
        Object[] case2 = {20, 0.25, 0.1, new int[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3}};
        Object[] case3 = {30, 0.3, 0.05, new int[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3,
                3, 3, 3, 4, 4, 4, 4, 5, 5, 5}};

        parameters.add(case1);
        parameters.add(case2);
        parameters.add(case3);

        return parameters.toArray();
    }

    @Test
    @Parameters(method = "parametersTestCreateUnadjustedMTable")
    public void testCreateUnadjustedMTable(int k, double p, double alpha, int[] expected){
        Deltr deltr = new Deltr(k, p, alpha);
        assertArrayEquals(expected, deltr.createUnadjustedMTable());
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
        Deltr deltr = new Deltr(k, p, alpha);
        assertArrayEquals(expected, deltr.createAdjustedMTable());
    }

    @Test
    @Parameters({"10, 0.25, 0.15, 0.15",
                 "20, 0.25, 0.1, 0.07812500000000001",
                 "30, 0.3, 0.15, 0.075"})
    public void testAdjustAlpha(int k, double p, double alpha, double expected) {
        Deltr deltr = new Deltr(k, p, alpha);

        double actual = deltr.adjustAlpha();

        assertEquals(expected, actual, OFFSET);
    }

    @Test
    @Parameters({"10, 0.2, 0.15, 0.13421772800000065",
            "20, 0.25, 0.1, 0.10515247355215251",
            "30, 0.3, 0.05, 0.04877730797178359"})
    public void testComputeFailureProbability(int k, double p, double alpha, double expected) {
        Deltr deltr = new Deltr(k, p, alpha);

        int[] adjustedMTable = deltr.createAdjustedMTable();

        double actual = deltr.computeFailureProbability(adjustedMTable);

        assertEquals(expected, actual, OFFSET);
    }

    @Test
    @Parameters({"10, 0.2, 0.15",
            "20, 0.25, 0.1",
            "30, 0.3, 0.05"})
    public void testMTableGeneration(int k, double p, double alpha) {
        Deltr deltr = new Deltr(k, p, alpha);

        // create an adjusted mtable
        int[] adjustedMTable = deltr.createAdjustedMTable();

        // get alpha adjusted
        double alphaAdjusted = deltr.adjustAlpha();

        //create a new unadjusted mtable with the new alpha
        Deltr deltrAdjusted = new Deltr(k, p, alphaAdjusted);
        int[] unadjustedMTable = deltrAdjusted.createUnadjustedMTable();

        assertArrayEquals(adjustedMTable, unadjustedMTable);
    }


    public Object[] parametersTestReRank() {
        List<Object> parameters = new ArrayList<Object>();

//        FairScoreDoc[] docs1 = {new FairScoreDoc(20, 20, false), new FairScoreDoc(19, 19, false),
//                new FairScoreDoc(18, 18, false), new FairScoreDoc(17, 17, false),
//                new FairScoreDoc(16, 16, false), new FairScoreDoc(15, 15, false),
//                new FairScoreDoc(14, 14, false), new FairScoreDoc(13, 13, false),
//                new FairScoreDoc(12, 12, false), new FairScoreDoc(11, 11, false),
//                new FairScoreDoc(10, 10, false), new FairScoreDoc(9, 9, false),
//                new FairScoreDoc(8, 8, false), new FairScoreDoc(7, 7, false),
//                new FairScoreDoc(6, 6, false), new FairScoreDoc(5, 5, true),
//                new FairScoreDoc(4, 4, true), new FairScoreDoc(3, 3, true),
//                new FairScoreDoc(2, 2, true), new FairScoreDoc(1, 1, true)};
//        Object[] case1 = {20, 0.25, 0.1, new TopDocs(docs1.length, docs1, Float.NaN)};
//
//        FairScoreDoc[] docs2 = {new FairScoreDoc(20,20,false),new FairScoreDoc(19,19,false),
//                new FairScoreDoc(18,18,false),new FairScoreDoc(17,17,false),
//                new FairScoreDoc(16,16,false),new FairScoreDoc(15,15,false),
//                new FairScoreDoc(14,14,false),new FairScoreDoc(13,13,false),
//                new FairScoreDoc(12,12,false),new FairScoreDoc(11,11,true),
//                new FairScoreDoc(10,10,true),new FairScoreDoc(9,9,false),
//                new FairScoreDoc(8,8,true),new FairScoreDoc(7,7,false),
//                new FairScoreDoc(6,6,false),new FairScoreDoc(5,5,true),
//                new FairScoreDoc(4,4,true),new FairScoreDoc(3,3,true),
//                new FairScoreDoc(2,2,true),new FairScoreDoc(1,1,true)};
//        Object[] case2 = {20, 0.3, 0.1, new TopDocs(docs2.length, docs2, Float.NaN)};

        parameters.add(null);
        parameters.add(null);

        return parameters.toArray();
    }

    @Test
    @Parameters(method = "parametersTestReRank")
    public void testReRank(int k, double p, double alpha, TopDocs ranking) {
        Deltr deltr = new Deltr(k, p, alpha);

        TopDocs reRanked = deltr.reRank(ranking);

        // input should not be deltr
        assertEquals(false, deltr.isFair(ranking));

        // contents of both arrays should be same
        List<Integer> originalIds = Arrays.stream(ranking.scoreDocs).mapToInt(x -> x.doc).
                boxed().collect(Collectors.toList());
        List<Integer> rerankedIds = Arrays.stream(reRanked.scoreDocs).mapToInt(x -> x.doc).
                boxed().collect(Collectors.toList());
        originalIds.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1 < o2 ? 1 : -1;
            }
        });
        rerankedIds.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o1 < o2 ? 1 : -1;
            }
        });
        assertArrayEquals(originalIds.toArray(), rerankedIds.toArray());

        // output should not be deltr
        assertEquals(true, deltr.isFair(reRanked));
    }
}
