package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.DeltrDoc;
import com.github.fairsearch.deltr.models.DeltrDocImpl;
import com.github.fairsearch.deltr.models.DeltrTopDocs;
import com.github.fairsearch.deltr.models.DeltrTopDocsImpl;
import com.github.fairsearch.deltr.models.TrainStep;

import java.util.ArrayList;
import java.util.List;

public class HelloWorld {

    public static void main(String[] args) {
       // create some data
        List<DeltrTopDocs> trainSet = new ArrayList<>();

        // create the first "query" to learn
        DeltrTopDocs trainQuery = new DeltrTopDocsImpl(1); // 1 is the question ID

        // let's create each train doc manually (docId, judgement/score)

        // item 1
        DeltrDoc item1 = new DeltrDocImpl(1, 1);
        // add the features to the document (featureName, featureValue, isProtected - since this is a protected feature)
        item1.put("f0", true);
        // add the features to the document (featureName, featureValue)
        item1.put("f1", 0.962650646167003);

        // item 2
        DeltrDoc item2 = new DeltrDocImpl(2, 0.98f);
        item2.put("f0", false);
        item2.put("f1", 0.940172822166108);

        // item 3
        DeltrDoc item3 = new DeltrDocImpl(3, 0.96f);
        item3.put("f0", false);
        item3.put("f1", 0.925288002880488);

        // item 4
        DeltrDoc item4 = new DeltrDocImpl(2, 0.94f);
        item4.put("f0", true);
        item4.put("f1", 0.896143226020877);

        // item 5
        DeltrDoc item5 = new DeltrDocImpl(3, 0.92f);
        item5.put("f0", false);
        item5.put("f1", 0.89180775633204);

        // item 6
        DeltrDoc item6 = new DeltrDocImpl(3, 0.9f);
        item6.put("f0",false);
        item6.put("f1", 0.838704766545679);

        // add the items in the trainQuery
        DeltrDoc[] docsArr = new DeltrDoc[]{item1, item2, item3, item4, item5, item6};
        trainQuery.put(docsArr);

        // the trainQuery to the trainSet
        trainSet.add(trainQuery);

        // setup the parameters for the DELTR object
        double gamma = 1.0; // value of the gamma paramete
        int numberOfIterations = 10; // number of iterations the training should run
        boolean shouldStandardize = true; // let's apply standardization to the features

        // create the Deltr object
        Deltr deltr = new Deltr(gamma, numberOfIterations, shouldStandardize);

        // train the model
        deltr.train(trainSet);
        // deltr.getOmega() -> [0.025225676596164703, 0.0798153206706047]

        // let's create a sample prediction set
        DeltrTopDocs preidictionSet = new DeltrTopDocsImpl(2); // 2 is the question ID

        // let's create each prediction doc manually (docId, judgement/score)

        // item 7
        DeltrDoc item7 = new DeltrDocImpl(7, 0.9645f); // the curret score is not important
        item7.put("f0", false);
        item7.put("f1", 0.9645);

        // item 8
        DeltrDoc item8 = new DeltrDocImpl(8, 0.9524f);
        item8.put("f0", false);
        item8.put("f1", 0.9524);

        // item 9
        DeltrDoc item9 = new DeltrDocImpl(9, 0.9285f);
        item9.put("f0", false);
        item9.put("f1", 0.9285);

        // item 10
        DeltrDoc item10 = new DeltrDocImpl(10, 0.8961f);
        item10.put("f0", false);
        item10.put("f1", 0.8961);

        // item 11
        DeltrDoc item11 = new DeltrDocImpl(11, 0.8911f);
        item11.put("f0", true);
        item11.put("f1", 0.8911);

        // item 12
        DeltrDoc item12 = new DeltrDocImpl(12, 0.8312f);
        item12.put("f0", true);
        item12.put("f1", 0.8312);

        //add the items in the set
        DeltrDoc[] predArr = new DeltrDoc[]{item7, item8, item9, item10, item11, item12};
        preidictionSet.put(predArr);

        DeltrTopDocs reranked = deltr.rank(preidictionSet);
        // reranked ->
        // id:11, judgement:0,072242, isProtected:true
        // id:12, judgement:0,061806, isProtected:true
        // id:7, judgement:0,059804, isProtected:false
        // id:8, judgement:0,057696, isProtected:false
        // id:9, judgement:0,053532, isProtected:false
        // id:10, judgement:0,047887, isProtected:false

        //let's checkout the log (list of instances of <TrainStep>)
        for(TrainStep step : deltr.getLog()) {
            System.out.println(step);
        }
        //timestamp:1556475668410, lossStandard:5,999854, lossExposure:0,000000
        //timestamp:1556475668492, lossStandard:5,999854, lossExposure:0,000000
        //timestamp:1556475668548, lossStandard:5,999852, lossExposure:0,000000
        // .
        // .
        // .
    }
}
