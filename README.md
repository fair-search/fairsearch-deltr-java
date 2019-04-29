# Fair search DELTR for Java

[![image](https://api.travis-ci.org/fair-search/fairsearchcore-java.svg?branch=master)](https://travis-ci.org/fair-search/fairsearchcore-java)
[![image](https://img.shields.io/pypi/l/fairsearchcore.svg)](https://pypi.org/project/fairsearchcore/)

This is the Java library that implements the [DELTR](https://arxiv.org/pdf/1805.08716.pdf) model for fair ranking.

## Installation

You can import the library with maven in your `pom.xml` file:
```xml
<dependency>
  <groupId>com.github.fair-search</groupId>
  <artifactId>fairsearch-deltr</artifactId>
  <version>1.0.0</version>
</dependency>
```
or, if you are using Gradle, in your `build.gradle` file add this in the `dependencies` block:
```gradle
compile "com.github.fair-search:fairsearch-deltr:1.0.0"
```

And, that's it!

## Using it in your code

Add the JAR file to the build path of your project and you are *set*. The key methods are contained in the following class:
- `com.github.fairsearch.Deltr`

The library contains sufficient Java doc for each of the functions.

## Sample usage
Creating and analyzing mtables:

### Train a model

You need to train the model before it can rank documents.
```java
package com.github.fairsearch.deltr;

import com.github.fairsearch.deltr.models.DeltrDoc;
import com.github.fairsearch.deltr.models.DeltrDocImpl;
import com.github.fairsearch.deltr.models.DeltrTopDocs;
import com.github.fairsearch.deltr.models.DeltrTopDocsImpl;

import java.util.ArrayList;
import java.util.Arrays;
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
        item1.put("f0", 1.0, true);
        // add the features to the document (featureName, featureValue)
        item1.put("f1", 0.962650646167003);

        // item 2
        DeltrDoc item2 = new DeltrDocImpl(2, 0.98f);
        item2.put("f0", 0.0, false);
        item2.put("f1", 0.940172822166108);

        // item 3
        DeltrDoc item3 = new DeltrDocImpl(3, 0.96f);
        item3.put("f0", 0.0, false);
        item3.put("f1", 0.925288002880488);

        // item 4
        DeltrDoc item4 = new DeltrDocImpl(2, 0.94f);
        item4.put("f0", 1.0, true);
        item4.put("f1", 0.896143226020877);

        // item 5
        DeltrDoc item5 = new DeltrDocImpl(3, 0.92f);
        item5.put("f0", 0.0, false);
        item5.put("f1", 0.89180775633204);

        // item 6
        DeltrDoc item6 = new DeltrDocImpl(3, 0.9f);
        item6.put("f0", 0.0, false);
        item6.put("f1", 0.838704766545679);

        // add the items in the trainQuery
        DeltrDoc[] docsArr = new DeltrDoc[]{item1, item2, item3, item4, item5, item6};
        trainQuery.setDocs(docsArr);

        // the trainQuery to the trainSet
        trainSet.add(trainQuery);

        // setup the parameters for the DELTR object
        double gamma = 1.0; // value of the gamma paramete
        int numberOfIterations = 10000; // number of iterations the training should run
        boolean shouldStandardize = true; // let's apply standardization to the features

        // create the Deltr object
        Deltr deltr = new Deltr(gamma, numberOfIterations, shouldStandardize);

        // train the model
        deltr.train(trainSet);
        // deltr.getOmega() -> [0.025225676596164703, 0.0798153206706047]
    }
}

```

### Use the model to rank 

Now, you can use the obtained model to rank some data.
```java
// let's create a sample prediction set
DeltrTopDocs preidictionSet = new DeltrTopDocsImpl(2); // 2 is the question ID

// let's create each prediction doc manually (docId, judgement/score)

// item 7
DeltrDoc item7 = new DeltrDocImpl(7, 0.9645f); // the current score is not really important
item7.put("f0", 0.0, false);
item7.put("f1", 0.9645);

// item 8
DeltrDoc item8 = new DeltrDocImpl(8, 0.9524f);
item8.put("f0", 0.0, false);
item8.put("f1", 0.9524);

// item 9
DeltrDoc item9 = new DeltrDocImpl(9, 0.9285f);
item9.put("f0", 0.0, false);
item9.put("f1", 0.9285);

// item 10
DeltrDoc item10 = new DeltrDocImpl(10, 0.8961f);
item10.put("f0", 0.0, false);
item10.put("f1", 0.8961);

// item 11
DeltrDoc item11 = new DeltrDocImpl(11, 0.8911f);
item11.put("f0", 1.0, true);
item11.put("f1", 0.8911);

// item 12
DeltrDoc item12 = new DeltrDocImpl(12, 0.8312f);
item12.put("f0", 1.0, true);
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
```

The library contains sufficient code documentation for each of the functions.

### Checking the model a bit deeper

You can check how the training of the model progressed using a special property called `log` (`getLog()`).
```java
for(TrainStep step : deltr.getLog()) {
    System.out.println(step);
}
//timestamp:1556475668410, lossStandard:5,999854, lossExposure:0,000000
//timestamp:1556475668492, lossStandard:5,999854, lossExposure:0,000000
//timestamp:1556475668548, lossStandard:5,999852, lossExposure:0,000000
// .
// .
// .
```
The `log` returns a list of objects from the `com.github.fairsearch.deltr.models.TrainStep` class. 
The class is a representation of the parameters in each step of the training. The parameters can be acquired with:

- `getTimestamp()`
- `getOmega()`
- `getLoss()` 
- `getLossStandard()`
- `getLossExposure()`

## Development

1. Clone this repository `git clone https://github.com/fair-search/fairsearchdeltr-python`
2. Change directory to the directory where you cloned the repository `cd WHERE_ITS_DOWNLOADED/fairsearchdeltr-python`
3. Use any IDE to work with the code

If you want to make your own builds you can do that with the Gradle wrapper:
- To make a JAR without the external dependencies: 
```
./gradlew clean jar
```
- To make a JAR with all external dependencies included:
```
./gradlew clean farJar
```

The output will go under `build/libs`.

## Testing

Just run:
```
./gradlew clean check
```
*Note*: Due to the high volume of the datasets, the tests take a bit *longer* time to execute. Also, because there is a *randomness* factor involved in 
the tests, it can happen that (very rarely) they fail sometimes.  

## Credits

The DELTR algorithm is described in this paper:

* Zehlike, Meike, and Carlos Castillo. "[Reducing Disparate Exposure in Ranking:
A Learning to Rank Approach](https://doi.org/10.1145/3132847.3132938)." arXiv preprint arXiv:1805.08716 (2018).

This library was developed by [Ivan Kitanovski](http://ivankitanovski.com/) based on the paper. See the 
[license](https://github.com/fair-search/fairsearchdeltr-java/blob/master/LICENSE) file for more information.

## See also

You can also see the [DELTR plug-in for ElasticSearch](https://github.com/fair-search/fairsearchdeltr-elasticsearch-plugin)
 and [DELTR Python library](https://github.com/fair-search/fairsearchdeltr-python).



 
  
